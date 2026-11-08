"""Fixed observation-only synthetic opponents for the overnight search.

These policies are independent implementations, not real entrants or Astra.
They share only the exact card scorer/legal Monte Carlo sampler with the bot.
No production strategy, opponent model, fitted lookup or neural weights are
imported. Sampled future cards never determine betting-range likelihoods.

API: ``PROFILES`` and ``ChallengeOpponent(profile, seed=830907)`` with the
standard new_half/choose_action/on_hand_end callbacks. The search runner can
capture this entire source and append callbacks for a literal profile, avoiding
a mutable transitive import. Freeze the source/configuration before tuning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equity_engine import sample_equities
from game import Action, Street

SUITE_VERSION = 1
PROFILES = ("challenge_value", "challenge_polarized", "challenge_checkraise",
            "challenge_adaptive", "challenge_station", "challenge_shifting")
RAISES = (Action.RAISE_HALF, Action.RAISE_POT)
STRAIGHTS = ((8, 0, 1, 2, 3),) + tuple(tuple(range(start, start+5)) for start in range(5))
_PAIR_LEFT, _PAIR_RIGHT = np.triu_indices(5, 1)
_POPCOUNTS = np.array([value.bit_count() for value in range(512)], dtype=np.int8)


@dataclass(frozen=True)
class Settings:
    samples: int
    call_margin: float
    value_floor: float
    raise_caution: float
    bluff_frequency: float
    trap_frequency: float
    decay: float = .994


SETTINGS = {
    "challenge_value": Settings(384, .025, .575, .020, .025, .08),
    "challenge_polarized": Settings(256, .012, .640, .015, .210, .16),
    "challenge_checkraise": Settings(256, .015, .610, .018, .120, .58),
    "challenge_adaptive": Settings(192, .008, .580, .015, .180, .12),
    "challenge_station": Settings(128, -.200, .760, .030, .000, .02),
    "challenge_shifting": Settings(256, .015, .610, .018, .120, .20),
}


def suite_manifest():
    """Record fixed profiles separately from the runner's captured-source hash."""
    return {"version": SUITE_VERSION, "default_seed": 830907,
            "profiles": {name: asdict(SETTINGS[name]) for name in PROFILES},
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Synthetic legal-observation stress policies; no human/LLM or equilibrium claim"}


def _midranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts)[inverse]-.5*counts[inverse])/len(values)


def _conditional_percentiles(values, weights):
    _, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    return (np.cumsum(mass)[inverse]-.5*mass[inverse])/mass.sum()


def _curve(values, center, slope=11.):
    return 1./(1.+np.exp(np.clip((center-values)*slope, -30., 30.)))


def _preflop_values(hands):
    """Simple private-card texture, independently defined rather than fitted."""
    ranks, suits = hands >> 2, hands & 3
    value = .17*ranks.sum(axis=1).astype(np.float64)
    counts = np.stack([(ranks == r).sum(axis=1) for r in range(9)], axis=1)
    for rank in range(9):
        value += (counts[:, rank] >= 2)*(1.7+.32*rank)
        value -= np.maximum(0, counts[:, rank]-2)*3.8
    for suit in range(4):
        suited = suits == suit
        count = suited.sum(axis=1)
        value += (count >= 2)*1.1-(count >= 4)*.8
        value += ((ranks == 8) & suited).any(axis=1)*(count >= 2)*1.4
    for straight in STRAIGHTS:
        distinct = (counts[:, straight] > 0).sum(axis=1)
        value += np.maximum(0, distinct-2)*.18
    return value


def _draw_values(hands, board):
    """Draw texture from a historical visible board prefix only.

    Draws require exactly two distinct private ranks for straights or at least
    two private cards of a suit for flushes. This covariate is not claimed to be
    calibrated equity; legal full-board rollouts supply the equity outcomes.
    """
    result = np.zeros(len(hands))
    if len(board) not in (3, 4):
        return result
    ranks, suits = hands >> 2, hands & 3
    board_ranks = {c >> 2 for c in board}
    for suit in range(4):
        same = suits == suit
        count = sum((c & 3) == suit for c in board)
        can_flush = same.sum(axis=1) >= 2
        high = np.where(same, ranks, -1).max(axis=1)/8.
        if count == 2:
            result = np.maximum(result, can_flush*(.70+.30*high))
        elif count == 1 and len(board) == 3:
            result = np.maximum(result, can_flush*(.12+.08*high))
    left, right = ranks[:, _PAIR_LEFT], ranks[:, _PAIR_RIGHT]
    pairs = (1 << left) | (1 << right)
    public_mask = sum(1 << rank for rank in board_ranks)
    for straight in STRAIGHTS:
        window = sum(1 << rank for rank in straight)
        eligible = (left != right) & ((pairs & window) == pairs)
        hits = _POPCOUNTS[public_mask & (window ^ pairs)]
        # Two visible board ranks complete with one card; one visible rank
        # can complete with both remaining cards on the flop.
        result = np.maximum(result, np.any(eligible & (hits >= 2), axis=1)*.50)
        if len(board) == 3:
            result = np.maximum(result, np.any(eligible & (hits == 1), axis=1)*.10)
    return result


def _contexts(obs):
    """Replay blind posting and visible action amounts without private state."""
    small = max(1, obs.big_blind//2)
    bets = [small, obs.big_blind] if obs.is_button else [obs.big_blind, small]
    street = Street.PREFLOP
    voluntary = [False, False]
    result = []
    for action in obs.action_history:
        if action.street != street:
            bets, voluntary, street = [0, 0], [False, False], action.street
        actor = 0 if action.is_me else 1
        owed = max(0, bets[1-actor]-bets[actor])
        result.append((bool(owed), bool(owed and voluntary[1-actor])))
        bets[actor] = action.street_total
        if action.action in RAISES:
            voluntary[actor] = True
    return result


def _blockers(obs):
    """Return a bluff preference, not a claim that one card makes a flush."""
    suited_aces = 0.
    for suit in range(4):
        public = sum((c & 3) == suit for c in obs.board)
        private = sum((c & 3) == suit for c in obs.hole_cards)
        if public >= 2 and 32+suit in obs.hole_cards:
            suited_aces = max(suited_aces, 1. if private == 1 else .55)
    paired = {c >> 2 for c in obs.board if sum((b >> 2) == (c >> 2) for b in obs.board) >= 2}
    full_house_block = any((c >> 2) in paired for c in obs.hole_cards)
    return min(1., suited_aces+.20*full_house_block)


class ChallengeOpponent:
    def __init__(self, profile, *, seed=830907):
        if profile not in PROFILES:
            raise ValueError(f"Unknown challenge profile: {profile}")
        self.profile = profile
        self.seed = int(seed)

    def new_half(self, config, memory):
        # The public half configuration carries no deal seed or hidden cards.
        digest = hashlib.blake2s(f"challenge-v{SUITE_VERSION}:{self.seed}:{self.profile}".encode(), digest_size=8).digest()
        memory.clear()
        memory.update(rng=random.Random(int.from_bytes(digest, "big")), hand=-1,
                      seen=0, cache={}, actions=np.zeros((4, 2, 3)),
                      fold_trials=np.zeros((4, 2)), folds=np.zeros((4, 2)),
                      last_obs=None, last_action=None, finished=-1,
                      phase_length=max(20, config.num_hands//4), diagnostics={})

    def effective_profile(self, obs, memory):
        if self.profile != "challenge_shifting":
            return self.profile
        sequence = ("challenge_value", "challenge_polarized", "challenge_station", "challenge_checkraise")
        return sequence[min(3, obs.hand_index//memory["phase_length"])]

    def _observe(self, obs, memory):
        if memory["hand"] != obs.hand_index:
            elapsed = max(0, obs.hand_index-max(0, memory["hand"]))
            factor = SETTINGS[self.profile].decay**elapsed
            for name in ("actions", "fold_trials", "folds"):
                memory[name] *= factor
            memory.update(hand=obs.hand_index, seen=0, cache={})
        contexts = _contexts(obs)
        for index in range(memory["seen"], len(obs.action_history)):
            action = obs.action_history[index]
            if action.is_me:
                continue
            facing, _ = contexts[index]
            bucket = 2 if action.action in RAISES else 0 if action.action == Action.FOLD else 1
            memory["actions"][int(action.street), int(facing), bucket] += 1.
            if index:
                previous = obs.action_history[index-1]
                if previous.is_me and previous.action in RAISES and previous.street == action.street:
                    size = int(previous.action == Action.RAISE_POT)
                    memory["fold_trials"][int(action.street), size] += 1.
                    memory["folds"][int(action.street), size] += action.action == Action.FOLD
        memory["seen"] = len(obs.action_history)
        memory["last_obs"] = obs

    def on_hand_end(self, result, memory):
        if result.hand_index <= memory["finished"]:
            return
        memory["finished"] = result.hand_index
        obs = memory.get("last_obs")
        if obs is None or obs.hand_index != result.hand_index:
            return
        last = memory.get("last_action")
        # An opponent's terminal call/fold is absent from our last request.
        # Only public winner/showdown flags are read, never revealed hole cards.
        if last in RAISES:
            size = int(last == Action.RAISE_POT)
            st = int(obs.street)
            folded = result.winner == "me" and not result.went_to_showdown
            memory["fold_trials"][st, size] += 1.
            memory["folds"][st, size] += folded
            memory["actions"][st, 1, 0 if folded else 1] += 1.

    def _particles(self, obs, memory, samples):
        key = (tuple(obs.hole_cards), tuple(obs.board), samples)
        if key not in memory["cache"]:
            particles = sample_equities(obs.hole_cards, obs.board, n=samples,
                                       seed=memory["rng"].randrange(1, 2**31), include_history=True)
            ranks = {0: _midranks(_preflop_values(particles["villain_holes"]))}
            for st in range(1, int(obs.street)+1):
                made = _midranks(particles["villain_scores_by_street"][:, st-1])
                visible = obs.board[:st+2]
                draw = _draw_values(particles["villain_holes"], visible)
                ranks[st] = _midranks(made+.30*draw*(1.-made))
            memory["cache"][key] = particles, ranks
        return memory["cache"][key]

    def _posterior(self, obs, memory, particles, ranks):
        log_weights = np.zeros(len(particles["wins"]))
        contexts = _contexts(obs)
        for index, action in enumerate(obs.action_history):
            if action.is_me:
                continue
            facing, voluntary = contexts[index]
            row = memory["actions"][int(action.street), int(facing)]
            aggression = (row[2]+3.)/(row.sum()+12.)
            rank = ranks[int(action.street)]
            cutoff = np.clip(.83-.85*aggression+.07*facing, .22, .84)
            background = np.clip(.06+.55*aggression, .08, .40)
            curve = background+(1.-background)*_curve(rank, cutoff)
            if action.action in RAISES:
                if action.action == Action.RAISE_POT:
                    curve = background*.8+(1.-background*.8)*_curve(rank, min(.92, cutoff+.035))
                likelihood = curve
            elif action.action == Action.CHECK_CALL and voluntary:
                likelihood = .20+.80*_curve(rank, .24)
            elif action.action == Action.CHECK_CALL and not facing and action.street != Street.PREFLOP:
                # Strong checks retain a trap mixture rather than being cut.
                likelihood = .18+.82*(1.-np.minimum(.96, curve*aggression/.40))
            else:
                likelihood = np.ones(len(log_weights))
            log_weights += np.log(np.maximum(likelihood, 1e-9))
        weights = np.exp(log_weights-log_weights.max())
        weights /= weights.sum()
        return weights

    def _fold_estimate(self, obs, memory, action):
        size = int(action == Action.RAISE_POT)
        st = int(obs.street)
        # Shrink street/size evidence toward all observed replies at this size.
        total_trials = memory["fold_trials"][:, size].sum()
        pooled = (memory["folds"][:, size].sum()+3.2+size)/(total_trials+10.)
        trials = memory["fold_trials"][st, size]
        mean = (memory["folds"][st, size]+10.*pooled)/(trials+10.)
        error = np.sqrt(mean*(1.-mean)/(trials+11.))
        return float(mean), float(error), float(trials)

    def choose_action(self, obs, memory):
        if not memory:
            from game import AgentConfig
            self.new_half(AgentConfig(), memory)
        self._observe(obs, memory)
        if len(obs.legal_actions) == 1:
            memory["last_action"] = obs.legal_actions[0]
            return obs.legal_actions[0]
        profile = self.effective_profile(obs, memory)
        settings = SETTINGS[profile]
        particles, ranks = self._particles(obs, memory, settings.samples)
        weights = self._posterior(obs, memory, particles, ranks)
        outcomes = particles["wins"]
        equity = float(weights@outcomes)
        call = min(obs.to_call, obs.my_stack)
        terminal = obs.street == Street.RIVER or call == obs.my_stack
        realization = 1. if terminal else (.990 if obs.is_button else .955)
        margin = settings.call_margin if not terminal or profile == "challenge_station" else 0.
        if call <= obs.big_blind and not terminal:
            margin = min(margin, -.025)
        call_ev = (equity*realization-margin)*(obs.pot+call)-call
        best = Action.CHECK_CALL
        best_ev = call_ev
        if call and call_ev < 0 and Action.FOLD in obs.legal_actions:
            best, best_ev = Action.FOLD, 0.

        street = int(obs.street)
        rank = _conditional_percentiles(ranks[street], weights)
        checked = any(a.street == obs.street and a.is_me and a.action == Action.CHECK_CALL and a.amount == 0
                      for a in obs.action_history)
        check_raise = bool(checked and call and obs.raises_this_street == 1)
        trap = (not call and not obs.is_button and obs.street != Street.PREFLOP
                and equity > .74 and memory["rng"].random() < settings.trap_frequency)
        blocker = _blockers(obs)
        coin = memory["rng"].random()
        for action in RAISES:
            if action not in obs.legal_actions or trap:
                continue
            investment = obs.raise_to[action]-obs.my_bet
            response = min(obs.opp_stack, max(0, obs.raise_to[action]-obs.opp_bet))
            if response <= 0:
                continue
            fold_mean, fold_error, trials = self._fold_estimate(obs, memory, action)
            fold_target = np.clip(fold_mean+.20*(response/max(1, obs.pot+investment+response)-.25), .03, .85)
            continuation = .05+.95*_curve(rank, fold_target, slope=13.)
            probability_call = float(weights@continuation)
            called_win = float((weights*continuation)@outcomes)
            called_equity = called_win/max(1e-12, probability_call)
            bet_realization = 1. if terminal or investment == obs.my_stack else realization
            ev = ((1.-probability_call)*obs.pot+called_win*bet_realization*(obs.pot+investment+response)
                  -probability_call*investment)
            value_floor = settings.value_floor+.018*obs.raises_this_street
            if action == Action.RAISE_POT:
                value_floor += .025
            if profile == "challenge_checkraise" and check_raise:
                value_floor -= .045
            value = called_equity >= value_floor and (obs.street != Street.PREFLOP or equity >= .53)
            pressure = False
            if profile == "challenge_adaptive":
                # Wager only where recent folds can cover the actual legal cost;
                # a conservative estimate stops escalating against stations.
                conservative_fold = max(0., fold_mean-.65*fold_error)
                frequency = min(.62, settings.bluff_frequency+max(0., fold_mean-.32)*.95)
                bluff_ev = conservative_fold*obs.pot-(1.-conservative_fold)*investment
                pressure = (bluff_ev > max(0., best_ev) and trials >= 6 and coin < frequency)
            elif profile in ("challenge_polarized", "challenge_checkraise"):
                eligible = (obs.street != Street.PREFLOP and equity < .43 and obs.raises_this_street <= 1)
                frequency = settings.bluff_frequency*(.35+.65*blocker)
                if profile == "challenge_checkraise":
                    frequency *= 1.4 if check_raise else .35
                pressure = eligible and coin < frequency and (not call or check_raise)
                # Polarized river bets use the larger available size; checkraise
                # semibluffs can use either legal size after an inducing check.
                if profile == "challenge_polarized" and obs.street == Street.RIVER and action != Action.RAISE_POT:
                    pressure = False
            elif profile == "challenge_value":
                pressure = (not call and obs.street == Street.RIVER and equity < .25
                            and blocker > .8 and coin < settings.bluff_frequency)
            risk = settings.raise_caution*(obs.pot+investment)
            if value and ev > best_ev+risk:
                best, best_ev = action, ev
            elif pressure and not value:
                # A fixed mixed bluff component deliberately has no guarantee of
                # positive EV. It tests whether a candidate can handle value and
                # bluffs sharing the same betting size and public situation.
                best, best_ev = action, ev
        memory["last_action"] = best
        memory["diagnostics"] = {"profile": profile, "equity": equity,
                                 "uniform_equity": float(outcomes.mean()),
                                 "samples": len(outcomes), "check_raise_opportunity": check_raise}
        return best if best in obs.legal_actions else Action.CHECK_CALL


if __name__ == "__main__":
    print(json.dumps(suite_manifest(), indent=2))
