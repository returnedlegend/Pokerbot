"""Independent, deliberately untimed short-deck PLO5 strength probe.

This is an observation-only opponent, not a solver or a tournament submission.
Its 512 showdown particles and separate reference-equity simulations are more
expensive than the event bank permits. No production policy, fitted table,
opponent model or neural-network parameters are imported. Exact evaluation and
legal sampling are shared with the event-specific card engine.
"""
from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import equity_engine as engine
from game import Action, Street

SAMPLES = 512
REFERENCE_SAMPLES = 12
RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


@njit(cache=False)
def _reference_equities(hero, candidates, board, refs, seed, lut, choose):
    """Opponent strengths from draws independent of actual equity outcomes.

    Every candidate sees only the supplied historical board prefix. No later
    public card or actual opponent/future card enters this calculation. Hero's
    private cards are legitimate card-removal information for this observer.
    """
    result = np.empty(len(candidates), dtype=np.float64)
    state = np.uint64(seed) | np.uint64(1)
    known = np.empty(36, dtype=np.bool_)
    reference = np.empty(5, dtype=np.int32)
    full_board = np.empty(5, dtype=np.int32)
    full_board[:len(board)] = board
    for h in range(len(candidates)):
        known[:] = False
        for c in hero: known[c] = True
        for c in candidates[h]: known[c] = True
        for c in board: known[c] = True
        count = 0
        for c in range(36):
            if not known[c]: count += 1
        deck = np.empty(count, dtype=np.int32)
        k = 0
        for c in range(36):
            if not known[c]:
                deck[k] = c
                k += 1
        wins = 0.0
        for _ in range(refs):
            for i in range(10-len(board)):
                state = engine._next(state)
                j = i+int(state % np.uint64(count-i))
                deck[i], deck[j] = deck[j], deck[i]
            reference[:] = deck[:5]
            for i in range(5-len(board)):
                full_board[len(board)+i] = deck[5+i]
            theirs = engine._score(candidates[h], full_board, 5, lut, choose)
            other = engine._score(reference, full_board, 5, lut, choose)
            wins += 1.0 if theirs > other else (.5 if theirs == other else 0.0)
        result[h] = wins/refs
    return result


def _rank(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts)[inverse]-.5*counts[inverse])/len(values)


def _preflop_score(hands):
    """Independent unsupervised covariate; not our fitted preflop table."""
    ranks = hands >> 2
    suits = hands & 3
    score = ranks.sum(axis=1).astype(np.float64)
    for i in range(4):
        for j in range(i+1, 5):
            pair = ranks[:, i] == ranks[:, j]
            score += pair*(5.+.6*ranks[:, i])
            score += (suits[:, i] == suits[:, j])*(2.+.4*np.maximum(ranks[:, i], ranks[:, j]))
            score += (np.abs(ranks[:, i]-ranks[:, j]) == 1)*1.5
    # Excess duplicate ranks reduce the number of useful Omaha combinations.
    for rank in range(9):
        score -= np.maximum(0, (ranks == rank).sum(axis=1)-2)*9.
    return score


def new_half(config, memory):
    engine.warmup()
    memory.update(rng=random.Random(64028179), hand=-1, cache={}, seen=0,
                  raises=np.zeros(4), actions=np.zeros(4), faced=0., folds=0.,
                  last_obs=None, last_action=None, finished=-1)


def _observe(obs, memory):
    if memory['hand'] != obs.hand_index:
        memory['hand'] = obs.hand_index
        memory['seen'] = 0
        memory['cache'] = {}
        # Retain changing-style sensitivity without production model code.
        for field in ('raises', 'actions', 'faced', 'folds'):
            memory[field] *= .995
    for a in obs.action_history[memory['seen']:]:
        if not a.is_me:
            st = int(a.street)
            memory['actions'][st] += 1
            memory['raises'][st] += a.action in RAISES
    memory['seen'] = len(obs.action_history)
    memory['last_obs'] = obs


def _particles(obs, memory):
    key = tuple(obs.board)
    if key not in memory['cache']:
        rng = memory['rng']
        particles = engine.sample_equities(obs.hole_cards, obs.board, n=SAMPLES,
                                           seed=rng.randrange(1, 2**31))
        strengths = {0:_rank(_preflop_score(particles['villain_holes']))}
        for st in range(1, int(obs.street)+1):
            made = _rank(particles['villain_scores_by_street'][:, st-1])
            if st == int(Street.RIVER):
                strengths[st] = made
            else:
                ref = _reference_equities(
                    np.asarray(obs.hole_cards, np.int32), particles['villain_holes'],
                    np.asarray(obs.board[:st+2], np.int32), REFERENCE_SAMPLES,
                    np.uint64(rng.randrange(1, 2**31)), engine._LUT, engine._CHOOSE)
                # A small current-hand contribution stabilizes noisy reference
                # draws without discarding nut-flush and other redraw equity.
                strengths[st] = _rank(.80*ref+.20*made)
        memory['cache'][key] = particles, strengths
    return memory['cache'][key]


def _weights(obs, memory, strengths):
    weights = np.ones(SAMPLES)
    bets = [1, 2] if obs.is_button else [2, 1]
    street = Street.PREFLOP
    voluntary = False
    for a in obs.action_history:
        if a.street != street:
            bets = [0, 0]
            voluntary = False
            street = a.street
        who = 0 if a.is_me else 1
        facing = bets[who] < bets[1-who]
        if not a.is_me:
            st = int(a.street)
            aggression = (memory['raises'][st]+3.)/(memory['actions'][st]+12.)
            rank = strengths[st]
            if a.action in RAISES:
                # Hard range filtering with a nonzero bluff/background mixture.
                cutoff = np.clip(.81-.95*aggression, .12, .78)
                mixture = np.clip(.035+.45*aggression, .04, .38)
                weights *= np.where(rank >= cutoff, 1., mixture)
            elif a.action == Action.CHECK_CALL and facing and voluntary:
                weights *= np.where(rank >= .22, 1., .40)
            elif a.action == Action.CHECK_CALL and not facing and a.street != Street.PREFLOP:
                weights *= np.where(rank >= .85, .48, 1.)
        bets[who] += a.amount
        if a.action in RAISES:
            voluntary = True
    weights /= weights.sum()
    return weights


def _nut_blocker(obs):
    for suit in range(4):
        if sum((c & 3) == suit for c in obs.board) >= 3:
            if (32+suit) in obs.hole_cards:
                return True
    return False


def choose_action(obs, memory):
    if not memory:
        from game import AgentConfig
        new_half(AgentConfig(), memory)
    _observe(obs, memory)
    if len(obs.legal_actions) == 1:
        memory['last_action'] = obs.legal_actions[0]
        return memory['last_action']
    particles, strengths = _particles(obs, memory)
    weights = _weights(obs, memory, strengths)
    outcomes = particles['wins']
    equity = float(weights@outcomes)
    rank = strengths[int(obs.street)]
    call = min(obs.to_call, obs.my_stack)
    realization = 1. if obs.street == Street.RIVER or call == obs.my_stack else (.985 if obs.is_button else .95)
    call_ev = equity*realization*(obs.pot+call)-call
    best = Action.CHECK_CALL
    best_ev = call_ev
    if call and call_ev < 0 and Action.FOLD in obs.legal_actions:
        best, best_ev = Action.FOLD, 0.

    # The independent probe can trap strong hands against demonstrated bettors.
    aggression = (memory['raises'][int(obs.street)]+3.)/(memory['actions'][int(obs.street)]+12.)
    if not call and not obs.is_button and equity > .79 and aggression > .25:
        if memory['rng'].random() < .22:
            memory['last_action'] = Action.CHECK_CALL
            return Action.CHECK_CALL

    fold_rate = (memory['folds']+3.)/(memory['faced']+10.)
    for action in RAISES:
        if action not in obs.legal_actions:
            continue
        investment = obs.raise_to[action]-obs.my_bet
        response = min(obs.opp_stack, max(0, obs.raise_to[action]-obs.opp_bet))
        if response <= 0:
            continue
        price = response/max(1, obs.pot+investment+response)
        cutoff = np.clip(fold_rate+.50*(price-.25), .05, .75)
        # Smooth response policy avoids equating our hard betting-range filter
        # with the opponent's call threshold.
        continuation = .10+.90/(1+np.exp(-10*(rank-cutoff)))
        probability_call = float(weights@continuation)
        win_when_called = float((weights*continuation)@outcomes)
        called_equity = win_when_called/max(1e-12, probability_call)
        value = ((1-probability_call)*obs.pot +
                 win_when_called*(obs.pot+investment+response)-probability_call*investment)
        # Facing substantial action, require a real value edge before raising;
        # calling remains available with medium-strength draws and bluff catchers.
        minimum = .57 if obs.street != Street.RIVER else .61
        if call:
            minimum += .025*obs.raises_this_street
        polarized_bluff = (obs.street == Street.RIVER and not call and equity < .23
                           and _nut_blocker(obs) and fold_rate > .28
                           and memory['rng'].random() < .22)
        if called_equity < minimum and not polarized_bluff:
            continue
        if obs.street == Street.PREFLOP and equity < .535:
            continue
        risk = .018*(obs.pot+investment)
        if value > best_ev+risk or polarized_bluff:
            best, best_ev = action, value
    memory['last_action'] = best
    return best


def on_hand_end(result, memory):
    if memory.get('finished') == result.hand_index:
        return
    memory['finished'] = result.hand_index
    obs = memory.get('last_obs')
    if obs is None or obs.hand_index != result.hand_index:
        return
    # A terminal response to our final raise is absent from the last request.
    if memory.get('last_action') in RAISES:
        memory['faced'] += 1
        memory['folds'] += result.winner == 'me' and not result.went_to_showdown
