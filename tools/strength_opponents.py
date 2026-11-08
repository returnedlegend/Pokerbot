"""Independent stress policies using only legal observations and exact equity.

These probes do not import strategy, opponent_model, policy.json, or its fitted
preflop model. They are evaluation opponents, not tournament-strength claims.
The range probe includes draw information visible at the time of each action;
the pressure probe learns whether its bets make the other player fold.
"""
from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from equity_engine import sample_equities
from game import Action

PROFILES = ("draw_range", "adaptive_pressure")
RAISES = (Action.RAISE_HALF, Action.RAISE_POT)
STRAIGHTS = ((8, 0, 1, 2, 3),) + tuple(tuple(range(start, start+5)) for start in range(5))
STRAIGHT_MASKS = np.array([sum(1 << rank for rank in sequence) for sequence in STRAIGHTS])
PAIR_LEFT, PAIR_RIGHT = np.triu_indices(5, 1)


def midranks(values):
    """Percentiles with tied hands assigned the same midpoint rank."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    positions = np.cumsum(counts) - .5*counts
    return positions[inverse]/len(values)


def draw_features(holes, board):
    """Known-card flush/straight potential respecting exactly two hole cards.

    This is a small range covariate, never a replacement for sampled showdown
    equity. In particular it never inspects sampled future community cards.
    """
    count = len(holes)
    if not board or len(board) == 5:
        return np.zeros(count)
    holes = np.asarray(holes)
    board = np.asarray(board)
    ranks, suits = holes >> 2, holes & 3
    board_ranks, board_suits = board >> 2, board & 3
    draw = np.zeros(count)
    for suit in range(4):
        suited = suits == suit
        can_use_two = suited.sum(axis=1) >= 2
        number = int((board_suits == suit).sum())
        if number == 2:
            top = np.where(suited, ranks, -1).max(axis=1)/8
            draw = np.maximum(draw, can_use_two*(.70+.30*top))
        elif number == 1 and len(board) == 3:
            draw = np.maximum(draw, can_use_two*.20)
    # A straight draw needs two distinct hole ranks and two already visible
    # board ranks within one legal five-rank pattern. Duplicate ranks cannot
    # fill two positions, including the A6789 short-deck wheel.
    left, right = ranks[:, PAIR_LEFT, None], ranks[:, PAIR_RIGHT, None]
    eligible = ((left != right) & (((1 << left) & STRAIGHT_MASKS) != 0)
                & (((1 << right) & STRAIGHT_MASKS) != 0))
    board_mask = sum(1 << rank for rank in set(map(int, board_ranks)))
    visible_masks = board_mask & STRAIGHT_MASKS
    board_count = np.array([int(mask).bit_count() for mask in visible_masks])
    board_count = board_count-((visible_masks >> left) & 1)-((visible_masks >> right) & 1)
    straight_draw = np.any(eligible & (board_count >= 2), axis=(1, 2))
    draw = np.maximum(draw, straight_draw*.60)
    return draw


class StrengthOpponent:
    def __init__(self, profile):
        if profile not in PROFILES:
            raise ValueError(profile)
        self.profile = profile

    def new_half(self, config, memory):
        memory.update(rng=random.Random(169831), hand=-1, seen=0, cache={},
                      last_obs=None, last_action=None, fold_trials=12., folds=4.,
                      opponent_actions=20., opponent_raises=5.)

    def _observe(self, obs, memory):
        if memory["hand"] != obs.hand_index:
            memory.update(hand=obs.hand_index, seen=0, cache={}, last_obs=None, last_action=None)
        history = obs.action_history
        for index in range(memory["seen"], len(history)):
            action = history[index]
            if not action.is_me:
                memory["opponent_actions"] += 1
                memory["opponent_raises"] += action.action in RAISES
                if index and history[index-1].is_me and history[index-1].action in RAISES:
                    memory["fold_trials"] += 1
                    memory["folds"] += action.action == Action.FOLD
        memory["seen"] = len(history)

    def on_hand_end(self, result, memory):
        # A terminal fold has no subsequent choose_action callback. The result
        # reveals the winner, not the folded cards, and supplies this response.
        if (memory["last_action"] in RAISES and not result.went_to_showdown
                and result.winner == "me"):
            memory["fold_trials"] += 1
            memory["folds"] += 1

    def _range_equity(self, obs, memory):
        key = tuple(obs.board)
        if key not in memory["cache"]:
            sample = sample_equities(
                obs.hole_cards, obs.board, n=256,
                seed=memory["rng"].randrange(1, 2**31), include_history=True)
            ranks = {0: midranks(sample["villain_preflop"])}
            for street, visible in ((1, 3), (2, 4), (3, 5)):
                if len(obs.board) >= visible:
                    made = midranks(sample["villain_scores_by_street"][:, street-1])
                    draws = draw_features(sample["villain_holes"], obs.board[:visible])
                    ranks[street] = np.minimum(1., made+.22*draws*(1.-made))
            sample["probe_ranks"] = ranks
            memory["cache"][key] = sample
        sample = memory["cache"][key]
        n = len(sample["wins"])
        weights = np.ones(n)
        ranks = sample["probe_ranks"]
        aggression = memory["opponent_raises"]/memory["opponent_actions"]
        for action in obs.action_history:
            if action.is_me:
                continue
            strength = ranks[int(action.street)]
            if action.action in RAISES:
                # Aggressive opponents retain more weak hands; passive ones'
                # raises concentrate near the top without an absolute cutoff.
                floor = min(.35, .06+aggression)
                likelihood = floor + (1.-floor)*strength**(3.0 if action.action == Action.RAISE_POT else 2.0)
                weights *= likelihood
            elif action.action == Action.CHECK_CALL and action.amount > obs.big_blind:
                weights *= .25+.75*strength
        weights = np.maximum(weights, .005)
        return float(np.dot(weights, sample["wins"])/weights.sum())

    def choose_action(self, obs, memory):
        self._observe(obs, memory)
        equity = self._range_equity(obs, memory)
        price = obs.to_call/max(1, obs.pot+obs.to_call)
        pressure = self.profile == "adaptive_pressure"
        margin = .018 if pressure else .035
        if obs.to_call <= obs.big_blind:
            margin = -.035
        action = Action.CHECK_CALL
        if obs.to_call and equity < price+margin and Action.FOLD in obs.legal_actions:
            action = Action.FOLD
        else:
            fold_rate = memory["folds"]/memory["fold_trials"]
            # Value bets become selective after re-raises. Pressure probes
            # punish excess folding, but always retain a real value range.
            value_line = (.59 if pressure else .64)+.035*obs.raises_this_street
            if equity > value_line:
                preferred = Action.RAISE_POT if equity > .72 else Action.RAISE_HALF
                if preferred in obs.legal_actions:
                    action = preferred
            elif pressure and obs.raises_this_street <= 1:
                bluff_rate = .10 + max(0., fold_rate-.32)*.9
                if obs.to_call:
                    bluff_rate *= .45
                # Legal bluff EV uses the actual event bet size, not a fixed
                # no-limit formula. A weak hand bluffs only with learned folds.
                if Action.RAISE_HALF in obs.legal_actions:
                    added = obs.raise_to[Action.RAISE_HALF]-obs.my_bet
                    risk = max(0, added-obs.to_call)
                    threshold = risk/max(1, risk+obs.pot)
                    if fold_rate > threshold and memory["rng"].random() < min(.48, bluff_rate):
                        action = Action.RAISE_HALF
        memory["last_obs"] = obs
        memory["last_action"] = action
        return action
