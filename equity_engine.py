"""Fast equity model for the event's exact Short Deck PLO5 rules.

Monte Carlo model adapted conceptually from PyPokerEngine's HonestPlayer;
see THIRD_PARTY_NOTICES.md. This is not pretrained Hold'em neural weights.
All card rankings come from the unmodified event evaluator. Numba compiles
locally at process startup without writing a cache (the event FS is read-only).
"""
from itertools import combinations
from math import comb

import numpy as np
from numba import njit

_LUT = None
_CHOOSE = np.zeros((36, 6), dtype=np.int64)
for _c in range(36):
    for _k in range(1, 6):
        _CHOOSE[_c, _k] = comb(_c, _k) if _c >= _k else 0


@njit(cache=False)
def _score5(a, b, c, d, e, lut, choose):
    # Sorting network, 9 compare/exchanges rather than allocating/sorting.
    if a > b: a, b = b, a
    if d > e: d, e = e, d
    if c > e: c, e = e, c
    if c > d: c, d = d, c
    if b > e: b, e = e, b
    if a > d: a, d = d, a
    if a > c: a, c = c, a
    if b > d: b, d = d, b
    if b > c: b, c = c, b
    return lut[a + choose[b, 2] + choose[c, 3] + choose[d, 4] + choose[e, 5]]


@njit(cache=False)
def _score(hole, board, board_len, lut, choose):
    best = 0
    for i in range(4):
        for j in range(i + 1, 5):
            for a in range(board_len - 2):
                for b in range(a + 1, board_len - 1):
                    for c in range(b + 1, board_len):
                        v = _score5(hole[i], hole[j], board[a], board[b], board[c], lut, choose)
                        if v > best:
                            best = v
    return best


@njit(cache=False)
def _preflop(hole):
    """Smooth opponent-range covariate, not a calibrated win probability."""
    ranks = np.zeros(9, dtype=np.int32)
    suits = np.zeros(4, dtype=np.int32)
    value = 0.0
    for c in hole:
        rank = c >> 2
        ranks[rank] += 1
        suits[c & 3] += 1
        value += rank / 40.0
    pairs = 0
    for r in range(9):
        if ranks[r] >= 2:
            value += .10 + .012 * r
            pairs += 1
        if ranks[r] >= 3:
            value -= .15 * (ranks[r] - 2)
    for s in range(4):
        if suits[s] >= 2:
            value += .10
            # Nut-suit potential matters much more than low suitedness.
            for c in hole:
                if c & 3 == s and c >> 2 == 8:
                    value += .10
        if suits[s] >= 4:
            value -= .04 * (suits[s] - 3)
    for r in range(8):
        if ranks[r] and ranks[r + 1]:
            value += .045
    if ranks[8] and ranks[0]:
        value += .025
    # Keep upper tail distinct: clipping at 1 collapses roughly a quarter of
    # hands and makes strong preflop ranges impossible to distinguish.
    return max(0.0, value)


@njit(cache=False)
def _next(state):
    state ^= state << np.uint64(13)
    state ^= state >> np.uint64(7)
    state ^= state << np.uint64(17)
    return state


@njit(cache=False)
def _sample(hole, board, n, seed, lut, choose, include_history):
    known = np.zeros(36, dtype=np.bool_)
    for c in hole: known[c] = True
    for c in board: known[c] = True
    deck = np.empty(36 - 5 - len(board), dtype=np.int32)
    pos = 0
    for c in range(36):
        if not known[c]:
            deck[pos] = c
            pos += 1
    full_board = np.empty(5, dtype=np.int32)
    full_board[:len(board)] = board
    villain = np.empty(5, dtype=np.int32)
    wins = np.empty(n, dtype=np.float32)
    scores = np.empty(n, dtype=np.int32)
    preflop = np.empty(n, dtype=np.float32)
    villain_holes = np.empty((n, 5), dtype=np.int32)
    street_scores = np.zeros((n, 3), dtype=np.int32)
    hero_score = _score(hole, full_board, len(board), lut, choose)
    state = np.uint64(seed) | np.uint64(1)
    take = 10 - len(board)
    for sample in range(n):
        # Partial Fisher-Yates; each sample starts from a valid full deck
        # permutation, so leaving the permutation in place is unbiased.
        for i in range(take):
            state = _next(state)
            j = i + int(state % np.uint64(len(deck) - i))
            deck[i], deck[j] = deck[j], deck[i]
        for i in range(5): villain[i] = deck[i]
        for i in range(5 - len(board)): full_board[len(board) + i] = deck[5 + i]
        villain_score = _score(villain, full_board, 5, lut, choose)
        my_score = hero_score if len(board) == 5 else _score(hole, full_board, 5, lut, choose)
        wins[sample] = 1.0 if my_score > villain_score else (.5 if my_score == villain_score else 0.0)
        scores[sample] = villain_score if len(board) == 5 else _score(villain, full_board, len(board), lut, choose)
        preflop[sample] = _preflop(villain)
        villain_holes[sample] = villain
        if len(board) >= 3:
            street_scores[sample, len(board) - 3] = scores[sample]
        if include_history:
            for size in range(3, len(board)):
                street_scores[sample, size - 3] = _score(villain, full_board, size, lut, choose)
    return wins, scores, preflop, hero_score, villain_holes, street_scores


def warmup():
    """Build compact ranking table and compile all runtime paths (<15s target)."""
    global _LUT
    if _LUT is not None:
        return
    from evaluator import get_table
    table = get_table()
    lut = np.empty(comb(36, 5), dtype=np.int32)
    for cards in combinations(range(36), 5):
        a, b, c, d, e = cards
        index = a + _CHOOSE[b, 2] + _CHOOSE[c, 3] + _CHOOSE[d, 4] + _CHOOSE[e, 5]
        mask = (1 << a) | (1 << b) | (1 << c) | (1 << d) | (1 << e)
        lut[index] = table[mask]
    _LUT = lut
    _sample(np.array([0, 1, 2, 3, 4], np.int32), np.array([5, 6, 7], np.int32), 1, np.uint64(1), _LUT, _CHOOSE, True)
    _score(np.array([0, 1, 2, 3, 4], np.int32), np.array([5, 6, 7, 8, 9], np.int32), 5, _LUT, _CHOOSE)
    _preflop(np.array([0, 1, 2, 3, 4], np.int32))


def _validate(hole, board):
    if len(hole) != 5 or len(board) not in (0, 3, 4, 5):
        raise ValueError("expected 5 hole cards and 0, 3, 4 or 5 board cards")
    cards = list(hole) + list(board)
    if len(set(cards)) != len(cards) or any(int(c) != c or c < 0 or c >= 36 for c in cards):
        raise ValueError("cards must be distinct integers in 0..35")


def score_hand(hole, board):
    """Exact best 2-hole + 3-board hand; returns 0 before the flop."""
    _validate(hole, board)
    warmup()
    return int(_score(np.asarray(hole, np.int32), np.asarray(board, np.int32), len(board), _LUT, _CHOOSE))


def preflop_strength(hole):
    _validate(hole, [])
    warmup()
    return float(_preflop(np.asarray(hole, np.int32)))


def sample_equities(hole, board, n=256, seed=1, include_history=True):
    """Return paired equity outcomes and current villain-range covariates.

    `wins` gives 1/.5/0 per sampled opponent+runout. `villain_scores`
    gives that opponent's current made-hand score (0 preflop), allowing a
    caller to weight by the empirical strength percentile. `villain_preflop`
    is a simple fixed hole-card covariate for preflop range weighting.
    None of these calculations use the actual unseen opponent cards.
    """
    _validate(hole, board)
    if not 1 <= n <= 100000:
        raise ValueError("n must be between 1 and 100000")
    warmup()
    wins, scores, preflop, hero_score, holes, streets = _sample(
        np.asarray(hole, np.int32), np.asarray(board, np.int32), int(n),
        np.uint64(int(seed) & ((1 << 64) - 1)), _LUT, _CHOOSE, bool(include_history))
    return {"wins": wins, "villain_scores": scores,
            "villain_preflop": preflop, "hero_score": int(hero_score),
            "villain_holes": holes, "villain_scores_by_street": streets}
