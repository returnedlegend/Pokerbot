"""Exact-event ranking, sampling and reproducibility checks."""
from itertools import combinations
import random

import numpy as np
import pytest

from evaluator import omaha_score, score_five
from equity_engine import sample_equities, score_hand


def test_exact_score_matches_official_thousands():
    rng = random.Random(192039)
    for _ in range(2500):
        cards = rng.sample(range(36), 10)
        hole, board = cards[:5], cards[5:]
        assert score_hand(hole, board) == omaha_score(hole, board)
    for _ in range(100):
        cards = rng.sample(range(36), 9)
        for size in (3, 4):
            hole, board = cards[:5], cards[5:5 + size]
            expected = max(score_five(h + b) for h in combinations(hole, 2)
                           for b in combinations(board, 3))
            assert score_hand(hole, board) == expected


def test_sample_legality_repeatability_and_history():
    hole, board = [0, 5, 10, 15, 20], [25, 30, 35, 4, 9]
    a = sample_equities(hole, board, 512, 792)
    b = sample_equities(hole, board, 512, 792)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert set(a['wins']).issubset({0., .5, 1.})
    for i, villain in enumerate(a['villain_holes']):
        assert len(set(list(villain) + hole + board)) == 15
        for size in (3, 4, 5):
            assert a['villain_scores_by_street'][i, size - 3] == score_hand(villain, board[:size])
        expected = .5 if a['hero_score'] == a['villain_scores'][i] else float(a['hero_score'] > a['villain_scores'][i])
        assert a['wins'][i] == expected


def test_monte_carlo_calibration_against_independent_reference():
    # Large samples, separately seeded pure-Python reference, broad enough
    # tolerance to detect deck/ranking mistakes without fragile luck assertions.
    rng = random.Random(39)
    for size in (0, 3, 4, 5):
        known = rng.sample(range(36), 5 + size)
        hole, board = known[:5], known[5:]
        deck = sorted(set(range(36)) - set(known))
        actual = float(sample_equities(hole, board, 10000, 922 + size)['wins'].mean())
        total = 0.
        for _ in range(2000):
            dealt = rng.sample(deck, 10 - size)
            villain = dealt[:5]
            complete = board + dealt[5:]
            me, them = omaha_score(hole, complete), omaha_score(villain, complete)
            total += 1. if me > them else (.5 if me == them else 0.)
        assert abs(actual - total / 2000) < .055


@pytest.mark.parametrize('hole,board', [([0]*5, []), ([0,1,2,3,36], []),
                                      ([0,1,2,3,4], [0,5,6]),
                                      ([0,1,2,3,4], [5,6])])
def test_invalid_cards_are_rejected(hole, board):
    with pytest.raises(ValueError):
        sample_equities(hole, board)
