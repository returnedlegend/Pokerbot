"""Inference lookup symmetry and independent Monte Carlo calibration."""
from itertools import permutations
from pathlib import Path
import random

import numpy as np

from equity_engine import sample_equities
from strategy import _lookup_preflop


def test_runtime_preflop_lookup_respects_all_suit_renamings():
    rng = random.Random(89201)
    hands = [rng.sample(range(36), 5) for _ in range(40)]
    for hand in hands:
        equivalents = [[(card & ~3) + p[card & 3] for card in hand]
                       for p in permutations(range(4))]
        values = _lookup_preflop(equivalents)
        assert np.all(values == _lookup_preflop(hand))
        assert _lookup_preflop(hand[::-1]) == _lookup_preflop(hand)


def test_preflop_fit_has_correct_global_mean_and_independent_calibration():
    path = Path(__file__).resolve().parents[1] / 'preflop_model.npz'
    with np.load(path) as data:
        assert data['equity'].shape == (376992,)
        assert abs(float(data['equity'].mean()) - .5) < .003
        max_se = float(data['standard_error'].max())
        assert int(data['samples_per_class']) >= 2048
    rng = random.Random(20261110)
    for i in range(8):
        hand = rng.sample(range(36), 5)
        # Different seeds from training; compare to a much larger live sample.
        wins = sample_equities(hand, [], n=100000, seed=109281+i)['wins']
        fitted, reference = float(_lookup_preflop(hand)), float(wins.mean())
        combined_se = np.sqrt(max_se**2 + float(wins.var())/len(wins))
        assert abs(fitted-reference) < 4*combined_se
