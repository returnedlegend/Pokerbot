"""Equity-network information boundaries, symmetry and rollout calibration."""
from itertools import permutations

import numpy as np
import pytest

import equity_engine as engine
import equity_network as network


def test_all_suit_relabelings_and_private_order_have_identical_features():
    rng = np.random.default_rng(91483)
    engine.warmup()
    for size in (3, 4):
        cards = rng.permutation(36)[:5 + size]
        hand, board = cards[:5], cards[5:]
        expected = network.batchpredict(hand.reshape(1, 5), board)[0]
        for suits in permutations(range(4)):
            changed = np.array([(c & ~3) + suits[c & 3] for c in cards], np.int32)
            actual = network.batchpredict(changed[:5][::-1].reshape(1, 5), changed[5:][::-1])[0]
            assert actual == pytest.approx(expected, abs=1e-7)


def test_batch_scores_equal_individual_predictions_and_are_probabilities():
    rng = np.random.default_rng(812731)
    board = np.array([4, 18, 28], dtype=np.int32)
    remaining = np.array([c for c in range(36) if c not in board])
    hands = np.array([rng.choice(remaining, size=5, replace=False) for _ in range(31)], np.int32)
    together = network.batchpredict(hands, board)
    scores = np.array([engine.score_hand(hand, board) for hand in hands], np.int32)
    np.testing.assert_array_equal(together, network.batchpredict(hands, board, scores=scores))
    separate = np.array([network.batchpredict(hand.reshape(1, 5), board)[0] for hand in hands])
    assert together.shape == (31,)
    assert np.all((together >= 0) & (together <= 1))
    np.testing.assert_allclose(together, separate, atol=4e-7)


def test_api_accepts_only_current_flop_or_turn():
    hand = np.array([[0, 5, 10, 15, 20]], np.int32)
    with pytest.raises(ValueError, match="flop or turn"):
        network.batchpredict(hand, [24, 29, 30, 33, 34])
    with pytest.raises(ValueError, match="flop or turn"):
        network.batchpredict(hand, [])
    assert network.batchpredict(np.empty((0, 5), np.int32), [24, 29, 34]).shape == (0,)


def test_export_is_calibrated_on_independent_high_sample_positions():
    report = network.metadata()
    assert report["validation_positions"] >= 1000
    assert report["validation_rollouts"] >= 1024
    assert report["training_seed"] != report["validation_seed"]
    assert report["neural"]["rmse"] < .075
    assert report["neural"]["rmse"] < .75 * report["calibrated_made_hand"]["rmse"]
    assert report["neural"]["rank_correlation"] > .9
    fixed = report["fixed_board_validation"]
    assert fixed["boards"] >= 8
    assert fixed["neural_mean_rank_correlation"] > .95
    assert fixed["neural_mean_rank_correlation"] > fixed["made_hand_mean_rank_correlation"] + .05


def test_fresh_monte_carlo_labels_not_used_in_training():
    rng = np.random.default_rng(928374713)
    predicted, actual = [], []
    for i in range(32):
        size = 3 + i % 2
        cards = rng.permutation(36)[:5 + size]
        hand, board = cards[:5], cards[5:]
        predicted.append(float(network.batchpredict(hand.reshape(1, 5), board)[0]))
        actual.append(float(engine.sample_equities(hand, board, n=4096,
                            seed=7728173 + i, include_history=False)["wins"].mean()))
    difference = np.asarray(predicted) - actual
    assert np.sqrt(np.mean(difference ** 2)) < .09
    assert abs(difference.mean()) < .04
