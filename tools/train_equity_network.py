#!/usr/bin/env python3
"""Train a flop/turn equity surrogate from the official evaluator's rollouts.

Run from the project root, e.g.:
 OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/train_equity_network.py
Only this offline script imports scikit-learn.  Training and validation use
independent positions, opponent hands, runouts and random seeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from numba import njit
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import equity_engine as engine
import equity_network as network


@njit(cache=False)
def _label_batch(hands, boards, samples, seed, lut, choose):
    """Equity labels without storing per-simulation opponent covariates."""
    result = np.empty(len(hands), dtype=np.float32)
    state = np.uint64(seed) | np.uint64(1)
    for row in range(len(hands)):
        hole, board = hands[row], boards[row]
        used = np.zeros(36, dtype=np.bool_)
        for c in hole:
            used[c] = True
        for c in board:
            used[c] = True
        deck = np.empty(31 - len(board), dtype=np.int32)
        index = 0
        for card in range(36):
            if not used[card]:
                deck[index] = card
                index += 1
        full_board = np.empty(5, dtype=np.int32)
        full_board[:len(board)] = board
        opponent = np.empty(5, dtype=np.int32)
        wins = 0.0
        for _ in range(samples):
            for j in range(10 - len(board)):
                state = engine._next(state)
                k = j + int(state % np.uint64(len(deck) - j))
                deck[j], deck[k] = deck[k], deck[j]
            for j in range(5):
                opponent[j] = deck[j]
            for j in range(5 - len(board)):
                full_board[len(board) + j] = deck[5 + j]
            ours = engine._score(hole, full_board, 5, lut, choose)
            theirs = engine._score(opponent, full_board, 5, lut, choose)
            wins += 1.0 if ours > theirs else .5 if ours == theirs else 0.0
        result[row] = wins / samples
    return result


def make_data(count, samples, seed):
    rng = np.random.default_rng(seed)
    features, labels, made = [], [], []
    for size in (3, 4):
        n = count // 2 + (count % 2 if size == 3 else 0)
        # Shuffling full decks avoids duplicate cards and does not condition on
        # any actual hero hand or actual future board.
        draws = np.array([rng.permutation(36)[:5 + size] for _ in range(n)], dtype=np.int32)
        hands, boards = np.ascontiguousarray(draws[:, :5]), np.ascontiguousarray(draws[:, 5:])
        start = time.perf_counter()
        y = _label_batch(hands, boards, samples, np.uint64(seed + size * 1949), engine._LUT, engine._CHOOSE)
        x = network._features_batch(hands, boards, engine._LUT, engine._CHOOSE)
        raw = np.empty((n, 2), dtype=np.float32)
        for i in range(n):
            raw[i] = [size, engine._score(hands[i], boards[i], size, engine._LUT, engine._CHOOSE)]
        print(f"board={size} positions={n} rollouts={samples} seconds={time.perf_counter()-start:.1f}", flush=True)
        features.append(x)
        labels.append(y)
        made.append(raw)
    return np.concatenate(features), np.concatenate(labels), np.concatenate(made)


def metrics(y, predicted):
    from scipy.stats import spearmanr
    return {"rmse": float(np.sqrt(np.mean((y - predicted) ** 2))),
            "mae": float(np.mean(np.abs(y - predicted))),
            "correlation": float(np.corrcoef(y, predicted)[0, 1]),
            "rank_correlation": float(spearmanr(y, predicted).statistic),
            "bias": float(np.mean(predicted - y))}


def fixed_board_validation(model, scaler, seed, count=8, hands_per_board=64, rollouts=2048):
    """Measure the within-board hand ordering used by opponent range weights."""
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    neural_ranks, made_ranks, errors = [], [], []
    for j in range(count):
        size = 3 + j % 2
        cards = rng.permutation(36)
        board, available = cards[:size], cards[size:]
        hands = np.array([rng.permutation(available)[:5] for _ in range(hands_per_board)], np.int32)
        boards = np.tile(board, (hands_per_board, 1)).astype(np.int32)
        y = _label_batch(hands, boards, rollouts, np.uint64(seed + 971*j), engine._LUT, engine._CHOOSE)
        raw = np.array([engine._score(hand, boards[0], size, engine._LUT, engine._CHOOSE)
                        for hand in hands], np.int32)
        x = network._features_batch(hands, boards, engine._LUT, engine._CHOOSE, scores=raw)
        p = np.clip(model.predict(scaler.transform(x)), 0, 1)
        neural_ranks.append(float(spearmanr(y, p).statistic))
        made_ranks.append(float(spearmanr(y, raw).statistic))
        errors.extend((p - y).tolist())
    return {"boards": count, "hands_per_board": hands_per_board, "rollouts": rollouts,
            "seed": seed, "neural_mean_rank_correlation": float(np.mean(neural_ranks)),
            "made_hand_mean_rank_correlation": float(np.mean(made_ranks)),
            "neural_per_board_rank_correlation": neural_ranks,
            "made_hand_per_board_rank_correlation": made_ranks,
            "neural_rmse": float(np.sqrt(np.mean(np.square(errors))))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=80000)
    parser.add_argument("--rollouts", type=int, default=384)
    parser.add_argument("--validation", type=int, default=1500)
    parser.add_argument("--validation-rollouts", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026110807)
    parser.add_argument("--output", type=Path, default=ROOT / "equity_network.npz")
    args = parser.parse_args()
    began = time.perf_counter()
    engine.warmup()
    x, y, raw = make_data(args.positions, args.rollouts, args.seed)
    vx, vy, vraw = make_data(args.validation, args.validation_rollouts, args.seed + 999983)
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                         alpha=.02, batch_size=512, learning_rate_init=.0015,
                         max_iter=220, random_state=args.seed, early_stopping=True,
                         validation_fraction=.08, n_iter_no_change=16, tol=1e-5,
                         verbose=True)
    model.fit(xs, y)
    pred = np.clip(model.predict(scaler.transform(vx)), 0, 1)
    # A flexible independently fitted calibration gives the current-made-hand
    # covariate a fair comparison, rather than comparing score units to equity.
    baseline = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=31,
                                             l2_regularization=1., random_state=args.seed)
    baseline.fit(raw, y)
    baseline_pred = np.clip(baseline.predict(vraw), 0, 1)
    report = {"feature_version": network.FEATURE_VERSION, "feature_count": network.FEATURE_COUNT,
              "training_seed": args.seed, "training_positions": args.positions,
              "training_rollouts": args.rollouts, "validation_seed": args.seed + 999983,
              "validation_positions": args.validation, "validation_rollouts": args.validation_rollouts,
              "layers": [network.FEATURE_COUNT, 64, 32, 1], "iterations": int(model.n_iter_),
              "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
              "neural": metrics(vy, pred), "calibrated_made_hand": metrics(vy, baseline_pred),
              "raw_score_rank_correlation": metrics(vy, vraw[:, 1])["rank_correlation"],
              "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                                ("equity_network.py", "equity_engine.py", "evaluator.py", "tools/train_equity_network.py")},
              "elapsed_seconds": time.perf_counter() - began}
    report["fixed_board_validation"] = fixed_board_validation(model, scaler, args.seed + 2859281)
    for size, region in ((3, slice(0, len(vy) // 2)), (4, slice(len(vy) // 2, None))):
        report[f"board_{size}"] = {"neural": metrics(vy[region], pred[region]),
                                 "calibrated_made_hand": metrics(vy[region], baseline_pred[region])}
    weights = {"feature_version": np.array(network.FEATURE_VERSION),
               "mean": scaler.mean_.astype(np.float32), "scale": scaler.scale_.astype(np.float32),
               "metadata": np.array(json.dumps(report, sort_keys=True))}
    for layer in range(3):
        weight = model.coefs_[layer].astype(np.float32)
        weight[np.abs(weight) < 1e-12] = 0
        weights[f"w{layer}"] = weight
        weights[f"b{layer}"] = model.intercepts_[layer].astype(np.float32)
    np.savez_compressed(args.output, **weights)
    print(json.dumps(report, indent=2), flush=True)
    print(f"saved={args.output} bytes={args.output.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
