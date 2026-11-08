#!/usr/bin/env python3
"""Fit an offline Monte Carlo preflop-equity lookup for exact event rules.

Every five-card hand is assigned to its canonical suit-isomorphism class.
Each class receives independent Monte Carlo samples against a uniform random
five-card opponent and random board. Thus suit-renaming symmetry is exact,
while each class's equity is estimated, not solved. No external neural weights
are used. Expected training: roughly 1–3 minutes on one CPU at 2,048 draws/class.

Run from repository root:
    python tools/train_preflop.py --samples 2048 --seed 20261108
"""
import argparse
from itertools import combinations, permutations
from math import comb
from pathlib import Path
import sys
import time

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import equity_engine as engine


@njit(cache=False)
def canonical_indices(hands, suit_permutations, choose):
    """Map combinadic hand index to the least suit-equivalent index."""
    out = np.empty(len(hands), np.int32)
    cards_by_index = np.empty((len(hands), 5), np.int32)
    for row in range(len(hands)):
        hand = hands[row]
        index = 0
        for j in range(5):
            index += choose[hand[j], j + 1]
        cards_by_index[index] = hand
        best_mask = np.uint64(1) << np.uint64(40)
        for p in suit_permutations:
            mask = np.uint64(0)
            for j in range(5):
                card = ((hand[j] >> 2) << 2) + p[hand[j] & 3]
                mask |= np.uint64(1) << np.uint64(card)
            if mask < best_mask:
                best_mask = mask
        canonical = 0
        slot = 1
        for card in range(36):
            if best_mask & (np.uint64(1) << np.uint64(card)):
                canonical += choose[card, slot]
                slot += 1
        out[index] = canonical
    return out, cards_by_index


def verify_symmetry(equities, mapping, draws=200):
    """Check independently generated suit permutations and indexing."""
    rng = np.random.default_rng(1981)
    for _ in range(draws):
        hand = sorted(rng.choice(36, 5, replace=False).tolist())
        index = sum(comb(c, i + 1) if c >= i + 1 else 0 for i, c in enumerate(hand))
        for p in permutations(range(4)):
            changed = sorted((c & ~3) + p[c & 3] for c in hand)
            other = sum(comb(c, i + 1) if c >= i + 1 else 0 for i, c in enumerate(changed))
            assert mapping[index] == mapping[other], (hand, changed)
            assert equities[index] == equities[other], (hand, changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=20261108)
    parser.add_argument('--output', type=Path, default=ROOT / 'preflop_model.npz')
    args = parser.parse_args()
    if args.samples < 32:
        parser.error('--samples must be >=32')
    start = time.perf_counter()
    engine.warmup()
    hands = np.array(list(combinations(range(36), 5)), np.int32)
    suit_permutations = np.array(list(permutations(range(4))), np.int32)
    mapping, cards = canonical_indices(hands, suit_permutations, engine._CHOOSE)
    classes = np.unique(mapping)
    print(f'{len(hands):,} hands, {len(classes):,} suit classes, '
          f'{len(classes) * args.samples:,} rollouts', flush=True)
    values = np.empty(len(hands), np.float32)
    stderr = np.empty(len(hands), np.float32)
    empty_board = np.empty(0, np.int32)
    train_start = time.perf_counter()
    for i, key in enumerate(classes):
        seed = np.uint64((args.seed + int(key) * 11400714819323198485) & ((1 << 64) - 1))
        wins, *_ = engine._sample(cards[key], empty_board, args.samples, seed,
                                 engine._LUT, engine._CHOOSE, False)
        values[key] = wins.mean()
        stderr[key] = wins.std(ddof=1) / np.sqrt(args.samples)
        if (i + 1) % 2000 == 0:
            elapsed = time.perf_counter() - train_start
            print(f'{i+1:,}/{len(classes):,} classes; {elapsed:.1f}s; '
                  f'ETA {elapsed / (i + 1) * (len(classes)-i-1):.1f}s', flush=True)
    equity = values[mapping]
    standard_errors = stderr[mapping]
    verify_symmetry(equity, mapping)
    # Persist only inference data plus auditable training metadata. The full
    # canonical mapping is reproducible and is not needed at inference.
    np.savez_compressed(args.output, equity=equity, standard_error=standard_errors,
                        samples_per_class=np.int64(args.samples), seed=np.int64(args.seed),
                        class_count=np.int64(len(classes)),
                        version=np.int64(1),
                        description='Uniform-opponent Short Deck PLO5 offline Monte Carlo equity; combinadic index')
    print(f'Saved {args.output}: {args.output.stat().st_size:,} bytes; '
          f'wall {time.perf_counter()-start:.1f}s; '
          f'equity range [{equity.min():.4f}, {equity.max():.4f}]; '
          f'max standard error {standard_errors.max():.4f}; '
          '200 hands ×24 suit-invariance checks passed.', flush=True)


if __name__ == '__main__':
    main()
