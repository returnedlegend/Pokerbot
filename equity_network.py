"""Simulation-trained flop/turn equity network for Short Deck PLO5.

The network estimates showdown equity against a uniformly random five-card
opponent, integrating over unknown runouts.  Its inputs contain only the
candidate's five cards and the current public board.  It is a range-strength
covariate, not a substitute for equity against a particular betting range.

Training uses scikit-learn offline; inference needs only NumPy and the existing
Numba hand evaluator.  Suits and private-card ordering have no significance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numba import njit

import equity_engine as engine

FEATURE_VERSION = 1
FEATURE_COUNT = 115
_MODEL = None
_STRAIGHTS = np.array([0b100001111, 0b000011111, 0b000111110,
                       0b001111100, 0b011111000, 0b111110000], dtype=np.int32)
_STRAIGHT_LOOKUP = None


def _straight_lookup():
    """Exact draw descriptors for the 512 possible private/public rank masks.

    Vectorized construction takes about 0.1s and removes pair enumeration
    from every candidate hand at inference.  This is a rules-derived table,
    not learned data, and does not contain cards from any actual runout.
    """
    global _STRAIGHT_LOOKUP
    if _STRAIGHT_LOOKUP is None:
        masks = np.arange(512, dtype=np.int32)
        counts = np.array([int(mask).bit_count() for mask in masks], dtype=np.float32)
        table = np.empty((512, 512, 24), dtype=np.float32)
        for k, window in enumerate(_STRAIGHTS):
            best = np.zeros((512, 512), dtype=np.int32)
            outs = np.zeros_like(best)
            ranks = [r for r in range(9) if window & (1 << r)]
            for a in range(4):
                for b in range(a + 1, 5):
                    pair = (1 << ranks[a]) | (1 << ranks[b])
                    needed = int(window) ^ pair
                    present = ((masks & pair) == pair)[:, None]
                    hits = counts[masks & needed].astype(np.int32)[None, :]
                    best = np.maximum(best, np.where(present, hits, 0))
                    outs |= np.where(present & (hits == 2), (needed ^ (masks & needed))[None, :], 0)
            table[:, :, 4*k] = counts[masks & window][:, None] / 5
            table[:, :, 4*k + 1] = counts[masks & window][None, :] / 5
            table[:, :, 4*k + 2] = best / 3
            table[:, :, 4*k + 3] = counts[outs] / 9
        _STRAIGHT_LOOKUP = table
    return _STRAIGHT_LOOKUP


@njit(cache=False)
def _features_impl(hands, boards, scores, straight_lookup):
    """Boards can contain one shared board or one board per private hand."""
    out = np.zeros((len(hands), FEATURE_COUNT), dtype=np.float32)
    board_len = boards.shape[1]
    hr = np.zeros(9, dtype=np.int32)
    br = np.zeros(9, dtype=np.int32)
    hs = np.zeros(4, dtype=np.int32)
    bs = np.zeros(4, dtype=np.int32)
    hm = np.zeros(4, dtype=np.int32)
    bm = np.zeros(4, dtype=np.int32)
    order = np.empty(4, dtype=np.int32)
    keys = np.empty(4, dtype=np.int64)
    for i in range(len(hands)):
        board = boards[0 if len(boards) == 1 else i]
        hr[:] = 0
        br[:] = 0
        hs[:] = 0
        bs[:] = 0
        hm[:] = 0
        bm[:] = 0
        hmask, bmask = 0, 0
        for card in hands[i]:
            rank, suit = card >> 2, card & 3
            hr[rank] += 1
            hs[suit] += 1
            hm[suit] |= 1 << rank
            hmask |= 1 << rank
        for card in board:
            rank, suit = card >> 2, card & 3
            br[rank] += 1
            bs[suit] += 1
            bm[suit] |= 1 << rank
            bmask |= 1 << rank
        p = 0
        for rank in range(9):
            out[i, p] = hr[rank] / 4.0
            p += 1
        for rank in range(9):
            out[i, p] = br[rank] / 4.0
            p += 1
        for rank in range(9):
            out[i, p] = 1.0 if hr[rank] >= 2 else 0.0
            p += 1
        for rank in range(9):
            out[i, p] = min(hr[rank], 2) * min(br[rank], 3) / 6.0
            p += 1
        out[i, p] = board_len - 3
        p += 1
        score = scores[i]
        category = score >> 20
        for cat in range(9):
            out[i, p] = 1.0 if category == cat else 0.0
            p += 1
        for shift in (16, 12, 8, 4, 0):
            out[i, p] = ((score >> shift) & 15) / 8.0
            p += 1

        # Canonical suit ordering includes the full masks as tie breakers.
        # Global suit relabeling therefore produces exactly the same vector.
        for suit in range(4):
            order[suit] = suit
            keys[suit] = (bs[suit] * 6 + hs[suit]) * 262144 + hm[suit] * 512 + bm[suit]
        for a in range(1, 4):
            k = a
            while k > 0 and keys[order[k]] > keys[order[k - 1]]:
                order[k], order[k - 1] = order[k - 1], order[k]
                k -= 1
        for suit in order:
            htop, hsecond, btop, bsecond, nut = -1, -1, -1, -1, -1
            for rank in range(8, -1, -1):
                if hm[suit] & (1 << rank):
                    if htop < 0:
                        htop = rank
                    elif hsecond < 0:
                        hsecond = rank
                if bm[suit] & (1 << rank):
                    if btop < 0:
                        btop = rank
                    elif bsecond < 0:
                        bsecond = rank
                elif nut < 0:
                    nut = rank
            higher = 0
            for rank in range(htop + 1, 9):
                if not (bm[suit] & (1 << rank)):
                    higher += 1
            need = max(0, 3 - bs[suit])
            possible = hs[suit] >= 2 and need <= 5 - board_len
            out[i, p] = hs[suit] / 5.0
            out[i, p + 1] = bs[suit] / 4.0
            out[i, p + 2] = (htop + 1) / 9.0
            out[i, p + 3] = (hsecond + 1) / 9.0
            out[i, p + 4] = (btop + 1) / 9.0
            out[i, p + 5] = (bsecond + 1) / 9.0
            out[i, p + 6] = higher / 9.0
            out[i, p + 7] = 1.0 if possible else 0.0
            out[i, p + 8] = 1.0 if possible and htop == nut else 0.0
            out[i, p + 9] = need / 3.0
            p += 10

        # Every legal short-deck straight, including A6789, using exactly
        # two distinct private ranks and three board ranks.
        for k in range(24):
            out[i, p + k] = straight_lookup[hmask, bmask, k]
    return out


def _features_batch(hands, boards, lut, choose, scores=None):
    if scores is None:
        scores = np.fromiter((engine._score(hand, boards[0 if len(boards) == 1 else i],
                              boards.shape[1], lut, choose) for i, hand in enumerate(hands)),
                             dtype=np.int32, count=len(hands))
    return _features_impl(hands, boards, scores, _straight_lookup())


def _load_model():
    global _MODEL
    if _MODEL is None:
        with np.load(Path(__file__).with_name("equity_network.npz"), allow_pickle=False) as data:
            if int(data["feature_version"]) != FEATURE_VERSION:
                raise ValueError("equity network feature version mismatch")
            _MODEL = {name: data[name].astype(np.float32) for name in
                      ("mean", "scale", "w0", "b0", "w1", "b1", "w2", "b2")}
            # L2 training can leave unused weights below float32's normal
            # range.  Those values have no predictive effect but trigger
            # expensive denormal arithmetic on some tournament CPUs.
            for name in ("w0", "w1", "w2"):
                _MODEL[name][np.abs(_MODEL[name]) < 1e-12] = 0
    return _MODEL


def _forward(features):
    model = _load_model()
    x = (features - model["mean"]) / model["scale"]
    x = np.maximum(x @ model["w0"] + model["b0"], 0)
    x = np.maximum(x @ model["w1"] + model["b1"], 0)
    return np.clip((x @ model["w2"] + model["b2"]).reshape(-1), 0, 1)


def batchpredict(hands, board, scores=None):
    """Estimate equity for each (n,5) candidate against a random opponent.

    ``board`` must be the current three- or four-card public board.  No actual
    opponent cards or future board cards are accepted by this interface.
    Candidate hands must contain distinct cards and exclude the board.  The
    optional ``scores`` vector reuses exact current made-hand scores already
    computed by equity_engine for those same hands and that same board.
    """
    hands = np.asarray(hands, dtype=np.int32)
    board = np.ascontiguousarray(board, dtype=np.int32)
    if hands.ndim != 2 or hands.shape[1] != 5 or board.ndim != 1 or len(board) not in (3, 4):
        raise ValueError("expected hands shaped (n,5) and a flop or turn board")
    if len(hands) == 0:
        return np.empty(0, dtype=np.float32)
    if scores is not None:
        scores = np.ascontiguousarray(scores, dtype=np.int32)
        if scores.shape != (len(hands),):
            raise ValueError("scores must have one current made-hand score per candidate")
    engine.warmup()
    x = _features_batch(np.ascontiguousarray(hands), board.reshape(1, -1),
                        engine._LUT, engine._CHOOSE, scores)
    return _forward(x)


def warmup():
    """Load exported weights and compile the feature extractor at startup."""
    engine.warmup()
    batchpredict(np.array([[0, 5, 10, 15, 20]], np.int32), [24, 29, 34],
                 scores=np.array([0], np.int32))


def metadata():
    """Return the model's reproducible training and holdout measurements."""
    with np.load(Path(__file__).with_name("equity_network.npz"), allow_pickle=False) as data:
        return json.loads(str(data["metadata"]))
