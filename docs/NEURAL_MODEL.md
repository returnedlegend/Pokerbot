# Neural equity model

`equity_network.npz` contains an event-specific supervised equity predictor.
It was trained from legal Short Deck PLO5 simulations using the unchanged
official hand evaluator and the open-source scikit-learn `MLPRegressor`.
These are newly trained weights, not a downloaded pretrained poker policy.

The model estimates showdown equity against a uniformly random five-card
opponent from five private cards and the currently visible flop or turn.
Its 115 features include rank counts, the exact current hand category and
kickers, canonical suit patterns, nut-flush potential and all six legal
short-deck straight windows. Suit relabeling and private-card ordering leave
predictions unchanged. The interface accepts no actual opponent hand or
unrevealed board cards.

The selected production policy uses the estimate as a covariance-weighted
anchor for its own Monte Carlo equity estimate. Opponent adaptation and
bet sizing remain explicit policy calculations. The alternative that used
neural estimates to rank opponent ranges is disabled after development
comparisons. A well-calibrated equity predictor alone does not establish a
strong betting policy or a tournament win probability.

## Training and independent validation

The network has layers 115 → 64 → 32 → 1. Training uses 80,000 randomly
generated flop/turn positions and 384 legal rollouts per position, with seed
`2026110807`. A separate seed supplies 1,500 validation positions with 4,096
rollouts each. The exported NPZ stores training settings, dependency versions,
source SHA256 fingerprints and validation measurements.

| Independent measurement | Neural predictor | Current made-hand baseline |
| --- | ---: | ---: |
| Equity RMSE, 1,500 positions | 0.02082 | 0.10645 after flexible calibration |
| Rank correlation across those positions | 0.99278 | 0.77049 after calibration |
| Mean within-board rank correlation | 0.98964 | 0.83077, raw hand score |

Within-board validation uses eight additional boards, 64 private hands per
board and 2,048 rollouts per hand. Its neural equity RMSE is 0.01932.
Simulation noise contributes to these reported errors. These measurements
validate equity prediction, not incremental match EV. Development comparisons
do not establish a statistically significant improvement against a balanced
field of independent opponents.

## Reproduction and deployment

Install offline training dependencies with
`python -m pip install -r requirements-training.txt`, then run:

```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/train_equity_network.py
.venv/bin/python -m pytest tests/test_equity_network.py -q
```

The training script writes a compact NPZ with standardized-input statistics,
three dense weight matrices and biases. Runtime imports only NumPy and the
project's existing Numba evaluator; it does not import scikit-learn.
An exact rank-mask table accelerates draw-feature extraction. Tiny learned
weights below `1e-12` are zeroed to avoid CPU denormal arithmetic.

`equity_network.batchpredict(hands, board, scores=None)` accepts candidate
hands shaped `(n, 5)` and a three- or four-card public board. Optional `scores`
must contain exact current made-hand scores for those same cards and board.
Call `equity_network.warmup()` during startup to compile the feature extractor.
Fresh-process tests cover suit symmetry, batch consistency, visible-board
restrictions and independent live Monte Carlo calibration. Restricted Docker
startup and thinking-bank measurements are recorded separately in
`results/neural-runtime-validation.json`.
