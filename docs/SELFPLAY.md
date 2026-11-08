# Historical adaptive bot self-play

Commit: `97883009569ad2d383b8145e46ee642262390049`.
The newer selected policy is documented in [NEURAL_REVISION.md](NEURAL_REVISION.md).

**Three official duplicate matches, 6,000 played hands: all CHOP, exactly
0.000 bb per duplicate deal in every match.** Both bots ran as independent
server processes using the identical policy at that commit; learned memory resets
at each half. These are run-only checks; no strategy or model was changed.

| Seed | Hands | bb/duplicate deal | 95% interval | Verdict |
|---|---:|---:|---|---|
| selfplay-101 | 2000 | 0.000 | [0.000, 0.000] | CHOP |
| selfplay-307 | 2000 | 0.000 | [0.000, 0.000] | CHOP |
| selfplay-809 | 2000 | 0.000 | [0.000, 0.000] | CHOP |

## Reliability

Across all 12 bot-half observations:

- Errors: **0**; skipped decisions: **0**.
- Forfeited halves: **0**; exhausted time banks: **0**.
- Charged thinking time: **2.536–5.354s** per bot/half, against an 8s bank.
- Smallest remaining bank: **2.646s**.

## Observed behavior

Actions pooled across both bots and both halves (CHECK and CALL split by
whether CHECK_CALL added zero or positive chips):

| Action | Count | Share of decisions |
|---|---:|---:|
| CALL | 8726 | 21.5% |
| CHECK | 18854 | 46.3% |
| FOLD | 4086 | 10.0% |
| RAISE_HALF | 6874 | 16.9% |
| RAISE_POT | 2140 | 5.3% |

- Showdown: **31.9%** of played hands; **68.1%** ended by a fold.
- Mean/median pot: **15.41/8 chips**; 95th percentile **48 chips**.
- Full 160-chip showdowns: **84 (1.4%)**.
- Half-pot/pot raises per played hand: **1.146/0.357**, pooled across both players.

## What the draw means

All paired per-deal outcomes are zero, and the full seat-level action logs
match between duplicate halves after ignoring only the half label. The
sampler is deterministic for the same information and each seat encounters
the same prior history when the servers exchange seats. Gains for one copy
therefore become exactly equal losses for that same copy in the other half.
The degenerate [0,0] interval is expected symmetry, not a statistical claim
that the bot is optimal or safe against other strategies.

Self-play confirms legal play, half resets, repeatability and bank headroom.
It cannot rank the bot against tournament opponents: both copies can share
the same weaknesses. The repeated halves are not independent new deals.

Detailed evidence: `results/selfplay/analysis.json`, plus each seed’s
official match JSON and text log.
