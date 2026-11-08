# Evaluation and limits

The bot is developed for the exact starter-bundle revision recorded in README.
Results are **big blinds per duplicate deal**: the sum of two seat-swapped hands,
not bb per single hand. Intervals use the tournament's paired 95% normal CI.

## Frozen-policy results

The selected policy won **24/24 fresh-deal benchmark matches (48,000 hands)**,
with zero agent errors. Each opponent was tested on seeds 11, 29 and 73,
1,000 duplicate deals per seed. Every individual match had a positive lower
95% confidence bound. These are synthetic-opponent results, not tournament standings.

| Opponent | Mean bb/deal | 95% CI |
|---|---:|---|
| Random | +8.860 | [+8.055, +9.665] |
| Calling station | +10.593 | [+10.062, +11.124] |
| Always half-pot | +10.620 | [+9.844, +11.397] |
| Always pot | +7.127 | [+6.209, +8.044] |
| Check/fold | +1.493 | [+1.489, +1.497] |
| Independent tight equity | +1.808 | [+1.506, +2.109] |
| Independent range-aware equity | +0.982 | [+0.803, +1.162] |
| Independent fast range policy | +1.126 | [+0.912, +1.340] |

**Official HTTP judge:** three additional full matches passed (6,000 hands).
Against FastRange with fresh processes per half: +1.029 bb/deal, CI
[+0.625,+1.433], 0 errors, thinking 4.59/5.16s of the 8s banks. Against
random: +8.906, CI [+7.530,+10.281], 0 errors, 2.82/2.39s. Inside the
restricted one-CPU container against the calling station: +11.052, CI
[+10.139,+11.965], 0 errors, no skipped decisions, 3.143/4.077s.

**Deployment:** the final image cold-started in 8.3s (limit 15s), adds
202.57 MiB (limit 3 GiB), and passed the no-network, read-only, nonroot,
one-CPU/4GB smoke check. All 29 tests passed. Recorded runtime hashes identify the
container that was tested. The participant name was subsequently changed from
`Pokerbot` to `jr007`; strategy and model files are unchanged. See `results/summary.json` and
`results/runtime-validation-final.json`.

One direct benchmark half took 8.027s during concurrent computation; it
did not enforce a time bank and is not a deployment readiness claim. The
official HTTP results above are the bank checks. Tournament hardware may vary.

## Evidence hierarchy

- `results/tuning.json`: nine development candidates, each evaluated on two
  development seeds and four opponent profiles. Selection is exploratory.
- `results/dev-retest.json`: current-code retest of selected parameters. The
  mean improvement over defaults is not statistically established by itself.
- `results/range-development.json`: manual range-conditioning sweep, using
  development seeds. This motivated stronger conditioning against rare raises.
- `results/dev-table-ablation.json`: offline-table development ablation. All
  measured table/no-table differences in this first ablation include zero.
- `results/preflop-training.json`: offline fitted model provenance/calibration
  The table is trained on randomly generated hands, independent
  of the benchmark's game seeds.
- `results/dev-table-ablation-range25.json`: the final table ablation at the
  selected range weight, with a positive paired development improvement.
- `results/final-baselines.json` and `results/final-validation.json`: fresh-deal
  frozen-policy results and source fingerprints.
- `results/summary.json`: combined final results and HTTP evidence paths.

The synthetic opponent pool tests random actions, passive calling, unrelenting
aggression, tight play and equity-aware range selection. It does not contain
other teams' private submissions. We cannot estimate a tournament win probability
from this pool or claim the policy is game-theoretically optimal.

The slow independent `mc_loose`, `mc_tight` and `mc_range` opponents are strength
probes. They use the official Python evaluator, and can exceed the event time
bank. Their in-process matches allow full play without time-bank defaults.
`tools/fast_range_opponent.py` is an independent policy using the fast sampler;
its HTTP matches test a more realistic strong opponent under the actual bank.

## Model provenance and fitting

This is an adaptation of an existing open source **algorithmic equity model**
(PyPokerEngine HonestPlayer), not a downloaded pretrained neural policy. Generic
Hold'em weights would have the wrong rules. The event-specific model uses exact
rankings and offline Monte Carlo fitting, plus a small tuned betting policy.
The preflop table covers every legal five-card hole hand, grouped into 19,998
suit-isomorphic classes, using 2,048 rollouts per class (40,955,904 total).

The policy's action likelihoods are approximations, especially on drawing hands.
Its raise calculation models a call/fold response and equity realization; it
is not a full future-street solver. Opponents can potentially exploit it.

## Runtime verification

Organizer rules, judge, evaluator and Dockerfiles are unchanged. The unmodified
build script encountered the managed cloud proxy's CA trust issue inside the
inner Docker build. Equivalent copies of the Docker recipes were built with the
host CA supplied as a temporary BuildKit secret and `PIP_CERT`; TLS verification
remained enabled. This is a cloud transport adaptation, not a strategy or
container restriction change. The repository keeps the original Dockerfiles. `tools/check_build_cloud.sh`
saves this cloud build procedure; its shell syntax and generated recipe identity
were checked, without repeating the already successful expensive builds.

The restricted smoke test uses one CPU, 4 GB RAM, no network, nonroot, read-only
root and a 512 MB `/tmp`. The initial image passed startup in 8.0 seconds and
five legal sample decisions. See `results/runtime-validation.json` for exact
image identity and `results/runtime-validation-final.json` for the frozen-policy rerun. Startup and strategy
performance can differ on the tournament's hardware.

In-process timing excludes HTTP and JSON and cannot establish bank compliance.
Only final `judge_engine.py` reports substantiate zero errors and remaining bank.
