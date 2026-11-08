# Strength research and range stability

The baseline audits below refer to adaptive commit
`97883009569ad2d383b8145e46ee642262390049`. They informed development; selected
features and fresh results are recorded in [NEURAL_REVISION.md](NEURAL_REVISION.md).

Research checked against the event's exact 36-card, five-hole-card Omaha
rules. These are engineering recommendations and source notes, not a claim of
equilibrium play or a tournament win probability.

## What existing open-source models can supply

* [OpenSpiel CFR](https://github.com/google-deepmind/open_spiel/blob/3c5361621d8b0fc05fc0456604d8fada7096a2cb/open_spiel/python/algorithms/cfr.py)
  is an Apache-2.0 implementation of counterfactual regret minimization. Its
  module explains that the average policy, rather than the latest regret-matched
  policy, converges to an equilibrium under the algorithm's assumptions.
* [OpenSpiel external-sampling MCCFR](https://github.com/google-deepmind/open_spiel/blob/3c5361621d8b0fc05fc0456604d8fada7096a2cb/open_spiel/python/algorithms/external_sampling_mccfr.py)
  samples chance and opponent actions during regret updates. Sampling reduces
  traversal cost; it does not provide an already-trained policy for this event.
* [RLCard's documented algorithms and model zoo](https://github.com/datamllab/rlcard/blob/d7d0a957baf4cc7225a50522adb0164bf130a9d0/README.md)
  include chance-sampling CFR and a pretrained **Leduc Hold'em** CFR model,
  alongside rule-based Limit Hold'em models. The documented model zoo does not
  supply pretrained short-deck PLO5 weights. The differences in hole-card rules,
  hand ordering, available cards, betting and stack sizes make direct policy
  transfer unjustified.

The event has 376,992 possible five-card private hands. Before considering
betting history, combining a private hand with a visible three-card flop gives
376,992 × C(31, 3) = 1,694,579,040 raw observations. Suit symmetry helps, but a
full tabular solve remains far larger than a small deployment lookup. A toy
CFR abstraction should not be called a solved event policy.

The practical path is to train compact event-specific predictors offline,
using the unchanged official evaluator for labels, and retain explicit action
and range reasoning. An equity predictor trained on independent Monte Carlo
positions is supervised learning of showdown equity, not a pretrained betting
AI and not a game-theoretic guarantee.

## Concrete model weakness

The submitted policy ranks a sampled opponent's current made hand on the flop
and turn. This omits future flushes, trips, full houses and straights. A strong
draw can be assigned the same weak likelihood as an unpromising high-card hand.
In this game's unusual ordering, a generic Hold'em draw heuristic can also be
misleading: flushes beat full houses and trips beat straights.

Use a draw-aware estimate of opponent equity against independently sampled
legal reference hands, conditioned only on the board prefix visible when an
action occurred. Historical flop actions must use the flop prefix even while
evaluating a river decision. The current equity outcome against our actual
private hand must not itself determine an opponent action likelihood: doing so
would create a circular range model. A separately trained predictor is a cheap
way to approximate this independent reference calculation.

## Effective sample size audit

For normalized range weights w, effective sample size (ESS) is
`1 / sum(w * w)`. It describes concentration of Monte Carlo weights; it does
not measure correctness of the assumed opponent model.

Replaying the unchanged policy on all 132 bot decisions from the 40-hand
direct Astra exhibition, with 192 legal particles per decision, gave:

| Statistic | ESS |
| --- | ---: |
| Minimum | 28.82 |
| 5th percentile | 70.29 |
| Median | 172.04 |
| Decisions below 48 | 1 |
| Decisions below 96 | 22 |

The one decision below 48 was half 1, hand 3, river: estimated equity 76.4%
against a 16.25% call price. The correct call was far from the estimated
decision boundary. Half 2, hand 12's expensive turn fold had ESS 136.85, raw
equity 29.16%, and call price 27.5%. The policy's realization and call margins
caused that fold. Particle collapse therefore does not explain that particular
decision. These measurements describe this small trace, not tournament rates.

A separate development audit played 500 duplicate deals (1,000 actual hands)
per fixed profile, using seed `ess-research-dev` and the unchanged submitted
policy. It recomputed weights after each real decision, without feeding audit
results back into either player:

| Opponent | Decisions | Minimum ESS | ESS below 48 | Large-pot decisions below 48 |
| --- | ---: | ---: | ---: | ---: |
| Selective aggressive | 3,432 | 8.91 | 215 (6.3%) | 138 |
| Tight passive | 3,595 | 13.47 | 195 (5.4%) | 124 |
| Loose aggressive | 3,745 | 34.92 | 4 (0.1%) | 3 |
| Always pot | 3,687 | 73.33 | 0 | 0 |

Here a large pot means `pot + to_call >= 24` chips. Against selective and
tight-passive profiles, respectively, 24 and 22 decisions fell below ESS 24;
all were large-pot decisions. This broader audit supports a selective
resampling safeguard even though the Astra trace did not exhibit a damaging
collapse. Increasing the base particle count also improves precision, subject
to the actual HTTP thinking-bank check.

## Appropriate safeguard

Estimate Monte Carlo uncertainty for weighted equity with
`sqrt(sum(w**2 * (outcome - equity)**2))`. The approximation
`sqrt(equity*(1-equity)/ESS)` is a useful quick diagnostic, but does not fully
account for the relationship between weight and outcome. Neither formula
captures an incorrect opponent likelihood model.

When a large-pot choice is close to its estimated boundary and ESS is low,
draw additional independent legal particles, recompute all range features and
weights, and cache the expanded result for that board. Triggering only on
expensive close decisions bounds average runtime. Fresh samples must be
generated independently of their observed wins; keeping only favorable
particles would bias equity.

Do not automatically mix a tight range toward uniform merely because its ESS
is low: that invents weak hands against a selective opponent. More particles
address numerical variance without changing the target range. If deployment
limits force weight regularization, tune a small, explicit floor against
independent opponents and report the induced range-model bias. For a uniform
mixture `w_new=(1-alpha)*w+alpha/N`, the ESS after mixing is exactly
`1 / ((1-alpha)**2*(1/ESS-1/N)+1/N)`.

Use common sampled outcomes when comparing candidate betting actions so the
variance of the *difference* is lower. Confidence margins should distinguish
Monte Carlo noise from strategic model uncertainty; a low ESS alone does not
justify folding a clearly profitable call or escalating aggression.

## Validation implications

Existing independent style probes mostly use honest equity thresholds. Add a
separate higher-sample range-aware opponent with polarized betting and
check-raises, and label it an untimed strength probe if it exceeds the event
bank. Its decisions must use its own legal observations, never actual hidden
cards. Freeze that probe before candidate tuning. A new direct-Astra matchup
must use fresh deals and contexts. Report uncertainty and deployment time-bank
results separately from the strength matchup.

`tools/slow_expert_opponent.py` supplies the additional untimed probe. It uses
512 independent showdown particles, separately seeded 12-reference estimates
for each sampled opponent hand at each historical flop/turn prefix, a hard
range filter with a nonzero bluff mixture, conservative value reraises,
occasional traps, and a small polarized river bluff component when holding a
nut-flush blocker. It imports no production strategy, opponent model, fitted
preflop lookup or neural parameters. Its exact card scorer is shared. Higher
sampling and a richer policy make it a useful challenge, not proof that the
probe itself is expert or equilibrium play.

The probe completed an observation-only 200-duplicate-deal sanity match against
an always-call opponent (`expert-sanity`), winning 10.595 bb per duplicate deal
with a normal 95% interval [8.676, 12.514] and zero invalid actions. Its direct
strategy times were 14.92 and 9.21 seconds for the two 200-hand halves. This
confirms the probe can exploit a simple station and also confirms why it must
be labeled untimed; it says nothing about the final candidate's win rate.
