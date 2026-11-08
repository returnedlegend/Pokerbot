# Public-information review of the Astra exhibition

This reviews the baseline at commit `97883009569ad2d383b8145e46ee642262390049`, using `results/astra-exhibition.json`. The exhibition comprised 20 duplicate deals, 40 hands, and 132 bot decisions. The bot lost 34 chips, but the reported 95% interval for paired performance includes zero. This is a debugging sample, not evidence that either player is stronger.

## Method

The referee replayed the recorded actions to reconstruct the bot's observation and its opponent-model memory at each decision. All 132 baseline choices reproduced exactly. Diagnostic estimates received only the bot's hole cards, the visible board, public betting history, and its memory from preceding hands. The saved opponent hole cards and unrevealed runout were not estimator inputs, labels, or tuning targets.

At selected decisions, the same baseline likelihood model was evaluated with 50,000 independently sampled legal opponent holdings and runouts, instead of 192. The random seeds were `177777` and, where reported, `277777`; neither was derived from the exhibition deal seed. Both uniform equity and equity weighted by the baseline's public-history likelihood were computed. The likelihood is a heuristic, not a calibrated posterior; narrowing sampling uncertainty does not remove model uncertainty.

The baseline source hashes recorded by the exhibition include:

```
strategy.py       c6c34f488c70a16e787c931f793c5853269da2f1db58a04f285a3ea68da84b78
opponent_model.py 5c50216228bb3e14afc0cb37df11bca6bba5b6f0acabb7683deb2537518f65e9
policy.json       493d1dc8c748d2b2d38b27a8b97a8d6620def6f431c3d5645630c8b7f7b1d619
```

## A concrete terminal-call penalty

Half 2, zero-based hand 12, public history length 8:

- Bot hole cards: `8s Td Ts Ah As`.
- Visible board: `Qd Js 6d 8h`.
- Opponent raised preflop, called the bot's reraise, bet the flop, and moved all-in on the turn after the bot checked.
- Pot before calling: 116 chips. Call: 44 chips, the bot's entire remaining stack. Final pot if called: 160 chips.

The required equity is therefore `44 / 160 = 27.5%`. No subsequent betting remains. The baseline correctly sets realization to 1, but still subtracts the non-river `call_margin = 0.03544137179472241` from equity. This charges **5.671 chips** to a terminal call and raises its threshold to **31.044%**.

| Public-information estimate | Uniform equity | History-weighted equity | Terminal call EV, chips | Baseline penalized EV, chips |
|---|---:|---:|---:|---:|
| Recorded 192 particles | 39.583% | 29.155% | +2.648 | -3.022 |
| 50,000 particles, seed 177777 | 43.348% | 31.069% | +5.710 | +0.039 |
| 50,000 particles, seed 277777 | 43.452% | 30.909% | +5.454 | -0.217 |

The original estimate already makes calling profitable under the bot's own range model. The fixed margin alone changes that call into a fold. The two larger estimates have approximate weighted Monte Carlo standard errors of 0.22 percentage points each. They support a call under this range assumption; they do not establish an opponent-specific true equity or guarantee a winning runout.

The appropriate general fix is to waive future-play margins whenever the call ends betting, as already happens on the river. It does not require increasing bluffing or raising. For any terminal call, the baseline unnecessarily rejects its own estimated positive-EV interval from `pot_odds` through `pot_odds + call_margin`.

There is only one observed fold caused by this mechanism in this small trace. The other pre-river all-in call, half 2 hand 3, called eight chips into 152 with ample equity, so its action was unaffected; its estimated EV was still understated by the same 5.671 chips.

## Sampling changes several close decisions

These estimates preserve the same public-history likelihood and every risk margin. They change only the number and seed of legal sampled particles. `ESS` is `1 / sum(normalized_weights ** 2)`.

| Half / zero-based hand / history length | Street | Pot / call | Recorded equity / ESS | 50k equity | Recorded modeled call EV | 50k modeled call EV |
|---|---|---:|---:|---:|---:|---:|
| 1 / 2 / 2 | Preflop | 8 / 4 | 34.466% / 67.8 | 40.131% | -0.377 | +0.289 |
| 1 / 14 / 2 | Preflop | 8 / 4 | 30.168% / 101.1 | 37.892% | -0.882 | +0.026 |
| 2 / 0 / 1 | Preflop | 8 / 4 | 35.341% / 102.9 | 40.519% | -0.380 | +0.213 |
| 2 / 15 / 4 | Flop | 12 / 4 | 27.883% / 166.7 | 33.244% | -0.200 | +0.640 |

The hand 14 estimate is too close to zero to distinguish calling from folding reliably. These are modeled EV differences, not measured tournament gains. Increasing samples can change raises as well as folds; an adaptive rule should resolve uncertain comparisons in both directions.

An independent second 50,000-particle sample, seed `277777`, gave call EVs of +0.225, +0.004, +0.207, and +0.603 chips respectively for these four rows. All four sampling comparisons reproduced; the first, third, and fourth have stable positive modeled EV, while the second remains effectively indifferent.

A practical improvement is to spend extra samples where the action EV gap is small compared with its weighted Monte Carlo uncertainty, especially when many chips are at stake. Low ESS alone is insufficient: the trace's smallest ESS was 28.8 on an obvious river call that needed only 16.25% equity. Spending the entire budget there would not improve the action. The largest-pot fold had ESS 136.9, so an ESS-only threshold could miss it.

The higher-sample checks also support several folds. For example, half 1 hand 13 remained a clear flop fold: history-weighted equity fell from 22.37% to 13.13% against a 33.33% immediate price. Half 2 hands 2, 4, and 10 remained folds. This does not support indiscriminately loosening the bot.

## Range-model hypotheses for independent validation

The baseline scores flop and turn betting likelihoods using current made-hand percentiles. Strong straight or flush draws can have weak current made hands in Omaha; the approach can underrepresent these natural betting hands. This is a mechanism to test with independently generated legal holdings and runouts, rather than reverse-engineering Astra's saved cards.

The baseline also multiplies action likelihoods repeatedly across a hand. Correlated bets can narrow the range too far, particularly with only 192 particles. Sampling more accurately estimates that heuristic; it cannot establish that the heuristic is correct. Any draw-aware feature, likelihood tempering, or adaptation change should be accepted through fresh duplicate-deal benchmarks against multiple playing styles, including earlier bot versions and opponents withheld from parameter selection. Preserve the published baseline for those comparisons.
