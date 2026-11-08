# Conservative opponent adaptation: sources and design

The goal is to respond to an opponent's tendencies while retaining a stable
fallback. Copying an aggressive opponent's raise frequency is not the objective:
frequent weak bets often justify more calls and fewer speculative reraises.

## Sources inspected online

1. [OpenSpiel: Restricted Nash Response header](https://github.com/google-deepmind/open_spiel/blob/3c5361621d8b0fc05fc0456604d8fada7096a2cb/open_spiel/game_transforms/restricted_nash_response.h)
   and [implementation](https://github.com/google-deepmind/open_spiel/blob/3c5361621d8b0fc05fc0456604d8fada7096a2cb/open_spiel/game_transforms/restricted_nash_response.cc).
   The header cites Johanson et al.'s Restricted Nash Response work. The
   implementation has an initial chance choice between an opponent constrained
   to a modeled policy and a free opponent, with probability `p` versus `1-p`.
   This supplies a concrete research precedent for balancing exploitation with
   robustness instead of assuming the learned opponent model is always right.
   OpenSpiel is Apache-2.0; no source code is copied into our runtime.

2. [Stanford CS228: Bayesian Learning](https://github.com/ermongroup/cs228-notes/blob/b608bab345ec5e5d9949c8f015869a61a440484e/learning/bayesian/index.md).
   These notes derive Beta-Bernoulli and Dirichlet-categorical conjugate updates.
   The posterior uses prior pseudo-counts plus observed counts, so updates take
   constant time. Posterior variance distinguishes an observed 60% tendency
   after ten opportunities from the same percentage after a hundred. This is
   directly useful under the event's eight-second thinking bank.

## Application to this bot

The original policy pools all opponent actions into one aggression rate and all
fold responses into one fold rate. A player who raises many cheap preflops but
only bets strong river hands can therefore be misread. The new model should
separate opportunities by street and distinguish opening bets from raises while
facing an existing bet. It should also retain calls and folds that finish hands;
otherwise the history delivered at the next decision misses terminal responses.

Use small Bayesian count tables with prior pseudo-counts. Where context data is
sparse, blend toward pooled statistics and the stable default. Keep adaptation
small early in a half; a confidence blend such as `n/(n + k)` makes the learning
rate interpretable. Reset all learned statistics between halves.

Use an uncertainty-adjusted fold estimate for bluffing, for example a clipped
posterior mean minus a multiple of posterior standard deviation. This is a
conservative heuristic, not an exact credible bound. Do not increase bluffing
merely because a few early opponents folded.

Against frequent callers, reduce assumed fold equity and choose value bets.
Against frequent indiscriminate bettors, broaden plausible betting ranges and
allow profitable calls without automatically escalating raises. Against passive
opponents, credit rare large bets with more strength. Prefer the smaller raise
when the estimated advantage of a pot raise over a half-pot raise is marginal.
Avoid discarding profitable strong-hand value bets just to reduce a headline
aggression percentage.

## Limits and acceptance criteria

A small confidence-weighted opponent model is inspired by the robustness idea
in RNR, but it does not implement Restricted Nash Response and has no Nash or
bounded-exploitability guarantee. The existing baseline is itself a heuristic.
Showdown observations are selectively revealed, so they cannot be treated as an
unbiased sample of every betting hand. Made-hand ranks also omit some draw
strength; keep action likelihoods soft.

Accept changes based on fresh duplicate-deal comparisons against the frozen
previous bot, independent passive/aggressive/tight/bluffing opponents, and
opponents that change style mid-half. Report betting frequencies by opponent,
including preflop versus later streets, alongside profit and confidence
intervals. Validate legal actions, late-result accounting, fresh-half reset,
and the complete HTTP time bank. Lower aggression alone is not evidence of a
better poker policy.
