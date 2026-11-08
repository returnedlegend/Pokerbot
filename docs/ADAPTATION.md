# Adaptive opponent modeling: final results

This revision responds to learned tendencies rather than mirroring an
opponent's aggression. It preserves participant `jr007` and the event's
unchanged HTTP interface, rules, evaluator and dependencies.

## What changed

- Track opening bets separately from decisions facing a wager, on each street.
- Estimate folding responses by street and wager price. Smooth sparse data
  toward shared priors and discount uncertain fold equity.
- Fade old observations (0.99 per hand, about a 69-hand half-life), including
  pooled fallback rates, so early tight play does not dominate later reads.
- Apply a small uncertainty/risk margin, especially to pot raises. Preserve
  profitable smaller value bets and calls rather than imposing a raise quota.
- Keep every learned statistic in per-half memory; no opponent knowledge
  survives a fresh half. The exact evaluator and fitted preflop model are unchanged.

The methods were informed by Bayesian conjugate models and robustness ideas
in OpenSpiel. [Sources and limitations](OPPONENT_RESEARCH.md) identify what was
used; this is not a Restricted Nash Response solver or an equilibrium guarantee.

## Fresh-deal comparison with v1

Baseline: `50b4682550455b60bf262fe0a09c60dc6269e1a2`.
Development used `adapt-dev1` and `adapt-dev2`. After selecting the policy, it
was frozen before testing `adapt-hold101`, `adapt-hold307`, `adapt-hold809`:
1,000 duplicate deals per seed per profile, with fresh memory in both halves.

**96,000 total hands**: 48,000 for the candidate, 48,000 for v1. The candidate
had **21 match wins, 3 chops, 0 losses, and no illegal actions/exceptions**.
Its three chops were against v1; this does not establish head-to-head superiority.

| Opponent profile | New bb/duplicate deal | Change vs v1 | 95% CI for change |
|---|---:|---:|---|
| Tight/passive | +1.895 | +0.104 | [-0.088, +0.296] |
| Loose/passive | +7.785 | -0.134 | [-0.293, +0.025] |
| Selective/aggressive | +0.728 | +0.055 | [-0.155, +0.266] |
| Loose/aggressive | +4.696 | -0.086 | [-0.622, +0.450] |
| Tight → loose/aggressive | +3.363 | +1.244 | [+0.766, +1.722] |
| Loose/aggressive → tight | +2.861 | +0.297 | [-0.065, +0.659] |
| Fast range-aware | +0.852 | -0.060 | [-0.255, +0.135] |
| Previous v1 bot | -0.001 | -0.001 | [-0.157, +0.155] |

**Pooled improvement excluding selfplay:** +0.203 bb per duplicate deal,
95% CI [+0.052,+0.354]. Including v1: +0.177, CI [+0.043,+0.312].
Because all profiles reuse deals, these intervals cluster by seed/deal:
average differences across profiles first, then calculate the interval across
3,000 deal clusters. They measure this fixed synthetic lineup, not tournament
win probability. No significantly negative profile change was observed, but
the uncertain declines above must not be described as proven equivalence.

**Pot control:** pot raises per actual hand fell from 0.4493 to 0.3571
(**20.5% fewer**). Total raises rose about 1% through more half-pot bets.
This is more selective sizing, not uniformly passive play. Against the
tight-to-aggressive switch, post-switch improvement was +1.849 bb/deal,
CI [+1.080,+2.619]. These subgroup estimates are descriptive; the pooled
comparison above is the broad acceptance check.

## Reliability and deployment

- **50 tests passed**, including 21 focused opponent-model tests for context
  replay, terminal responses, reset, shrinkage, decay and duplicate delivery.
- Full saved cloud build helper succeeded with TLS verification intact.
- Restricted cold startup: **10.8s / 15s**; sample decisions legal in 2–5ms.
- Full official loose-aggressive match in a one-CPU/4GB/no-network/read-only
  container: **+4.973 bb/deal**, CI [+3.972,+5.973], 0 errors/skips, no
  forfeits or bank exhaustion; thinking **4.80/4.01s of 8s**. Bot, opponent
  and judge shared that one CPU. The image added about 202.6MiB.
- Official v1 matchup with fresh server processes between halves: **CHOP**,
  -0.021 bb/deal, CI [-0.274,+0.232], 0 errors. Thinking 6.26/3.76s of 8s;
  the first half ran under substantial concurrent host load.
- The final source/policy/model fingerprints match the tested image. In-process
  benchmark timing excludes HTTP and does not enforce the tournament bank.

Evidence: `results/adaptation-final-validation.json`,
`results/adaptation-runtime-validation.json`, `results/adaptation-container-loose.json`,
`results/adaptation-http-incumbent-combined.txt`. Earlier v1 results remain in
`EVALUATION.md` and are not claims about this revision.

## Reproduce

```sh
.venv/bin/python -m pytest -q
.venv/bin/python tools/evaluate_adaptation.py --phase validation --deals 1000 \
  --opponents tight_passive loose_passive selective_aggressive loose_aggressive \
  changing changing_reverse incumbent fast_range \
  --output results/adaptation-recheck.json
./tools/check_build_cloud.sh
```

The comparator needs the v1 commit in local Git history. For HTTP comparisons,
serve the current bot with `python agent.py` and the old policy with:

```sh
.venv/bin/python tools/serve_strategy.py \
  --revision 50b4682550455b60bf262fe0a09c60dc6269e1a2 --port 5009
```

Then run `judge_engine.py`. Existing validation seeds are already observed:
use new seeds for an independent evaluation after any future tuning.
