# Frozen neural revision: evidence and deployment

The upgrade is measurably stronger head-to-head than the previous submitted
bot, `97883009569ad2d383b8145e46ee642262390049`. It retains opponent adaptation
and controlled pot sizing. It does not establish superiority over arbitrary
competition entries or general-purpose AI systems.

## What changed

* Train a compact 115 → 64 → 32 → 1 neural predictor from 80,000 legal flop and
  turn positions, using the event evaluator and open-source scikit-learn.
  Use its uniform-opponent equity estimate as a bounded, covariance-weighted
  anchor for Monte Carlo estimation. Runtime weights occupy 38,287 bytes.
* Correct terminal-call EV: an all-in call cannot incur a future-play penalty.
  Cap the future-loss penalty in near-all-in decisions by the effective chips
  remaining. These fixes do not require extra bluffing or larger bets.
* Separate half-pot and pot-raise thresholds. Preserve the previous stricter
  pot threshold while allowing more inexpensive half-pot value bets.
* Set default numerical-library thread counts before importing NumPy so the
  small network stays within the single-CPU deployment budget.

These are newly trained simulation weights, not a fine-tuned ChatGPT model or
a downloaded pretrained event betting policy. The original open-source Monte
Carlo foundation remains. Training provenance, calibration and reproduction
are in [NEURAL_MODEL.md](NEURAL_MODEL.md).

Development examined more particles, draw-aware opponent ranking, showdown
learning, likelihood changes, altered preflop thresholds and a small CFR+
river abstraction. These additions did not produce a reliable wider-field
improvement and remain disabled in `policy.json`. Development ablations also
did not establish a significant incremental match gain from the neural anchor
alone. The head-to-head gain below belongs to the complete selected policy;
it cannot honestly be attributed to the network alone. Exact development
manifests are in `results/strength-selection-development-summary.json`.

## Fresh validation

The candidate source, weights and policy were frozen before validation.
`results/strength-final-validation.json` merges two separate workers with
identical candidate fingerprints. There was no cached baseline reuse, resumed
match, source exception or parameter update during validation.

Ten profiles were tested on reserved seeds `strength-hold211`,
`strength-hold503` and `strength-hold907`, with 1,000 duplicate deals per profile
and seed. Both candidate and committed baseline played each external profile
on the same cards. Each duplicate deal consists of two hands with seats
swapped and fresh memory at the start of each 1,000-hand half. This totals
120,000 newly simulated actual hands, including the baseline's self-match for
the incumbent comparison. All participants had zero recorded errors.

**Units:** bb per duplicate deal is the sum of both seat-swapped hand nets,
divided by the two-chip big blind. Divide by two for bb per actual hand.
For example, +0.37733 bb per duplicate deal equals +18.87 bb per 100 actual
hands. It does not mean +37.73 bb per 100 hands.

| Opponent profile | Candidate bb / duplicate deal | 95% whole-seed interval |
| --- | ---: | ---: |
| Tight passive | +1.697 | [+1.418, +1.976] |
| Loose passive | +7.202 | [+6.374, +8.029] |
| Selective aggressive | +0.951 | [+0.605, +1.298] |
| Loose aggressive | +3.921 | [+2.868, +4.974] |
| Changing style | +2.834 | [+2.007, +3.661] |
| Changing style, reverse order | +2.438 | [+1.811, +3.064] |
| Independent fast range | +1.157 | [+1.067, +1.247] |
| Independent draw range | +1.058 | [+0.930, +1.185] |
| Adaptive pressure | +0.432 | [−0.327, +1.191] |
| Previous submitted bot | **+0.377** | **[+0.139, +0.616]** |

The previous-bot results were positive on every reserved seed: +0.315,
+0.488 and +0.329 bb per duplicate deal. The card-paired interval is
[+0.225, +0.530], and the 50-deal block-bootstrap interval is [+0.228, +0.532].
The whole-seed Student-t interval above uses only three seed trajectories;
even a positive interval cannot establish performance on every opponent.

Against the nine external profiles, the equally weighted raw candidate mean
was +2.410 bb per duplicate deal. The candidate's paired improvement over the
old bot was only +0.07469, with card-paired interval [−0.03910, +0.18847] and
whole-seed interval [−0.11460, +0.26397]. **A broader external-field improvement
is unproven.** Both versions already exploit many of these synthetic profiles.
The profile table describes individual intervals, with no family-wise
multiple-comparison guarantee. These opponents are useful probes, not the
unknown tournament field.

Opponent learning creates serial dependence. The report therefore includes
card-cluster, 50-deal block-bootstrap and whole-seed intervals. Profiles share
cards, so pooled estimates average profiles within each seed/deal rather than
pretending the profile results are independent. The overall paired gain across
all ten profiles is +0.10495; its block interval [−0.00414, +0.21296] and seed
interval [−0.06766, +0.27756] include zero. The narrower card-only interval is
not used to claim a decisive league-wide improvement.

Pot-sized raises fell from 0.33573 to 0.32585 per actual hand, **2.94% fewer**.
Total raises rose from 0.96575 to 1.11448 per hand, **15.4% more**, through
smaller half-pot raises. This is more selective pressure at the cheaper size,
not a claim that the total aggression rate fell.

Direct-function validation does not enforce the HTTP thinking bank. Runtime
verification is separate below. The independent slow probe and the actual
ChatGPT Astra exhibition are reported separately so their different conditions
are not pooled into the synthetic league.

An additional independent probe used 512 particles per decision and separate
12-reference draw estimates, with selective reraises, traps and blocker-based
river bluffs. It imports no production strategy, learned weights or adaptation
code. It was frozen before selection but is deliberately slower than the
event time bank; the label "expert" in its filename is not a strength guarantee.
On 200 duplicate deals per reserved seed, the candidate won **+1.1775 bb per
duplicate deal**, whole-seed interval **[+0.3690, +1.9860]**, with zero errors.
Its paired improvement over the baseline was +0.2867, interval
[−0.5557, +1.1290]; incremental improvement here remains unproven. The report is
`results/strength-final-slow-expert-validation.json`.

Together, these final comparisons newly simulated **122,400 actual hands**:
61,200 for the selected bot and 61,200 for the frozen baseline. No policy was
retuned after either final validation began. The slow probe reuses the same
reserved seed/deal prefixes for comparability; these are additional played
hands, not additional independent seed trajectories.

## Direct ChatGPT Astra exhibition

The planned 200-hand blind exhibition against actual `gpt-6-astra` contexts
stopped at 189 completed hands when two players hit the account usage limit.
The bot was provisionally behind; the incomplete, unequal duplicate halves
cannot establish a complete match result. No other policy finished those
players' hands. [NEURAL_ASTRA_MATCH.md](NEURAL_ASTRA_MATCH.md) records every
partial seat score, the limit and the audit limitation. Superiority over Astra
is not established.

## Restricted deployment

The selected policy passed **92 tests**. The organizers' Dockerfiles and
official game/referee files remain unchanged. The managed-cloud build helper
uses temporary CA secret mounts while preserving TLS verification.

`results/neural-runtime-validation.json` records the image and runtime hashes.
The image ran as UID 10001 with no network, a read-only root filesystem and a
writable `/tmp` tmpfs. The first test requested one CPU and 4 GB through Docker;
the later [resource-enforcement audit](RUNTIME_LIMITS.md) found that the cloud
daemon does not enforce those cgroup requests. Those initial measurements are
therefore not proof of the one-CPU/4-GB allowance. Cold startup took
**7.3 seconds of the 15-second allowance**. A full official-judge match against
the independent loose-aggressive probe played 1,000 duplicate deals, or 2,000
hands, with zero errors, skipped calls, timeouts or forfeits. Bot thinking
banks were **5.554 and 4.218 seconds of 8 seconds** under that original setup.
The process was not restarted between the
two halves; learning was reset through the protocol's `/new-half` endpoint.

The overnight certifier adds verified OS CPU affinity and hard per-process
address-space limits before repeating the actual startup and full HTTP match.
Its certificate names the enforcement mechanism and reports measured memory;
it does not treat Docker's requested configuration as proof.

The completed stricter check is recorded in
`results/strict-runtime-certification.json`. Every sampled bot, probe and judge
thread used the same single CPU, and each process had a hard 4-GiB
address-space limit. Cold startup was **8.6 seconds**; a separate fresh process
and smoke check took **9.003 seconds**, measured from the kernel's process
start time and excluding Docker filesystem creation. The full 2,000-hand HTTP
match had zero errors, skips, timeouts or forfeits. Bot thinking banks were
**3.096 and 2.976 seconds of 8 seconds**. Bot peak resident memory was **297.4
MiB**; sampled aggregate resident memory was **630.1 MiB**. No bot child
processes were observed. A shared 4-GiB cgroup is not claimed.

That independent loose-aggressive runtime probe yielded +5.485 bb per
duplicate deal, interval [+4.444, +6.526]. Its cards differ from the earlier
Docker match, so the difference between their chip scores is not evidence of
a strategy improvement. The policy and model hashes were unchanged.

The one-probe Docker score was +3.891 bb per duplicate deal, approximate
card-paired interval [+2.801, +4.981]. This verifies a deployable winning match
against that probe; it adds no independent evidence about unknown entrants.

For local reproduction, run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python tools/evaluate_strength.py --help
./tools/check_build_cloud.sh
```

The paired development/validation comparator needs the baseline commit in
Git history. The tournament bot and official judge do not need that history.
