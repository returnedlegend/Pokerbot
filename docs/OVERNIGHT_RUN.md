# Six-hour improvement run

Requested work window: **2026-11-08 03:40:08–09:40:08 UTC**. Source audits,
benchmark integration, job testing and the compute search all belong to this
window. The search uses an absolute deadline, including on restart. A final
restricted-runtime certification may finish a few minutes afterward.

The measured neural runtime from commit
`7d51315a8da4fe33bed281f72c055168da594281` is the primary control and fallback.
The earlier `9788300` adaptive bot is a supplementary comparison. An overnight
gain must improve on the current neural bot, not recycle its already-known
improvement over the older version.

## Search and checks

* Copy and hash the complete policy, runtime modules, weights, evaluator,
  scheduler and opponent sources before optimization. Candidates change
  explicit policy parameters and optional features in that frozen code.
* Use three single-threaded CPU workers. Search multiple parameter families
  using successively larger, seat-swapped matches on shared development cards.
  Screen promising candidates across the independent style/range probes,
  additional [challenge policies](CHALLENGE_OPPONENTS.md), and three audited
  [actual public event bots](PUBLIC_BOT_RESEARCH.md).
* Keep the checkraise and shifting challenge profiles completely outside
  development scoring. Include them only after finalist selection.
* Penalize weak matchups and constrain bet sizing. Final acceptance requires
  positive independent field and current-bot head-to-head intervals, no
  material profile downside, no errors, bounded total raises and at most
  1.05 times the current pot-raise frequency. Direct timing is a preliminary
  filter, not the event's HTTP bank check.
* Reserve 50 minutes before the fixed deadline for one irrevocably selected
  finalist on five newly generated protected seeds, 1,000 duplicate deals per
  profile and seed. The seed commitment is recorded before tuning and the
  seeds are opened only after the finalist lock reaches disk. Final failure
  preserves the current bot; it never triggers testing a runner-up.
* Any remaining time verifies the fixed choice without changing parameters.
  A separate helper builds and tests the review bundle under the actual
  verified single-CPU affinity, hard per-process 4-GiB address-space limits,
  read-only filesystem and no network with the official judge. The cloud's
  Docker cgroup requests alone are not enforcement; see
  [RUNTIME_LIMITS.md](RUNTIME_LIMITS.md) for the distinction.

The public opponents retain their original callbacks and clock budgeting.
Some Monte Carlo sample counts therefore depend on load. Source commits and
adapter changes are recorded; these are local strategic comparisons, not a
ranking of all online poker bots. Their source and weights remain in private
benchmark bundles because no explicit redistribution license was found.

Many development trials are selection-biased. The protected finalist gates
apply to one chosen candidate only, with separate field and head-to-head
uncertainty checks. Neither a successful gate nor the synthetic/public lineup
establishes a universal best bot or tournament win probability. If no
candidate meets every gate, the output explicitly preserves the known bot.

## Durable worker and output

The detached worker writes under:

```text
/workspace/poker-overnight-runs/20261108/
  supervisor.json                 process/attempt/certification status
  status.json                     current phase, rounds, cases and hands
  progress.log                    search progress
  manifest.json                   frozen input fingerprints and configuration
  selection-lock.json             irreversible finalist, when chosen
  final-gate.json                  protected results and acceptance checks
  output/decision.json            chosen candidate or preserved baseline
  output/release-candidate.zip     runnable review bundle
  certification.json              actual Docker/HTTP result after completion
```

The supervisor can resume a failed worker only with its original deadline,
snapshots and finalist lock. It stops its own process group at the cutoff.
The original runtime and public GitHub submission are not automatically
replaced by this job. The final bundle is reviewable even when a candidate
fails and the fallback is retained. Check `final-gate.json` **and**
`certification.json` before calling a new candidate deployable.

The launch command is:

```sh
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -u tools/overnight_supervisor.py \
  --run-dir /workspace/poker-overnight-runs/20261108 \
  --deadline 2026-11-08T09:40:08Z --workers 3 --reserve-minutes 50 \
  --extra-opponent-bundle /workspace/poker-overnight-private/public-bundles/armaan-primary \
  --extra-opponent-bundle /workspace/poker-overnight-private/public-bundles/dushyant-byte-me \
  --extra-opponent-bundle /workspace/poker-overnight-private/public-bundles/malki-equitynet
```

It is started with a separate OS session, closed stdin and a file-backed log
so it can continue while this chat is idle in the same running environment.
This is an unattended compute worker, not a continuing language-model
conversation, and it cannot send an automatic chat follow-up. Its files are
the authoritative progress and completion record.

The actual Astra players hit the account usage limit during the previous
[blind exhibition](NEURAL_ASTRA_MATCH.md). No further Astra inference is
available for this job, and no scripted opponent is labeled Astra. The
unfinished 189-hand checkpoint remains separate from training and ranking.
