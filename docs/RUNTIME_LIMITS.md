# Correction: requested Docker limits and verified limits

During the overnight preflight, Docker reported `CgroupDriver="none"`,
`CgroupVersion="2"` and:

> WARNING: Running in rootless-mode without cgroups. Systemd is required to enable cgroups in rootless-mode.

Earlier runtime reports checked Docker's requested `HostConfig.NanoCpus` and
`HostConfig.Memory` values. Those fields **do not establish actual CPU or
memory enforcement on this cloud Docker service**. Descriptions of the
earlier runs as proven one-CPU/4-GB tests were too strong. Their startup,
thinking-bank and error measurements remain observed results under the
recorded setup, but are insufficient to certify the resource limits.
Read-only filesystem, user and network settings were separately inspected.

This qualification applies to the earlier adaptive and initial neural Docker
measurements, including `results/neural-runtime-validation.json`. That original
report is retained as a historical record rather than silently rewritten.

The new certifier, `tools/certify_overnight.py`, checks actual resource facts
inside the running container. Where cloud cgroups cannot enforce the requested
limits, it uses an OS launcher that pins the bot, independent probe and judge
to the **same single allowed CPU** and sets a hard **4-GiB virtual-address-space
limit per process** before starting Python. It verifies the resulting affinity
and address-space limits and measures aggregate resident memory.

Our bot runs in one process with threads; it launches no child worker
processes. Its per-process address-space limit therefore constrains the bot's
memory. This is a different enforcement mechanism from a shared 4-GiB
container-memory cgroup. The combined bot/probe/judge test must not be described
as an enforced aggregate 4-GiB cgroup. The certification report records that
distinction explicitly and still requires the actual startup, legal-play and
HTTP thinking-bank checks to pass.

An unverified, short-mode, interrupted or failed run cannot produce a full
runtime certificate. The overnight candidate's final resource facts and test
results are written to its run directory's `certification.json`.

The current neural bot completed the full corrected check in
`results/strict-runtime-certification.json`: all actual resource checks passed,
2,000 hands had zero errors, startup was 8.6 seconds, and its two thinking
banks were 3.096/2.976 seconds. Bot peak RSS was 297.4 MiB; sampled combined RSS
was 630.1 MiB. This new record supersedes the earlier enforcement claim while
preserving the distinction between per-process address-space limits and a
container-wide memory cgroup.
