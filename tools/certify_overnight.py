#!/usr/bin/env python3
"""Certify a frozen overnight ZIP under the organizer's actual CPU constraints.

Never edits or promotes the deployed bot. The default full check returns zero
only after a restricted cold startup and 1,000-duplicate-deal HTTP match pass.
Short development modes return 2 even when their limited checks pass.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CORE = ("agent.py", "strategy.py", "opponent_model.py", "policy.json", "game.py",
        "evaluator.py", "equity_engine.py", "preflop_model.npz", "equity_network.py",
        "equity_network.npz", "river_solver.py", "river_policy.npz", "requirements.txt",
        "docker/Dockerfile.base", "docker/Dockerfile.agent")
HELPERS = ("tools/check_build_cloud.sh", "tools/smoke_test.py", "judge_engine.py",
           "baseline-bots/baseline_random.py", "tools/style_opponents.py")
MAX_ZIP_BYTES = 128 * 1024 * 1024
DOCKER = ["docker", "--host=unix:///var/run/docker.sock"]

OPPONENT = '''import importlib.util
import sys
sys.path.insert(0, "/app")
sys.path.insert(0, "/checks")
from equity_engine import warmup
from style_opponents import StyleOpponent
warmup()
spec = importlib.util.spec_from_file_location("cert_style_server", "/checks/baseline_random.py")
template = importlib.util.module_from_spec(spec)
spec.loader.exec_module(template)
bot = StyleOpponent("loose_aggressive")
template.new_half = bot.new_half
template.choose_action = bot.choose_action
template.on_hand_end = bot.on_hand_end
template.PARTICIPANT = "Independent certification probe"
template.AGENT_NAME = "loose_aggressive"
import uvicorn
uvicorn.run(template.app, host="0.0.0.0", port=5009, access_log=False)
'''

LIMITER = '''import json, os, resource, sys
from pathlib import Path
cpu = int(os.environ["POKER_CERT_CPU"])
if cpu not in os.sched_getaffinity(0):
    raise RuntimeError("Certification CPU is not in the allowed affinity set")
os.sched_setaffinity(0, {cpu})
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
if sys.argv[1:] == ["--facts"]:
    def read(name):
        path = Path("/sys/fs/cgroup") / name
        return path.read_text().strip() if path.exists() else None
    status = Path("/proc/self/status").read_text().splitlines()
    print(json.dumps({"affinity": sorted(os.sched_getaffinity(0)),
                      "rlimit_as": list(resource.getrlimit(resource.RLIMIT_AS)),
                      "cpus_allowed_list": next(line.split(":",1)[1].strip() for line in status if line.startswith("Cpus_allowed_list:")),
                      "cgroup_cpu_max": read("cpu.max"), "cgroup_memory_max": read("memory.max")}))
else:
    script = sys.argv[2] if len(sys.argv) > 2 else ""
    role = {"agent.py": "bot", "/checks/opponent.py": "probe", "/checks/judge_engine.py": "judge", "/checks/monitor.py": "monitor"}.get(script, "management")
    marker = Path("/tmp/certification-role-" + str(os.getpid()) + ".json")
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": os.getpid(), "ppid": os.getppid(), "role": role,
                                    "affinity": sorted(os.sched_getaffinity(0)),
                                    "rlimit_as": list(resource.getrlimit(resource.RLIMIT_AS))}))
    temporary.replace(marker)
    os.execvp(sys.argv[1], sys.argv[1:])
'''

MONITOR = '''import json, os, resource, time
from pathlib import Path
cpu = int(os.environ["POKER_CERT_CPU"])
roles = {}
peak_rss = 0
max_children = 0
child_audit_errors = 0
samples = 0
parents = {}
registered = {}
deadline = time.monotonic() + 210
while not Path("/tmp/certification-done").exists() and time.monotonic() < deadline:
    total = 0
    parents = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            status = dict(line.split(":", 1) for line in (entry / "status").read_text().splitlines() if ":" in line)
            parents[pid] = int(status["PPid"])
            total += int(status.get("VmRSS", "0 kB").split()[0]) * 1024
            command = (entry / "cmdline").read_bytes().split(b"\\0")
            script = command[1] if len(command) > 1 else b""
            role = "bot" if pid == 1 else ("probe" if script == b"/checks/opponent.py" else ("judge" if script == b"/checks/judge_engine.py" else None))
            if role:
                affinity = sorted(os.sched_getaffinity(pid))
                limits = list(resource.prlimit(pid, resource.RLIMIT_AS))
                current = roles.setdefault(role, {"pids": [], "all_samples_bounded": True})
                if pid not in current["pids"]:
                    current["pids"].append(pid)
                current.update(affinity=affinity, rlimit_as=limits, cpus_allowed_list=status.get("Cpus_allowed_list", "").strip())
                thread_masks = [sorted(os.sched_getaffinity(int(task.name))) for task in (entry / "task").iterdir()]
                current["all_samples_bounded"] &= affinity == [cpu] and limits == [4 * 1024**3, 4 * 1024**3] and all(mask == [cpu] for mask in thread_masks)
                current["peak_rss_bytes_sampled"] = max(current.get("peak_rss_bytes_sampled", 0), int(status.get("VmRSS", "0 kB").split()[0]) * 1024)
                current["peak_rss_bytes_kernel"] = max(current.get("peak_rss_bytes_kernel", 0), int(status.get("VmHWM", "0 kB").split()[0]) * 1024)
        except (OSError, ValueError, ProcessLookupError):
            pass
    for marker in Path("/tmp").glob("certification-role-*.json"):
        try:
            item = json.loads(marker.read_text())
            if item["role"] in ("probe", "judge", "monitor", "management") and item["affinity"] == [cpu] and item["rlimit_as"] == [4 * 1024**3, 4 * 1024**3]:
                registered[int(item["pid"])] = item
        except (OSError, ValueError, KeyError):
            pass
    if 1 not in parents:
        child_audit_errors += 1
    children = [pid for pid, parent in parents.items() if parent == 1 and pid not in registered]
    max_children = max(max_children, len(children))
    peak_rss = max(peak_rss, total)
    samples += 1
    time.sleep(.1)
result = {"samples": samples, "aggregate_peak_rss_bytes_sampled": peak_rss,
          "aggregate_memory_cgroup_enforced": False, "bot_max_child_processes": max_children,
          "bot_child_audit_errors": child_audit_errors,
          "process_parent_mapping_last_sample": parents,
          "registered_external_processes": registered,
          "child_audit_method": "PPid from /proc/status; exclude only certification-launcher-registered probe/judge/monitor/management helpers, which are started externally through Dockerexec.",
          "roles": roles, "sampling_note": "100ms sampled RSS across all container processes, including this monitor; not a hard aggregate RSS cap."}
Path("/tmp/certification-memory.json").write_text(json.dumps(result))
'''

STARTUP_CLOCK = '''import json, os, time
from pathlib import Path
fields = Path("/proc/1/stat").read_text().rpartition(")")[2].split()
ticks = int(fields[19])
hz = os.sysconf("SC_CLK_TCK")
now = time.clock_gettime(time.CLOCK_BOOTTIME)
print(json.dumps({"pid1_start_ticks": ticks, "clock_ticks_per_second": hz,
                  "clock": "CLOCK_BOOTTIME", "process_startup_and_smoke_seconds": now - ticks / hz}))
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def verify_package(archive: Path, decision_path: Path) -> tuple[dict, dict[str, bytes]]:
    """Verify exact declared bytes, with no path traversal or untracked code."""
    decision = json.loads(decision_path.read_text())
    if archive.stat().st_size > MAX_ZIP_BYTES:
        raise ValueError("Release ZIP exceeds the bounded certification input size")
    raw = archive.read_bytes()
    if digest(raw) != decision.get("archive_sha256"):
        raise ValueError("Release ZIP hash differs from decision.json")
    files = decision.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Missing release file manifest")
    payload = {}
    with zipfile.ZipFile(archive) as handle:
        infos = [item for item in handle.infolist() if not item.is_dir()]
        if len({item.filename for item in infos}) != len(infos):
            raise ValueError("Duplicate ZIP entry")
        if sum(item.file_size for item in infos) > MAX_ZIP_BYTES:
            raise ValueError("Release ZIP exceeds the bounded certification input size")
        for item in infos:
            name = item.filename
            path = PurePosixPath(name)
            if path.as_posix() != name or path.as_posix() in payload:
                raise ValueError("Noncanonical or duplicate ZIP path")
            if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
                raise ValueError("Unsafe ZIP path")
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError("ZIP symlinks are not permitted")
            if name not in files and name not in ("MANIFEST.json", "README.md"):
                raise ValueError("Unmanifested release file: " + name)
            payload[name] = handle.read(item)
        if handle.testzip() is not None:
            raise ValueError("Release ZIP CRC failed")
    if set(files) - set(payload):
        raise ValueError("Manifest declares missing release files")
    for name, expected in files.items():
        if digest(payload[name]) != expected:
            raise ValueError("Release file hash mismatch: " + name)
    for name in CORE:
        if name not in files:
            raise ValueError("Required runtime file is not manifested: " + name)
    embedded = json.loads(payload["MANIFEST.json"])
    if embedded != {key: value for key, value in decision.items() if key != "archive_sha256"}:
        raise ValueError("Embedded and external release manifests differ")
    return decision, payload


def clean_stats(stats):
    return len(stats) == 2 and all(
        item.get("errors") == 0 and item.get("skipped_calls") == 0
        and not item.get("timed_out") and not item.get("forfeited")
        and 0 <= item.get("time_seconds", 9) <= 8
        for item in stats)


def certify(args):
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report_path = directory / "certification.json"
    artifacts = directory / "certification-artifacts"
    artifacts.mkdir(exist_ok=True)
    token = "jr007-cert-" + uuid.uuid4().hex[:12]
    cert_cpu = min(os.sched_getaffinity(0))
    base_tag, bot_tag = token + "-base:local", token + "-bot:local"
    container = token + "-match"
    smoke_container = token + "-build-smoke"
    environment = {key: value for key, value in os.environ.items() if key not in
                   ("DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_TLS", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH")}
    report = {"phase": "preflight", "status": "running", "runtime_verified": False,
              "auto_promoted": False, "original_runtime_modified": False,
              "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "scope": "Runtime certification only; statistical selection remains in output/decision.json.",
              "deals_per_half": args.deals, "smoke_only": args.smoke_only,
              "build_tags": {"base": base_tag, "bot": bot_tag}, "container": container}
    stage = None
    created = False

    def save(**fields):
        report.update(fields)
        atomic(report_path, report)

    def execute(command, log=None, timeout=600, **kwargs):
        if log is None:
            return subprocess.run(command, env=environment, check=True, timeout=timeout, **kwargs)
        with (artifacts / log).open("wb") as output:
            return subprocess.run(command, env=environment, check=True, timeout=timeout,
                                  stdout=output, stderr=subprocess.STDOUT, **kwargs)

    def docker(*command, **kwargs):
        return execute(DOCKER + list(command), **kwargs)

    def inspect(*command):
        return json.loads(docker(*command, capture_output=True, text=True).stdout)[0]

    save()
    try:
        archive = args.candidate_zip or directory / "output/release-candidate.zip"
        decision_path = directory / "output/decision.json"
        decision, payload = verify_package(archive, decision_path)
        save(archive_sha256=decision["archive_sha256"], chosen=decision.get("chosen"),
             decision_sha256=digest(decision_path.read_bytes()),
             statistics_gates_passed=bool(decision.get("statistics_gates_passed")),
             runtime_sha256=decision["files"], package_verified=True)
        # The tournament referee/build recipes must stay the known official ones.
        for name in ("game.py", "evaluator.py", "docker/Dockerfile.base", "docker/Dockerfile.agent"):
            if payload[name] != (ROOT / name).read_bytes():
                raise ValueError("Candidate changed an official rule/build file: " + name)
        helper_bytes = {name: (ROOT / name).read_bytes() for name in HELPERS}
        save(checker_sha256={name: digest(data) for name, data in helper_bytes.items()})
        daemon = json.loads(docker("info", "--format",
            '{"cgroup_driver":{{json .CgroupDriver}},"cgroup_version":{{json .CgroupVersion}},"warnings":{{json .Warnings}}}',
            capture_output=True, text=True).stdout)
        save(docker_daemon=daemon, actual_enforcement="shared_one_CPU_affinity_and_per_process_4GiB_address_space",
             memory_scope="4GiB hard virtual-address-space limit on each single-process bot/probe/judge; no claim of a shared4GiB cgroup when rootless Docker cannot enforce it.")
        stage = Path(tempfile.mkdtemp(prefix=token + "-", dir=artifacts))
        source, checks = stage / "source", stage / "checks"
        source.mkdir()
        checks.mkdir()
        for name, data in payload.items():
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        for name, data in helper_bytes.items():
            if name == "tools/check_build_cloud.sh":
                target = stage / "build-helper.sh"
            else:
                target = checks / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        (checks / "opponent.py").write_text(OPPONENT)
        (checks / "limited_launcher.py").write_text(LIMITER)
        (checks / "monitor.py").write_text(MONITOR)
        helper = stage / "build-helper.sh"
        text = helper.read_text()
        if len(re.findall(r"^POKER_ROOT=.*$", text, re.MULTILINE)) != 1:
            raise ValueError("Build helper changed; review isolated context substitution")
        text = re.sub(r"^POKER_ROOT=.*$", "POKER_ROOT=" + shlex.quote(str(source)),
                      text, count=1, flags=re.MULTILINE)
        for old, new in (("poker-agent-base:local", base_tag),
                         ("poker-agent-cloud-check:latest", bot_tag),
                         ('POKER_CONTAINER="poker-cloud-check-$$"', 'POKER_CONTAINER="' + smoke_container + '"')):
            if text.count(old) != 1:
                raise ValueError("Build helper changed; review unique-image substitution")
            text = text.replace(old, new, 1)
        helper.write_text(text)
        # Rootless Docker can accept resource flags without enforcing cgroups.
        # Apply real OS limits before every tested Python process starts.
        target = '"$POKER_BOT_TAG" >/dev/null'
        replacement = ('--volume ' + shlex.quote(str(checks) + ":/checks:ro")
                       + ' --env POKER_CERT_CPU=' + str(cert_cpu)
                       + ' "$POKER_BOT_TAG" python /checks/limited_launcher.py python agent.py >/dev/null')
        if text.count(target) != 1 or text.count('python /smoke/smoke_test.py') != 1:
            raise ValueError("Build helper changed; review bounded-launch substitution")
        text = text.replace(target, replacement).replace('python /smoke/smoke_test.py',
                'python /checks/limited_launcher.py python /smoke/smoke_test.py', 1)
        helper.write_text(text)
        # The original helper expects tools/smoke_test.py under its source root.
        (source / "tools").mkdir(exist_ok=True)
        (source / "tools/smoke_test.py").write_bytes(helper_bytes["tools/smoke_test.py"])
        for folder in [stage, *stage.rglob("*")]:
            folder.chmod(0o755 if folder.is_dir() else 0o644)
        save(phase="build_and_cold_smoke")
        execute(["bash", str(helper)], log="build-smoke.log", timeout=600)
        build_log = (artifacts / "build-smoke.log").read_text()
        match = re.search(r"GET / answered after ([0-9.]+)\s*s", build_log)
        if not match or "SMOKE TEST PASSED" not in build_log:
            raise ValueError("No successful cold smoke measurement in build output")
        cold = float(match.group(1))
        if cold > 15:
            raise ValueError("Cold startup exceeded 15 seconds")
        base, bot = inspect("image", "inspect", base_tag), inspect("image", "inspect", bot_tag)
        added = bot["Size"] - base["Size"]
        if added > 3 * 1024**3:
            raise ValueError("Bot exceeds 3GiB added-image limit")
        save(build={"base_image": base["Id"], "bot_image": bot["Id"], "base_bytes": base["Size"],
                    "bot_bytes": bot["Size"], "added_bytes": added,
                    "organizer_dockerfiles_unchanged": True,
                    "tls_verification": "Preserved through cloud helper temporary CA secret mounts; unique tags and OS-limited test launch only. Organizer Dockerfiles unchanged.",
                    "log": "certification-artifacts/build-smoke.log"},
             smoke={"status": "passed", "cold_start_seconds": cold, "startup_limit_seconds": 15})
        if args.smoke_only:
            save(status="smoke_only_passed", phase="complete", runtime_verified=False,
                 reason="Development short mode did not run the full event-length HTTP match.")
            return 2
        save(phase="restricted_http_match")
        begin = time.monotonic()
        docker("run", "--detach", "--name", container, "--network", "none", "--read-only",
               "--tmpfs", "/tmp:rw,nosuid,nodev,size=512m", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--pids-limit", "256", "--cpus", "1",
               "--memory", "4g", "--env", "PORT=5008", "--env", "AGENT_TOKEN=certification-local",
               "--env", "AGENT1_TOKEN=certification-local", "--env", "AGENT2_TOKEN=certification-local",
               "--env", "POKER_CERT_CPU=" + str(cert_cpu), "--volume", str(checks) + ":/checks:ro",
               bot_tag, "python", "/checks/limited_launcher.py", "python", "agent.py", stdout=subprocess.DEVNULL)
        created = True
        creation_wall = time.monotonic() - begin
        facts = inspect("inspect", container)
        host = facts["HostConfig"]
        restrictions = {"network_mode": host["NetworkMode"], "readonly_rootfs": host["ReadonlyRootfs"],
                        "nano_cpus": host["NanoCpus"], "memory_bytes": host["Memory"],
                        "user": facts["Config"]["User"], "cap_drop": host["CapDrop"],
                        "security_opt": host["SecurityOpt"], "pids_limit": host["PidsLimit"],
                        "tmpfs": host["Tmpfs"]}
        if not (host["NetworkMode"] == "none" and host["ReadonlyRootfs"]
                and host["NanoCpus"] == 1000000000 and host["Memory"] == 4294967296
                and facts["Config"]["User"] == "10001:10001" and "ALL" in host["CapDrop"]
                and "no-new-privileges" in host["SecurityOpt"]):
            raise ValueError("Container restriction verification failed")
        docker("exec", container, "python", "/checks/limited_launcher.py", "python", "/checks/smoke_test.py", "--port", "5008", "--startup", "15",
               log="fresh-process-smoke.log", timeout=25)
        # VFS Docker can spend seconds cloning image files before PID1 exists.
        # The event startup allowance concerns the running server process.
        startup_clock = json.loads(docker("exec", container, "python", "/checks/limited_launcher.py",
                                          "python", "-c", STARTUP_CLOCK, capture_output=True, text=True).stdout)
        restart_readiness = startup_clock["process_startup_and_smoke_seconds"]
        save(container_creation_wall_seconds=creation_wall, startup_clock=startup_clock,
             fresh_process_and_smoke_seconds=restart_readiness)
        if restart_readiness > 15:
            raise ValueError("Fresh process plus smoke exceeded the 15-second startup allowance")
        probe = ("import hashlib,json; from pathlib import Path; names=" + repr(list(decision["files"]))
                 + "; print(json.dumps({n:hashlib.sha256((Path('/app')/n).read_bytes()).hexdigest() for n in names}))")
        actual = json.loads(docker("exec", container, "python", "-c", probe,
                                  capture_output=True, text=True).stdout)
        if actual != decision["files"]:
            raise ValueError("Tested image runtime differs from selected release bytes")
        bounded = json.loads(docker("exec", container, "python", "/checks/limited_launcher.py", "--facts",
                                    capture_output=True, text=True).stdout)
        if bounded["affinity"] != [cert_cpu] or bounded["rlimit_as"] != [4 * 1024**3] * 2:
            raise ValueError("Actual CPU affinity or address-space limit differs from the requested OS bound")
        save(runtime={"cpu": 1, "memory": "4g", "verified_container_configuration": restrictions},
             actual_os_limits=bounded, image_runtime_matches_package=True, fresh_process_and_smoke_seconds=restart_readiness)
        docker("exec", "--detach", container, "python", "/checks/limited_launcher.py", "python", "/checks/monitor.py",
               stdout=subprocess.DEVNULL)
        docker("exec", "--detach", "--env", "PYTHONPATH=/app:/checks", container,
               "python", "/checks/limited_launcher.py", "python", "/checks/opponent.py", stdout=subprocess.DEVNULL)
        docker("exec", "--env", "PYTHONPATH=/app:/checks", container, "python", "/checks/limited_launcher.py", "python", "/checks/judge_engine.py",
               "--deals", str(args.deals), "--seed", args.seed, "--json", "/tmp/certification-match.json",
               log="http-match.log", timeout=180)
        raw = docker("exec", container, "cat", "/tmp/certification-match.json", capture_output=True).stdout
        (artifacts / "http-match.json").write_bytes(raw)
        result = json.loads(raw)
        docker("exec", container, "python", "/checks/limited_launcher.py", "python", "-c", "from pathlib import Path; Path('/tmp/certification-done').touch()", stdout=subprocess.DEVNULL)
        time.sleep(.3)
        measured = json.loads(docker("exec", container, "cat", "/tmp/certification-memory.json", capture_output=True, text=True).stdout)
        stats, other = result["agent_stats"]["agent1"], result["agent_stats"]["agent2"]
        checks_passed = {"cold_start_within_15_seconds": cold <= 15,
                         "fresh_process_startup_within_15_seconds": restart_readiness <= 15,
                         "image_runtime_matches_package": actual == decision["files"],
                         "bot_banks_clean_and_within_8_seconds": clean_stats(stats),
                         "probe_banks_clean_and_within_8_seconds": clean_stats(other),
                         "full_1000_deal_duplicate_match": args.deals == 1000 and result["num_deals"] == 1000,
                         "all_actual_processes_one_CPU_and_4GiB_address_space": set(measured["roles"]) == {"bot", "probe", "judge"} and all(role["all_samples_bounded"] for role in measured["roles"].values()),
                         "single_process_bot_no_children_observed": measured["bot_max_child_processes"] == 0 and measured["bot_child_audit_errors"] == 0,
                         "release_archive_unchanged": digest(archive.read_bytes()) == decision["archive_sha256"],
                         "selection_decision_unchanged": digest(decision_path.read_bytes()) == report["decision_sha256"]}
        http_match = {"opponent": "independent loose_aggressive StyleOpponent", "seed": args.seed,
                         "deals_per_half": result["num_deals"], "total_hands": 2 * result["num_deals"],
                         "verdict": result["verdict"], "mean_bb_per_deal": result["mean_bb_per_deal"],
                         "ci95": result["ci"], "agent_stats": stats, "opponent_stats": other,
                         "result": "certification-artifacts/http-match.json", "log": "certification-artifacts/http-match.log",
                         "resource_note": "Bot, independent probe and official judge share one CPU via actual affinity. Each process has a hard4GiB address-space limit; rootless Docker's requested4GiB aggregate cgroup is not claimed as enforced. New-half resets learning; a separate fresh-process cold smoke also passed."}
        save(validation_checks=checks_passed, memory_measurement=measured, http_match=http_match)
        failed = [name for name, passed in checks_passed.items() if not passed and name != "full_1000_deal_duplicate_match"]
        if failed:
            raise ValueError("Runtime checks failed: " + ", ".join(failed))
        full = all(checks_passed.values())
        save(phase="complete", status="certified" if full else "quick_match_passed",
             runtime_verified=full, ready_for_promotion=full and bool(decision.get("statistics_gates_passed")))
        return 0 if full else 2
    except BaseException as exc:
        save(status="failed", phase="failed", runtime_verified=False, ready_for_promotion=False,
             error=f"{type(exc).__name__}: {exc}", retain_frozen_current=True)
        if created:
            try:
                docker("logs", "--tail", "80", container, log="container-error.log", timeout=10)
            except Exception:
                pass
        print(report["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        # The supervisor grants 20 seconds after TERM. Remove daemon-owned
        # containers first; image-tag cleanup is secondary and tightly bounded.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        cleanup = [(DOCKER + ["rm", "-f", container, smoke_container], 10)]
        if not args.keep_images:
            cleanup.append((DOCKER + ["image", "rm", bot_tag, base_tag], 5))
        for command, limit in cleanup:
            try:
                subprocess.run(command, env=environment, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=limit)
            except Exception as exc:
                report.setdefault("cleanup_warnings", []).append(f"{type(exc).__name__}: {exc}")
        if stage is not None and not args.keep_stage:
            shutil.rmtree(stage, ignore_errors=True)
        report["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate-zip", type=Path)
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--seed", default="overnight-runtime-certification-20261108")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--keep-images", action="store_true")
    parser.add_argument("--keep-stage", action="store_true")
    args = parser.parse_args()
    if args.deals < 1:
        parser.error("--deals must be positive")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise InterruptedError("Certification interrupted by signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (args.run_dir / ".certification.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Certification for this run is already active", file=sys.stderr)
            return 1
        return certify(args)


if __name__ == "__main__":
    raise SystemExit(main())
