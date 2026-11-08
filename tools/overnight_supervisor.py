"""Run the bounded search as a detached job, then certify its review bundle.

This process never promotes code, pushes Git, or calls a language-model API.
Retries preserve the original deadline, captured inputs and selection lock.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STOP = False
CERTIFICATION_FILES = ("tools/overnight_supervisor.py", "tools/certify_overnight.py", "tools/check_build_cloud.sh", "tools/smoke_test.py",
                       "judge_engine.py", "baseline-bots/baseline_random.py", "tools/style_opponents.py",
                       "docker/Dockerfile.base", "docker/Dockerfile.agent", "game.py", "evaluator.py")


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def process_start_ticks(pid):
    try:
        # The parenthesized process name may itself contain spaces.
        return Path(f"/proc/{int(pid)}/stat").read_text().rpartition(")")[2].split()[19]
    except (OSError, ValueError, IndexError):
        return None


def scheduler_active(directory, previous=None):
    with (directory / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(lock, fcntl.LOCK_UN)
    if previous and previous.get("child_pid"):
        pid = previous["child_pid"]
        actual = process_start_ticks(pid)
        expected = previous.get("child_start_ticks")
        if actual is not None and (expected is None or actual == expected):
            try:
                command = Path(f"/proc/{int(pid)}/cmdline").read_bytes().decode().split("\0")
                owned = {str(ROOT / "tools/overnight_search.py"),
                         str(directory / "frozen/tools/overnight_search.py"),
                         str(ROOT / "tools/certify_overnight.py")}
                if any(script in command for script in owned) and "--run-dir" in command:
                    return Path(command[command.index("--run-dir") + 1]).resolve() == directory.resolve()
            except (OSError, UnicodeError, ValueError, IndexError):
                pass
    return False


def choose_scheduler(directory):
    captured = directory / "frozen/tools/overnight_search.py"
    original = ROOT / "tools/overnight_search.py"
    if (directory / "status.json").is_file() and (directory / "manifest.json").is_file():
        metadata = json.loads((directory / "manifest.json").read_text())
        expected = metadata["files"]["frozen/tools/overnight_search.py"]
        if hashlib.sha256(captured.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Captured scheduler fingerprint changed")
        return captured
    if captured.is_file() and original.read_bytes() != captured.read_bytes():
        raise RuntimeError("Initialization was interrupted and original scheduler changed")
    return original


def certification_status(code, report=None):
    if code is None:
        return "skipped"
    if (code == 0 and report and report.get("status") == "certified"
            and report.get("runtime_verified") is True and report.get("validation_checks")
            and all(report["validation_checks"].values())):
        return "passed"
    return "failed"


def freeze_certification_inputs(directory):
    target = directory / "control/certification-inputs.json"
    if target.is_file():
        metadata = json.loads(target.read_text())
    else:
        files = {}
        for name in CERTIFICATION_FILES:
            data = (ROOT / name).read_bytes()
            snapshot = directory / "control" / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            snapshot.write_bytes(data)
            files[name] = hashlib.sha256(data).hexdigest()
        metadata = {"files_sha256": files, "captured_at": now()}
        atomic(target, metadata)
    for name, digest in metadata["files_sha256"].items():
        for path in (ROOT / name, directory / "control" / name):
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise RuntimeError("Certification input changed: " + str(path))
    return metadata


def stop_child(process):
    """Signal only the process group created for this exact owned child."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def run_child(command, directory, log_name, cutoff, state):
    environment = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        environment[key] = "1"
    with (directory / log_name).open("a") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        state.update(child_pid=process.pid, child_start_ticks=process_start_ticks(process.pid), updated_at=now())
        atomic(directory / "supervisor.json", state)
        while process.poll() is None:
            if STOP or time.time() >= cutoff:
                stop_child(process)
                break
            try:
                process.wait(timeout=min(5, max(.05, cutoff-time.time())))
            except subprocess.TimeoutExpired:
                pass
        state.update(child_pid=None, child_exit=process.returncode, updated_at=now())
        atomic(directory / "supervisor.json", state)
        return process.returncode


def ensure_fallback(directory):
    """An early search failure still leaves the measured baseline runnable."""
    decision = directory / "output/decision.json"
    if decision.is_file():
        return
    source = directory / "frozen/tools/overnight_search.py"
    manifest = directory / "manifest.json"
    if not source.is_file() or not manifest.is_file():
        raise RuntimeError("Search failed before capturing a recoverable baseline")
    metadata = json.loads(manifest.read_text())
    expected = metadata["files"]["frozen/tools/overnight_search.py"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError("Fallback scheduler fingerprint changed")
    spec = importlib.util.spec_from_file_location("overnight_recovery", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify_frozen(directory, metadata)
    module.package(directory, metadata, "current", {
        "passed": False, "complete": False,
        "reason": "Supervisor recovered the frozen current bot after an incomplete search; no new improvement is certified.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reserve-minutes", type=float, default=50)
    parser.add_argument("--extra-opponent-bundle", type=Path, action="append", default=[])
    parser.add_argument("--max-restarts", type=int, default=2)
    parser.add_argument("--skip-certification", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.max_restarts <= 2 or not 1 <= args.workers <= 4:
        parser.error("max-restarts must be0..2 and workers1..4")
    directory = args.run_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    deadline = dt.datetime.fromisoformat(args.deadline.replace("Z", "+00:00")).timestamp()
    if deadline <= time.time():
        parser.error("The fixed search deadline must be in the future")
    lock = (directory / "supervisor.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("A supervisor already owns this run")
    state_path = directory / "supervisor.json"
    previous = json.loads(state_path.read_text()) if state_path.is_file() else None
    if previous and previous["deadline"] != args.deadline:
        parser.error("A restart cannot renew the original deadline")
    if scheduler_active(directory, previous):
        parser.error("An existing scheduler/certifier still owns this run; no duplicate recovery was started")
    if previous and previous.get("phase") == "completed":
        print(json.dumps(previous, indent=2))
        return
    state = {"pid": os.getpid(), "started_at": previous["started_at"] if previous else now(), "updated_at": now(),
             "deadline": args.deadline, "phase": "search", "attempts": previous.get("attempts", []) if previous else [],
             "auto_promote": False, "auto_push": False}
    atomic(state_path, state)
    if not args.skip_certification:
        freeze_certification_inputs(directory)

    def stop(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    for attempt in range(len(state["attempts"]), args.max_restarts + 1):
        if STOP or time.time() >= deadline:
            break
        if scheduler_active(directory, state):
            raise RuntimeError("Another scheduler still owns this run; refusing concurrent retry")
        script = choose_scheduler(directory)
        command = [sys.executable, "-u", str(script), "--run-dir", str(directory),
                   "--deadline", args.deadline, "--workers", str(args.workers),
                   "--reserve-minutes", str(args.reserve_minutes)]
        for bundle in args.extra_opponent_bundle:
            command.extend(["--extra-opponent-bundle", str(bundle.resolve())])
        if (directory / "status.json").exists():
            command.append("--resume")
        state["attempts"].append({"attempt": attempt + 1, "exit_code": None, "started_at": now()})
        atomic(state_path, state)
        code = run_child(command, directory, "launcher.log", deadline + 15, state)
        state["attempts"][-1].update(exit_code=code, finished_at=now())
        atomic(state_path, state)
        checkpoint = json.loads((directory / "status.json").read_text()) if (directory / "status.json").is_file() else {}
        if code == 0 or checkpoint.get("terminal") or STOP:
            break
        # At most three attempts, each using the captured script and checkpoint.
        # No candidate replacement or final-seed reset is possible on retry.
    if STOP:
        state.update(phase="stopped", updated_at=now(), message="Stopped by signal; original absolute deadline retained")
        atomic(state_path, state)
        return
    try:
        if scheduler_active(directory, state):
            raise RuntimeError("A scheduler is still active; refusing concurrent fallback/certification")
        ensure_fallback(directory)
    except Exception as error:
        state.update(phase="failed", updated_at=now(), error=str(error))
        atomic(state_path, state)
        raise
    state.update(phase="certification", updated_at=now())
    atomic(state_path, state)
    if args.skip_certification:
        code = None
    else:
        freeze_certification_inputs(directory)
        command = [sys.executable, "-u", str(ROOT / "tools/certify_overnight.py"), "--run-dir", str(directory)]
        code = run_child(command, directory, "certification-launcher.log", time.time() + 15 * 60, state)
        freeze_certification_inputs(directory)
    report_path = directory / "certification.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    state.update(phase="stopped" if STOP else "completed", updated_at=now(), certification_exit=code,
                 certification_status=certification_status(code, report),
                 message="Review artifacts recorded; inspect decision and certification before deployment. Original bot and GitHub unchanged.")
    atomic(state_path, state)


if __name__ == "__main__":
    main()
