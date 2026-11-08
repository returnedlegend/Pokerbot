"""Owned-child shutdown and truthful offline-supervisor completion guards."""
import json
from pathlib import Path
import signal
import subprocess

import pytest

from tools import overnight_supervisor as supervisor


class FakeProcess:
    pid = 918273

    def __init__(self, returncode=None, timeouts=0):
        self.returncode = returncode
        self.timeouts = timeouts
        self.waits = []

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.timeouts:
            self.timeouts -= 1
            raise subprocess.TimeoutExpired("owned-fake-child", timeout)
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_completed_child_never_receives_group_signal(monkeypatch):
    process = FakeProcess(returncode=0)
    monkeypatch.setattr(supervisor.os, "killpg", lambda *_: pytest.fail("Completed child must not be signaled"))
    supervisor.stop_child(process)
    assert process.waits == []


def test_stop_signals_only_exact_owned_child_group(monkeypatch):
    process, signals = FakeProcess(), []
    monkeypatch.setattr(supervisor.os, "killpg", lambda pgid, signum: signals.append((pgid, signum)))
    supervisor.stop_child(process)
    assert signals == [(process.pid, signal.SIGTERM)]
    assert process.waits == [20]


def test_unresponsive_child_group_is_force_stopped_with_bounded_waits(monkeypatch):
    process, signals = FakeProcess(timeouts=1), []

    def send(pgid, signum):
        signals.append((pgid, signum))
        if signum == signal.SIGKILL:
            process.returncode = -signal.SIGKILL

    monkeypatch.setattr(supervisor.os, "killpg", send)
    supervisor.stop_child(process)
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]
    assert process.waits == [20, 10]


def test_already_gone_child_group_is_harmless(monkeypatch):
    process = FakeProcess()

    def gone(*_):
        raise ProcessLookupError()

    monkeypatch.setattr(supervisor.os, "killpg", gone)
    supervisor.stop_child(process)
    assert process.waits == []


def test_child_has_private_process_group_and_bounded_numerical_threads(tmp_path, monkeypatch):
    captured, process, state = {}, FakeProcess(returncode=0), {}

    def launch(command, **kwargs):
        captured.update(command=command, **kwargs)
        return process

    monkeypatch.setattr(supervisor.subprocess, "Popen", launch)
    monkeypatch.setattr(supervisor, "STOP", False)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "99")
    assert supervisor.run_child(["owned-python", "captured-scheduler"], tmp_path, "child.log", 1e20, state) == 0
    assert captured["start_new_session"] is True
    assert captured["stdin"] == subprocess.DEVNULL
    assert captured["stderr"] == subprocess.STDOUT
    assert all(captured["env"][key] == "1" for key in
               ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"))
    recorded = json.loads((tmp_path/"supervisor.json").read_text())
    assert recorded["child_pid"] is None and recorded["child_exit"] == 0


def test_cutoff_stops_owned_child_and_clears_pid(tmp_path, monkeypatch):
    process, stopped = FakeProcess(), []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_, **__: process)
    monkeypatch.setattr(supervisor, "STOP", False)
    monkeypatch.setattr(supervisor.time, "time", lambda: 200.)

    def stop(child):
        stopped.append(child)
        child.returncode = -signal.SIGTERM

    monkeypatch.setattr(supervisor, "stop_child", stop)
    state = {}
    assert supervisor.run_child(["owned"], tmp_path, "child.log", 199., state) == -signal.SIGTERM
    assert stopped == [process]
    assert state["child_pid"] is None


def test_existing_decision_is_never_overwritten_by_fallback(tmp_path, monkeypatch):
    (tmp_path/"output").mkdir()
    decision = tmp_path/"output/decision.json"
    original = b'{"chosen":"locked-candidate","final_gate_passed":true}\n'
    decision.write_bytes(original)
    monkeypatch.setattr(supervisor.importlib.util, "spec_from_file_location",
                        lambda *_: pytest.fail("Existing decision must bypass recovery import"))
    supervisor.ensure_fallback(tmp_path)
    assert decision.read_bytes() == original


def test_drifted_recovery_source_is_rejected_before_execution(tmp_path):
    source = tmp_path/"frozen/tools/overnight_search.py"
    source.parent.mkdir(parents=True)
    source.write_text("raise AssertionError('drifted source must not execute')\n")
    supervisor.atomic(tmp_path/"manifest.json", {"files": {"frozen/tools/overnight_search.py": "0"*64}})
    with pytest.raises(RuntimeError, match="changed|hash|mismatch|provenance"):
        supervisor.ensure_fallback(tmp_path)


@pytest.fixture
def scheduler_paths(tmp_path, monkeypatch):
    repo, run = tmp_path/"repository", tmp_path/"private-run"
    original = repo/"tools/overnight_search.py"
    original.parent.mkdir(parents=True)
    original.write_text("# fixed original scheduler\n")
    captured = run/"frozen/tools/overnight_search.py"
    captured.parent.mkdir(parents=True)
    monkeypatch.setattr(supervisor, "ROOT", repo)
    return original, captured, run


def test_partial_initialization_retries_original_equal_source(scheduler_paths):
    original, captured, run = scheduler_paths
    captured.write_bytes(original.read_bytes())
    assert supervisor.choose_scheduler(run) == original


def test_partial_initialization_cannot_retry_changed_original(scheduler_paths):
    original, captured, run = scheduler_paths
    captured.write_bytes(original.read_bytes())
    original.write_text("# changed original scheduler\n")
    with pytest.raises(RuntimeError, match="changed|differ|captur"):
        supervisor.choose_scheduler(run)


def test_initialized_retry_uses_captured_scheduler(scheduler_paths):
    original, captured, run = scheduler_paths
    captured.write_bytes(original.read_bytes())
    digest = supervisor.hashlib.sha256(captured.read_bytes()).hexdigest()
    supervisor.atomic(run/"manifest.json", {"files": {"frozen/tools/overnight_search.py": digest}})
    supervisor.atomic(run/"status.json", {"phase": "development"})
    assert supervisor.choose_scheduler(run) == captured


def test_existing_scheduler_lock_blocks_duplicate_recovery(tmp_path):
    with (tmp_path/"run.lock").open("a+") as owned:
        supervisor.fcntl.flock(owned, supervisor.fcntl.LOCK_EX | supervisor.fcntl.LOCK_NB)
        assert supervisor.scheduler_active(tmp_path)
    assert not supervisor.scheduler_active(tmp_path)


def test_unrelated_live_pid_is_not_treated_as_owned_child(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor.os, "killpg", lambda *_: pytest.fail("Ownership probing must never signal another group"))
    pid = supervisor.os.getpid()
    assert not supervisor.scheduler_active(tmp_path, {"child_pid": pid,
                                                     "child_start_ticks": supervisor.process_start_ticks(pid)})


@pytest.mark.parametrize("expected_ticks,active", [("original-start", True), ("reused-pid-start", False)])
def test_owned_child_before_lock_is_identified_without_pid_reuse(tmp_path, monkeypatch, expected_ticks, active):
    previous = {"child_pid": 424242, "child_start_ticks": expected_ticks}
    monkeypatch.setattr(supervisor, "process_start_ticks", lambda _: "original-start")
    original_read = Path.read_bytes
    command = "\0".join(["python", str(supervisor.ROOT/"tools/overnight_search.py"), "--run-dir", str(tmp_path), ""])

    def read(path):
        return command.encode() if str(path) == "/proc/424242/cmdline" else original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    assert supervisor.scheduler_active(tmp_path, previous) is active


@pytest.mark.parametrize("changed_copy", ["original", "captured"])
def test_certifier_inputs_cannot_drift_during_search(tmp_path, monkeypatch, changed_copy):
    repo, run = tmp_path/"repo", tmp_path/"run"
    source = repo/"tools/certify_overnight.py"
    source.parent.mkdir(parents=True)
    source.write_text("# fixed certifier\n")
    run.mkdir()
    monkeypatch.setattr(supervisor, "ROOT", repo)
    monkeypatch.setattr(supervisor, "CERTIFICATION_FILES", ("tools/certify_overnight.py",))
    metadata = supervisor.freeze_certification_inputs(run)
    assert "tools/certify_overnight.py" in metadata["files_sha256"]
    changed = source if changed_copy == "original" else run/"control/tools/certify_overnight.py"
    changed.write_text("# changed certifier\n")
    with pytest.raises(RuntimeError, match="Certification input changed"):
        supervisor.freeze_certification_inputs(run)


@pytest.mark.parametrize("retries", [-1, 3])
def test_restart_budget_cannot_exceed_two_retries(tmp_path, monkeypatch, retries):
    monkeypatch.setattr(supervisor.sys, "argv", ["overnight_supervisor", "--run-dir", str(tmp_path),
                        "--deadline", "2099-11-08T09:40:08Z", "--max-restarts", str(retries), "--skip-certification"])
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_: pytest.fail("Invalid budget must not launch children"))
    with pytest.raises(SystemExit) as error:
        supervisor.main()
    assert error.value.code == 2


@pytest.mark.parametrize("code,report,expected", [
    (None, None, "skipped"),
    (0, None, "failed"),
    (0, {"status": "smoke_only_passed", "runtime_verified": False, "phase": "complete"}, "failed"),
    (0, {"status": "certified", "runtime_verified": True, "phase": "complete", "validation_checks": {"full_match": True}}, "passed"),
    (1, {"status": "certified", "runtime_verified": True, "phase": "complete", "validation_checks": {"full_match": True}}, "failed"),
    (-signal.SIGTERM, {"status": "running", "runtime_verified": False}, "failed"),
])
def test_certification_status_requires_successful_full_report(code, report, expected):
    assert supervisor.certification_status(code, report) == expected
