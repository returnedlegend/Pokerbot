"""Fail-closed checks before the unattended certifier executes packaged code."""
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import zipfile

import pytest

from tools.certify_overnight import CORE, LIMITER, MONITOR, clean_stats, verify_package


def package(tmp_path, *, extra=None, corrupt=None, embedded=None, symlink=False, manifest_extra=False):
    payload = {name: (name + " bytes").encode() for name in CORE}
    manifest = {"chosen": "current", "files": {
        name: hashlib.sha256(value).hexdigest() for name, value in payload.items()}}
    payload["MANIFEST.json"] = json.dumps(embedded or manifest).encode()
    payload["README.md"] = b"Ancillary review instructions"
    if extra:
        payload.update(extra)
        if manifest_extra:
            manifest["files"].update({name: hashlib.sha256(value).hexdigest() for name, value in extra.items()})
            payload["MANIFEST.json"] = json.dumps(embedded or manifest).encode()
    if corrupt:
        payload[corrupt] = b"unexpected source"
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for name, value in payload.items():
            if symlink and name == "agent.py":
                info = zipfile.ZipInfo(name)
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                handle.writestr(info, value)
            else:
                handle.writestr(name, value)
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps(dict(manifest, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
    return archive, decision


def test_exact_manifest_accepts_only_expected_runtime_and_review_metadata(tmp_path):
    archive, decision = package(tmp_path)
    metadata, payload = verify_package(archive, decision)
    assert metadata["chosen"] == "current"
    assert set(payload) == set(CORE) | {"MANIFEST.json", "README.md"}


def test_outer_archive_change_is_rejected(tmp_path):
    archive, decision = package(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="ZIP hash"):
        verify_package(archive, decision)


def test_changed_runtime_rejected_even_if_outer_archive_hash_is_updated(tmp_path):
    archive, decision = package(tmp_path, corrupt="strategy.py")
    with pytest.raises(ValueError, match="file hash mismatch"):
        verify_package(archive, decision)


@pytest.mark.parametrize("name", ["../outside.py", "/absolute.py", "folder\\outside.py"])
def test_path_escape_rejected(tmp_path, name):
    archive, decision = package(tmp_path, extra={name: b"x"})
    with pytest.raises(ValueError, match="Unsafe ZIP path"):
        verify_package(archive, decision)


def test_unmanifested_executable_rejected(tmp_path):
    archive, decision = package(tmp_path, extra={"unexpected.py": b"x"})
    with pytest.raises(ValueError, match="Unmanifested"):
        verify_package(archive, decision)


def test_manifested_alias_cannot_overwrite_a_canonical_staged_file(tmp_path):
    for alias in ("./agent.py", "nested//unexpected.py"):
        archive, decision = package(tmp_path, extra={alias: b"different source"}, manifest_extra=True)
        with pytest.raises(ValueError, match="Noncanonical"):
            verify_package(archive, decision)


def test_symlink_rejected(tmp_path):
    archive, decision = package(tmp_path, symlink=True)
    with pytest.raises(ValueError, match="symlinks"):
        verify_package(archive, decision)


def test_embedded_selection_cannot_disagree_with_external_decision(tmp_path):
    archive, decision = package(tmp_path, embedded={"chosen": "different"})
    with pytest.raises(ValueError, match="manifests differ"):
        verify_package(archive, decision)


def test_complete_bank_checks_reject_faults_and_missing_half():
    half = {"errors": 0, "skipped_calls": 0, "timed_out": False,
            "forfeited": False, "time_seconds": 5.0}
    assert clean_stats([half, dict(half)])
    assert not clean_stats([half])
    for key, value in (("errors", 1), ("skipped_calls", 1), ("timed_out", True),
                       ("forfeited", True), ("time_seconds", 8.01)):
        assert not clean_stats([half, dict(half, **{key: value})])


def test_os_launcher_applies_real_affinity_and_address_space_limits():
    cpu = min(os.sched_getaffinity(0))
    environment = dict(os.environ, POKER_CERT_CPU=str(cpu))
    result = subprocess.run([sys.executable, "-c", LIMITER, "--facts"],
                            env=environment, capture_output=True, text=True, check=True)
    actual = json.loads(result.stdout)
    assert actual["affinity"] == [cpu]
    assert actual["rlimit_as"] == [4 * 1024**3] * 2
    assert actual["cpus_allowed_list"] == str(cpu)


def test_runtime_monitor_source_compiles():
    compile(MONITOR, "certification-monitor.py", "exec")
