"""Guard the offline search's selection boundary and private input handling."""
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from tools import overnight_search as search


@pytest.fixture
def protected_run(tmp_path):
    snapshots = tmp_path/"snapshots"
    snapshots.mkdir()
    for name in ("current", "candidate-a", "candidate-b"):
        search.atomic(snapshots/(name+".json"), {"parameters": {"name": name}})
    seeds = ["review-protected-a", "review-protected-b", "review-protected-c"]
    raw = search.canonical(seeds)
    (tmp_path/"protected-final-seeds.json").write_bytes(raw)
    manifest = {"config": {"final_seed_commitment": search.sha(raw)}}
    state = {"deadline_epoch": time.time()+3600, "manifest_sha256": "review-manifest",
             "phase": "development", "archive": [{"candidate": "candidate-a", "feasible": True, "score": 1.}],
             "latest_long_ranking": [{"candidate": "candidate-a", "feasible": True, "score": 1.}]}
    return tmp_path, state, manifest, seeds


def test_failed_final_candidate_cannot_be_replaced_by_runner_up(protected_run):
    directory, state, manifest, _ = protected_run
    first = search.lock_selection(directory, state, manifest)
    assert first["selected"] == "candidate-a"
    search.atomic(directory/"final-gate.json", {"passed": False, "complete": True})
    state["archive"] = [{"candidate": "candidate-b", "feasible": True, "score": 100.}]
    state["latest_long_ranking"] = state["archive"]
    # A repeated selection request retains the immutable first finalist even
    # when a failed final test and a more attractive runner-up are present.
    assert search.lock_selection(directory, state, manifest) == first


def test_selection_uses_latest_complete_comparison_instead_of_historical_score(protected_run):
    directory, state, manifest, _ = protected_run
    state["archive"] = [{"candidate": "candidate-a", "feasible": True, "score": 100.}]
    state["latest_long_ranking"] = [{"candidate": "candidate-b", "feasible": True, "score": .1}]
    assert search.lock_selection(directory, state, manifest)["selected"] == "candidate-b"


def test_nonpositive_development_score_preserves_current(protected_run):
    directory, state, manifest, _ = protected_run
    state["latest_long_ranking"][0]["score"] = 0.
    assert search.lock_selection(directory, state, manifest)["selected"] == "current"


def test_protected_seed_access_requires_lock_and_matching_commitment(protected_run):
    directory, state, manifest, seeds = protected_run
    with pytest.raises(RuntimeError, match="before selection lock"):
        search.open_final_seeds(directory, state, manifest)
    search.lock_selection(directory, state, manifest)
    assert search.open_final_seeds(directory, state, manifest) == seeds
    (directory/"protected-final-seeds.json").write_text('["changed"]')
    with pytest.raises(RuntimeError, match="commitment mismatch"):
        search.open_final_seeds(directory, state, manifest)


def test_locked_snapshot_cannot_change_before_resume(protected_run):
    directory, state, manifest, _ = protected_run
    search.lock_selection(directory, state, manifest)
    search.atomic(directory/"snapshots/candidate-a.json", {"parameters": {"name": "changed"}})
    with pytest.raises(RuntimeError, match="Locked candidate snapshot changed"):
        search.lock_selection(directory, state, manifest)


def test_packaging_rechecks_exact_locked_snapshot(protected_run):
    directory, state, manifest, _ = protected_run
    manifest["files"] = {"snapshots/current.json": search.sha((directory/"snapshots/current.json").read_bytes())}
    search.lock_selection(directory, state, manifest)
    chosen, _ = search.verify_chosen_snapshot(directory, manifest, "candidate-a", {"passed": True})
    assert chosen == "candidate-a"
    search.atomic(directory/"snapshots/candidate-a.json", {"parameters": {"name": "unverified"}})
    with pytest.raises(RuntimeError, match="differs from locked/tested"):
        search.verify_chosen_snapshot(directory, manifest, "candidate-a", {"passed": True})


def test_failed_final_packages_only_unchanged_current_snapshot(protected_run):
    directory, _, manifest, _ = protected_run
    current = directory/"snapshots/current.json"
    manifest["files"] = {"snapshots/current.json": search.sha(current.read_bytes())}
    chosen, _ = search.verify_chosen_snapshot(directory, manifest, "candidate-a", {"passed": False})
    assert chosen == "current"
    search.atomic(current, {"parameters": {"name": "changed-current"}})
    with pytest.raises(RuntimeError, match="Frozen.*changed"):
        search.verify_chosen_snapshot(directory, manifest, "candidate-a", {"passed": False})


def test_resume_rejects_manifest_replacement(tmp_path):
    manifest_path = tmp_path/"manifest.json"
    search.atomic(manifest_path, {"files": {}, "config": {}})
    original_hash = search.sha(manifest_path.read_bytes())
    search.atomic(tmp_path/"status.json", {"manifest_sha256": original_hash, "deadline_epoch": time.time()+3600})
    search.atomic(manifest_path, {"files": {}, "config": {"changed": True}})
    args = SimpleNamespace(run_dir=tmp_path, resume=True, deadline=None, extra_opponent_bundle=[])
    with pytest.raises(RuntimeError, match="manifest changed"):
        search.initialize(args)


def test_resume_cannot_renew_deadline(tmp_path):
    deadline = time.time()+3600
    manifest_path = tmp_path/"manifest.json"
    search.atomic(manifest_path, {"files": {}, "config": {}})
    search.atomic(tmp_path/"status.json", {"manifest_sha256": search.sha(manifest_path.read_bytes()),
                                           "deadline_epoch": deadline})
    args = SimpleNamespace(run_dir=tmp_path, resume=True, deadline=search.utc(deadline+3600), extra_opponent_bundle=[])
    with pytest.raises(ValueError, match="original absolute deadline"):
        search.initialize(args)


@pytest.mark.parametrize("changed", ["frozen/tools/overnight_search.py", "frozen/tools/evaluate_strength.py"])
def test_scheduler_or_evaluator_drift_is_rejected(tmp_path, changed):
    expected = {"frozen/tools/overnight_search.py": search.sha(Path(search.__file__).read_bytes()),
                "frozen/tools/evaluate_strength.py": search.sha((search.ROOT/"tools/evaluate_strength.py").read_bytes())}
    search.verify_scheduler(tmp_path, {"files": expected})
    expected[changed] = "0"*64
    with pytest.raises(RuntimeError, match="Scheduler/evaluator changed"):
        search.verify_scheduler(tmp_path, {"files": expected})


def test_private_bundle_paths_cannot_live_inside_repository(tmp_path):
    with pytest.raises(ValueError, match="outside the Git repository"):
        search.require_private_path(search.ROOT/"results/private-input", [tmp_path/"bundle"])
    search.require_private_path(tmp_path/"private-run", [tmp_path/"bundle"])
    search.require_private_path(search.ROOT/"results/synthetic-only", [])


def test_candidate_snapshot_is_checked_before_any_cached_case_is_used(tmp_path, monkeypatch):
    snapshots = tmp_path/"snapshots"
    snapshots.mkdir()
    path = snapshots/"candidate-a.json"
    path.write_text('{"version": 1}')
    expected = search.sha(path.read_bytes())
    path.write_text('{"version": 2}')
    monkeypatch.setattr(search, "WORKER", {"directory": tmp_path, "manifest": {"files": {}}})
    with pytest.raises(RuntimeError, match="Candidate snapshot changed"):
        search.worker_case({"candidate": "candidate-a", "candidate_snapshot_sha256": expected})


def gate_rows(tmp_path, *, external_delta=1., candidate_pot=20, current_pot=20,
              candidate_half=5, current_half=5, released_gain=100000.):
    """Construct complete observations of three independent seed trajectories."""
    profiles, seeds, deals = ("field", "current", "released978"), ("seed-a", "seed-b", "seed-c"), 20
    paths = []

    def match(profile, seed, mean, pot, half):
        stats = {"seconds": .02, "errors": 0,
                 "actions": {"RAISE_POT": pot, "RAISE_HALF": half}}
        return {"opponent": profile, "seed": seed, "deals": deals,
                "paired_bb": [mean]*deals, "half_stats": [[stats, {}], [stats, {}]]}

    for profile in profiles:
        for seed in seeds:
            raw, delta = ((2.+external_delta, external_delta) if profile == "field"
                          else (.5, .5) if profile == "current" else (released_gain, released_gain))
            # Supplemental978 has intentionally extreme action rates. It must
            # not dilute the controlled-aggression gate against current.
            cp, op = (0, 1000) if profile == "released978" else (candidate_pot, current_pot)
            candidate = match(profile, seed, raw, cp, candidate_half)
            control = match(profile, seed, raw-delta, op, current_half)
            difference = {"opponent": profile, "seed": seed, "paired_bb": [delta]*deals}
            row = {"task": {"candidate": "candidate-a", "opponent": profile, "seed": seed, "deals": deals},
                   "candidate": candidate, "control": control, "difference": difference}
            path = tmp_path/(profile+"-"+seed+".json")
            search.atomic(path, row)
            paths.append(str(path))
    return paths, profiles, seeds, deals


def test_old978_winnings_cannot_qualify_failed_external_gain(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "SCHEDULER", {})
    paths, profiles, seeds, deals = gate_rows(tmp_path, external_delta=0.)
    gate = search.final_gate(paths, "candidate-a", profiles, seeds, deals)
    assert gate["complete"] and not gate["passed"]
    assert gate["field"]["mean"] == 0.
    assert gate["h2h_current"]["mean"] == .5
    assert not gate["checks"]["field_card_lower_positive"]
    assert "released978" not in gate["summary"]["per_profile_delta"]
    assert gate["summary"]["pot_ratio"] == 1.


@pytest.mark.parametrize("candidate_pot,expected_ratio,passes", [(21, 1.05, True), (22, 1.10, False)])
def test_exact_pot_frequency_boundary_excludes_supplemental978(tmp_path, monkeypatch, candidate_pot, expected_ratio, passes):
    monkeypatch.setattr(search, "SCHEDULER", {})
    paths, profiles, seeds, deals = gate_rows(tmp_path, candidate_pot=candidate_pot)
    gate = search.final_gate(paths, "candidate-a", profiles, seeds, deals)
    assert gate["summary"]["pot_ratio"] == pytest.approx(expected_ratio)
    assert gate["checks"]["pot_frequency_at_most_1_05_current"] is passes
    assert gate["passed"] is passes


def test_total_half_raise_escalation_fails_even_with_unchanged_pot_frequency(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "SCHEDULER", {})
    paths, profiles, seeds, deals = gate_rows(tmp_path, candidate_half=30)
    gate = search.final_gate(paths, "candidate-a", profiles, seeds, deals)
    assert gate["checks"]["pot_frequency_at_most_1_05_current"]
    assert not gate["checks"]["total_raise_frequency_controlled"]
    assert not gate["passed"]


def test_slow_high_ev_candidate_is_infeasible_in_development_and_final(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "SCHEDULER", {})
    paths, profiles, seeds, deals = gate_rows(tmp_path)
    for path in paths:
        row = search.read(path)
        if row["task"]["opponent"] == "field":
            for half in row["candidate"]["half_stats"]:
                half[0]["seconds"] = 6.51*deals/1000
            search.atomic(path, row)
    summary = search.metrics(paths, "candidate-a")
    assert summary["pooled"]["mean_bb_per_deal"] > 0
    assert summary["candidate_rates"]["max_direct_seconds_per_1000"] == pytest.approx(6.51)
    assert not summary["feasible"]
    gate = search.final_gate(paths, "candidate-a", profiles, seeds, deals)
    assert not gate["checks"]["direct_runtime_proxy_within_6_5_seconds_per_1000"]
    assert not gate["passed"]


def test_incomplete_final_suite_preserves_current(tmp_path, monkeypatch):
    monkeypatch.setattr(search, "SCHEDULER", {})
    paths, profiles, seeds, deals = gate_rows(tmp_path)
    gate = search.final_gate(paths[:-1], "candidate-a", profiles, seeds, deals)
    assert not gate["complete"] and not gate["passed"]
    assert "preserve current" in gate["reason"]
