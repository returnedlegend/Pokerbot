#!/usr/bin/env python3
"""Bounded, resumable offline policy search; never edits the deployed bot.

Example:
  .venv/bin/python tools/overnight_search.py --run-dir /workspace/poker-overnight-runs/01 \
    --deadline 2026-11-08T09:40:08Z --workers 3

Resume with the same command and --resume.  The original absolute deadline,
frozen code, baselines and final-seed commitment are retained.  --quick runs
a small end-to-end test, including the irrevocable final-selection lock.
No language-model API or network calls are made by this tool.
"""
from __future__ import annotations

import os
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
import base64
import concurrent.futures as cf
import datetime as dt
import fcntl
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import random
import secrets
import shutil
import signal
import statistics
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASED = "97883009569ad2d383b8145e46ee642262390049"
RUNTIME_FILES = ("agent.py", "strategy.py", "opponent_model.py", "policy.json", "game.py",
                 "evaluator.py", "equity_engine.py", "preflop_model.npz", "equity_network.py",
                 "equity_network.npz", "river_solver.py", "river_policy.npz", "requirements.txt",
                 "THIRD_PARTY_NOTICES.md", ".dockerignore", "docker/Dockerfile.base", "docker/Dockerfile.agent")
TOOL_FILES = ("tools/overnight_search.py", "tools/evaluate_strength.py", "tools/benchmark.py",
              "tools/style_opponents.py", "tools/strength_opponents.py", "tools/fast_range_opponent.py",
              "tools/slow_expert_opponent.py")
STOP = False
WORKER = {}
SCHEDULER = {}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f".{os.getpid()}.tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def utc(timestamp=None):
    return dt.datetime.fromtimestamp(timestamp or time.time(), dt.timezone.utc).isoformat()


def timestamp(value):
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def log(directory, message):
    line = f"{utc()} {message}"
    print(line, flush=True)
    with (Path(directory)/"progress.log").open("a") as handle:
        handle.write(line+"\n")


def status(directory, state, **extra):
    state.update(extra)
    state["updated_at"] = utc()
    state["seconds_remaining"] = max(0., state["deadline_epoch"]-time.time())
    atomic(Path(directory)/"status.json", state)


def verify_frozen(directory, manifest):
    for name, expected in manifest["files"].items():
        path = Path(directory)/name
        if not path.is_file() or sha(path.read_bytes()) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")


def verify_scheduler(directory, manifest):
    for source, frozen in ((Path(__file__), "frozen/tools/overnight_search.py"),
                           (ROOT/"tools/evaluate_strength.py", "frozen/tools/evaluate_strength.py")):
        if sha(source.read_bytes()) != manifest["files"][frozen]:
            raise RuntimeError("Scheduler/evaluator changed; resume with the captured frozen/tools/overnight_search.py")


def reconcile_counts(directory, state):
    cases = [read(p) for p in (Path(directory)/"cases").glob("*/*.json")]
    controls = [read(p) for p in (Path(directory)/"controls").glob("*.json")]
    state["completed_cases"] = len(cases)
    state["completed_candidate_hands"] = sum(2*c["task"]["deals"] for c in cases)
    state["newly_simulated_hands"] = state["completed_candidate_hands"]+sum(2*c["match"]["deals"] for c in controls)
    state["counts_scope"] = "Complete atomic candidate cases and control matches; interrupted partial games are not counted"


def require_private_path(directory, bundles):
    if bundles and directory.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Private third-party bundles require a run directory outside the Git repository")


def initialize(args):
    directory = args.run_dir.resolve()
    require_private_path(directory, args.extra_opponent_bundle)
    if (directory/"status.json").exists():
        if not args.resume:
            raise ValueError("Existing run needs --resume; deadlines cannot be renewed")
        state, manifest = read(directory/"status.json"), read(directory/"manifest.json")
        if sha((directory/"manifest.json").read_bytes()) != state["manifest_sha256"]:
            raise RuntimeError("Run manifest changed since checkpoint")
        if args.deadline and abs(timestamp(args.deadline)-state["deadline_epoch"]) > 1:
            raise ValueError("Resume must preserve the original absolute deadline")
        verify_frozen(directory, manifest)
        verify_scheduler(directory, manifest)
        reconcile_counts(directory, state)
        return directory, state, manifest
    if args.resume:
        raise ValueError("No checkpoint exists to resume")
    directory.mkdir(parents=True, exist_ok=True)
    code = directory/"frozen"
    for name in (*RUNTIME_FILES, *TOOL_FILES):
        source = ROOT/name
        if not source.is_file():
            raise FileNotFoundError(source)
        target = code/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    sys.path.insert(0, str(ROOT/"tools"))
    from evaluate_strength import factory_from_candidate, factory_from_commit
    current = factory_from_candidate(ROOT/"strategy.py")
    released = factory_from_commit(RELEASED)
    for name, factory in (("current", current), ("released978", released)):
        factory.file = code/"strategy.py"
    (directory/"snapshots").mkdir(exist_ok=True)
    current.save_snapshot(directory/"snapshots/current.json")
    released.save_snapshot(directory/"snapshots/released978.json")
    profiles = ["tight_passive", "loose_passive", "selective_aggressive", "loose_aggressive",
                "changing", "changing_reverse", "fast_range", "draw_range", "adaptive_pressure",
                "current", "released978"]
    opponents = {name: {"kind": "snapshot", "snapshot": name} if name in ("current", "released978")
                 else {"kind": "builtin", "name": name} for name in profiles}
    challenge = ROOT/"tools/challenge_opponents.py"
    if challenge.is_file():
        # The complete implementation, rather than a mutable transitive import,
        # is embedded separately for each literal profile.
        import ast
        source = challenge.read_text()
        tree = ast.parse(source)
        names = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PROFILES" for t in node.targets):
                names = ast.literal_eval(node.value)
        if not names:
            raise ValueError("Challenge source must contain literal PROFILES")
        for name in names:
            path = directory/"probes"/(name+".py")
            path.parent.mkdir(exist_ok=True)
            path.write_text(source+f"\n_instance = ChallengeOpponent({name!r})\nnew_half = _instance.new_half\nchoose_action = _instance.choose_action\non_hand_end = _instance.on_hand_end\n")
            profiles.append(name)
            opponents[name] = {"kind": "external", "path": str(path.relative_to(directory))}
    elif not args.quick:
        raise FileNotFoundError("Freeze the audited challenge suite before starting a real run")
    for bundle in args.extra_opponent_bundle:
        bundle = bundle.resolve()
        if not (bundle/"opponent.py").is_file():
            raise ValueError(f"Public bundle needs opponent.py: {bundle}")
        name = "public_"+bundle.name
        if name in opponents:
            raise ValueError(f"Duplicate opponent name: {name}")
        target = directory/"probes/public"/bundle.name
        shutil.copytree(bundle, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".venv"))
        profiles.append(name)
        opponents[name] = {"kind": "external", "path": str((target/"opponent.py").relative_to(directory))}
    if args.quick:
        profiles = ["loose_aggressive", "current"]+[p for p in profiles if p.startswith("public_")]
    else:
        profiles.append("slow_expert")
        opponents["slow_expert"] = {"kind": "external", "path": "frozen/tools/slow_expert_opponent.py"}
    final_seeds = ["overnight-final-"+secrets.token_hex(12) for _ in range(3 if args.quick else args.final_seeds)]
    sealed = canonical(final_seeds)
    final_file = directory/"protected-final-seeds.json"
    final_file.write_bytes(sealed)
    final_file.chmod(0o400)
    now = time.time()
    deadline = timestamp(args.deadline) if args.deadline else now+(args.hours*3600 if not args.quick else 180)
    if deadline <= now:
        raise ValueError("Deadline has already passed")
    config = {"workers": min(4, args.workers), "quick": args.quick,
              "screen_count": 2 if args.quick else args.screen_count,
              "max_rounds": 1 if args.quick else args.max_rounds,
              "screen_deals": 6 if args.quick else 128,
              "confirm_deals": 8 if args.quick else 256,
              "long_deals": 10 if args.quick else 750,
              "final_deals": 12 if args.quick else args.final_deals,
              "reserve_seconds": 40 if args.quick else args.reserve_minutes*60,
              "profiles": profiles,
              "development_profiles": [p for p in profiles if p not in ("challenge_checkraise", "challenge_shifting", "slow_expert")],
              "final_profiles": profiles,
              "opponents": opponents,
              "dev_namespace": "overnight-dev-"+secrets.token_hex(6),
              "final_seed_commitment": sha(sealed)}
    files = {str(p.relative_to(directory)): sha(p.read_bytes())
             for parent in (code, directory/"probes", directory/"snapshots") if parent.exists()
             for p in sorted(parent.rglob("*")) if p.is_file()}
    manifest = {"version": 1, "created_at": utc(), "files": files, "config": config,
                "current_baseline": current.manifest(), "released_baseline": released.manifest(),
                "current_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "python": sys.version, "deadline_epoch": deadline,
                "scope": "Offline parameter fitting and synthetic/pinned-public opponents; no Astra or LLM calls, no superiority or tournament guarantees."}
    atomic(directory/"manifest.json", manifest)
    state = {"version": 1, "run_dir": str(directory), "started_at": utc(now), "deadline_at": utc(deadline),
             "deadline_epoch": deadline, "manifest_sha256": sha((directory/"manifest.json").read_bytes()),
             "phase": "development", "round": 0, "completed_cases": 0, "completed_candidate_hands": 0,
             "newly_simulated_hands": 0, "selected_current": "current", "candidates": {},
             "archive": [], "history": [], "final_seed_commitment": config["final_seed_commitment"],
             "message": "Frozen current neural bot is primary control; released978 is supplemental only."}
    status(directory, state)
    log(directory, f"Frozen run at {manifest['current_git_commit']}; deadline {utc(deadline)}; workers={config['workers']}; profiles={len(profiles)}")
    return directory, state, manifest


def worker_init(directory):
    global WORKER
    directory = Path(directory)
    manifest = read(directory/"manifest.json")
    verify_frozen(directory, manifest)
    code = directory/"frozen"
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(code/"tools"))
    import evaluate_strength as ev
    # A spawned worker starts clean; imports must resolve to the captured code.
    if Path(ev.__file__).resolve() != (code/"tools/evaluate_strength.py").resolve():
        raise RuntimeError("Worker imported a mutable evaluator")
    WORKER = {"directory": directory, "manifest": manifest, "ev": ev, "factories": {}, "networks": {}, "rivers": {}}


def worker_factory(name):
    cache = WORKER["factories"]
    snapshot_path = WORKER["directory"]/"snapshots"/(name+".json")
    loaded_hash = sha(snapshot_path.read_bytes())
    if name in cache and cache[name]._snapshot_sha256 != loaded_hash:
        raise RuntimeError("Snapshot changed after its factory was cached")
    if name not in cache:
        f = WORKER["ev"].factory_from_snapshot(snapshot_path)
        f._snapshot_sha256 = loaded_hash
        f.name = name
        # Neural functions and weights are stateless and identical across these
        # parameter-only candidates. Share only that immutable module, never
        # strategy memory or opponent-model namespaces.
        for attribute, store, creator in (("network", "networks", "_frozen_network"), ("river", "rivers", "_frozen_river")):
            value = getattr(f, attribute)
            if value:
                key = sha(value[0]+value[1])
                if key not in WORKER[store]:
                    WORKER[store][key] = getattr(f, creator)()
                setattr(f, attribute+"_module", WORKER[store][key])
        cache[name] = f
    return cache[name]


def worker_match(candidate, opponent, seed, deals):
    config = WORKER["manifest"]["config"]
    info = config["opponents"].get(opponent)
    factory = worker_factory(candidate)
    external = None
    control = worker_factory("current")
    if opponent.startswith("candidate:"):
        control = worker_factory(opponent.split(":", 1)[1])
        name = "incumbent"
    elif info["kind"] == "snapshot":
        control = worker_factory(info["snapshot"])
        name = "incumbent"
    elif info["kind"] == "external":
        path = WORKER["directory"]/info["path"]
        name = str(path)
        external = {name: (str(path), path.read_bytes())}
    else:
        name = info["name"]
    value = WORKER["ev"].run_match(factory, name, control, seed, deals, external)
    value["opponent"] = opponent
    value["strategy"] = candidate
    return value


def worker_case(task):
    directory = WORKER["directory"]
    verify_frozen(directory, WORKER["manifest"])
    snapshot = directory/"snapshots"/(task["candidate"]+".json")
    if sha(snapshot.read_bytes()) != task["candidate_snapshot_sha256"]:
        raise RuntimeError("Candidate snapshot changed")
    if task.get("opponent_snapshot_sha256"):
        opponent_snapshot = directory/"snapshots"/(task["opponent"].split(":", 1)[1]+".json")
        if sha(opponent_snapshot.read_bytes()) != task["opponent_snapshot_sha256"]:
            raise RuntimeError("Round-robin opponent snapshot changed")
    fingerprint = sha(canonical(task))
    output = directory/"cases"/task["candidate"]/(fingerprint+".json")
    if output.exists():
        result = read(output)
        if result["task"] != task or result["task_sha256"] != fingerprint:
            raise RuntimeError("Case cache provenance mismatch")
        return {"path": str(output), "cached": True, "new_hands": 0}
    control_key = {key: task[key] for key in ("opponent", "seed", "deals", "manifest_sha256")}
    if task.get("opponent_snapshot_sha256"):
        control_key["opponent_snapshot_sha256"] = task["opponent_snapshot_sha256"]
    control_id = sha(canonical(control_key))
    control_path = directory/"controls"/(control_id+".json")
    control_path.parent.mkdir(exist_ok=True)
    new_hands = 0
    with control_path.with_suffix(".lock").open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if control_path.exists():
            old = read(control_path)
            if old["key"] != control_key:
                raise RuntimeError("Control cache provenance mismatch")
            control = old["match"]
        else:
            control = worker_match("current", task["opponent"], task["seed"], task["deals"])
            atomic(control_path, {"key": control_key, "match": control})
            new_hands += 2*task["deals"]
    try:
        candidate = worker_match(task["candidate"], task["opponent"], task["seed"], task["deals"])
    except Exception as error:
        raise RuntimeError(f"CANDIDATE_EVALUATION: {type(error).__name__}: {error}") from error
    differences = [a-b for a, b in zip(candidate["paired_bb"], control["paired_bb"])]
    difference = {"opponent": task["opponent"], "seed": task["seed"], "paired_bb": differences,
                  **WORKER["ev"].summarize(differences)}
    verify_frozen(directory, WORKER["manifest"])
    if sha(snapshot.read_bytes()) != task["candidate_snapshot_sha256"]:
        raise RuntimeError("Candidate snapshot changed during case")
    result = {"task": task, "task_sha256": fingerprint, "candidate": candidate, "control": control,
              "difference": difference, "completed_at": utc()}
    atomic(output, result)
    return {"path": str(output), "cached": False, "new_hands": new_hands+2*task["deals"]}


def make_tasks(directory, state, candidates, stage, profiles, seeds, deals):
    return [{"candidate": candidate, "stage": stage, "opponent": profile, "seed": seed, "deals": deals,
             "manifest_sha256": state["manifest_sha256"],
             "candidate_snapshot_sha256": sha((directory/"snapshots"/(candidate+".json")).read_bytes()),
             "opponent_snapshot_sha256": (sha((directory/"snapshots"/(profile.split(":", 1)[1]+".json")).read_bytes())
                                           if profile.startswith("candidate:") else None)}
            for candidate in candidates for profile in profiles for seed in seeds]


def run_tasks(directory, state, manifest, pool, tasks, cutoff):
    queue, pending, results = list(tasks), {}, []
    workers = manifest["config"]["workers"]
    completed = 0
    while queue or pending:
        if STOP or time.time() >= cutoff:
            for future in pending:
                future.cancel()
            return results, False
        while queue and len(pending) < workers and time.time() < cutoff-2:
            task = queue.pop(0)
            pending[pool.submit(worker_case, task)] = task
        if not pending:
            break
        done, _ = cf.wait(pending, timeout=min(2., max(.01, cutoff-time.time())), return_when=cf.FIRST_COMPLETED)
        for future in done:
            task = pending.pop(future)
            try:
                value = future.result()
            except Exception as error:
                if state["phase"] != "development" or "CANDIDATE_EVALUATION:" not in str(error):
                    raise
                state.setdefault("failed_candidates", {})[task["candidate"]] = {"error": str(error), "task": task}
                queue = [q for q in queue if q["candidate"] != task["candidate"]]
                atomic(directory/"errors"/(sha(canonical(task))+".json"), {"task": task, "error": str(error), "quarantined": True})
                status(directory, state)
                log(directory, f"Quarantined candidate {task['candidate']}: {error}")
                if len(state["failed_candidates"]) >= 20:
                    raise RuntimeError("Twenty distinct policy evaluations failed; stop for review rather than silently omit opponents")
                continue
            results.append(value["path"])
            completed += 1
            if not value["cached"]:
                state["completed_cases"] += 1
                state["completed_candidate_hands"] += 2*task["deals"]
                state["newly_simulated_hands"] += value["new_hands"]
            status(directory, state, active_jobs=len(pending), pending_jobs=len(queue),
                   last_case={key: task[key] for key in ("candidate", "stage", "opponent", "seed", "deals")})
            if completed % 10 == 0 or not queue and not pending:
                log(directory, f"{state['phase']} round={state['round']} stage={task['stage']} batch={completed}/{len(tasks)} cases={state['completed_cases']} hands={state['newly_simulated_hands']}")
    return results, not queue and not pending


def stop_pool(pool):
    processes = list(getattr(pool, "_processes", {}).values())
    for process in processes:
        if process.is_alive():
            process.terminate()
    pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        process.join(timeout=1)
        if process.is_alive():
            process.kill()


def metrics(paths, candidate):
    if SCHEDULER:
        verify_scheduler(SCHEDULER["directory"], SCHEDULER["manifest"])
    sys.path.insert(0, str(ROOT/"tools"))
    from evaluate_strength import clustered_summary, aggregate
    rows = [read(p) for p in paths if read(p)["task"]["candidate"] == candidate]
    if not rows:
        return None
    candidates, controls, differences = ([r[key] for r in rows] for key in ("candidate", "control", "difference"))
    # Do not let released978's already-known weakness qualify a new candidate.
    primary = [m for m in differences if m["opponent"] != "released978"]
    external = [m for m in primary if m["opponent"] != "current"]
    def rates(matches):
        stats = [half[0] for m in matches for half in m["half_stats"]]
        hands = 2*sum(m["deals"] for m in matches)
        return {"pot": sum(s["actions"]["RAISE_POT"] for s in stats)/max(1, hands),
                "raise": sum(s["actions"]["RAISE_HALF"]+s["actions"]["RAISE_POT"] for s in stats)/max(1, hands),
                "max_direct_seconds_per_1000": max((s["seconds"]*1000/m["deals"] for m in matches for half in m["half_stats"] for s in half[:1]), default=0),
                "errors": sum(s["errors"] for s in stats)}
    candidate_rates = rates([m for m in candidates if m["opponent"] != "released978"])
    control_rates = rates([m for m in controls if m["opponent"] != "released978"])
    per_profile = aggregate(primary)
    means = [v["mean_bb_per_deal"] for v in per_profile.values()]
    pooled = clustered_summary(primary)
    robust = pooled["mean_bb_per_deal"]-.20*(statistics.pstdev(means) if len(means)>1 else 0)-.10*max(0., -min(means))
    pot_ratio = candidate_rates["pot"]/control_rates["pot"] if control_rates["pot"] else (1. if not candidate_rates["pot"] else math.inf)
    feasible = (candidate_rates["pot"] <= 1.05*control_rates["pot"] and candidate_rates["raise"] <= 1.20*control_rates["raise"]+.02
                and candidate_rates["max_direct_seconds_per_1000"] <= 6.5
                and not candidate_rates["errors"] and min(means) >= -.90)
    return {"candidate": candidate, "cases": len(rows), "score": robust if feasible else robust-10., "feasible": feasible,
            "pooled": pooled, "external": clustered_summary(external), "per_profile_delta": per_profile,
            "candidate_raw": aggregate(candidates), "candidate_rates": candidate_rates, "current_rates": control_rates,
            "pot_ratio": pot_ratio, "paths": paths}


GRID = {"call_margin": [0., .0125, .025, .03544137179472241, .045, .06],
        "realization": [.90, .925, .9538643499251365, .975, 1.],
        "raise_margin": [.015, .025, .035, .045, .055],
        "pot_margin": [.035, .046244728490194446, .055, .065, .075],
        "range_weight": [1., 1.5, 2., 2.5, 3.25, 4.],
        "preflop_raise": [.52, .543415645869242, .56, .58],
        "bluff_scale": [.35, .50, .65, .8], "caution": [.003, .006, .012, .02],
        "style_decay": [.975, .99, .995, .999], "adaptation": [.5, 1., 1.5],
        "fold_prior": [.28, .37036196781968345, .45], "quality_prior": [8., 12., 20.],
        "neural_anchor": [False, True], "draw_model": [False, True],
        "showdown_learning": [False, True], "sensible_checks": [False, True],
        "samples": [128, 192, 256, 384], "refine_samples": [192, 256, 384, 768],
        "anchor_error": [.015, .022, .035]}


def propose(directory, state, manifest):
    config = manifest["config"]
    base_payload = read(directory/"snapshots/current.json")
    base = base_payload["parameters"]
    rng = random.Random(config["dev_namespace"]+str(state["round"]))
    seeds = [base]
    for value in state["archive"][:4]:
        seeds.append(read(directory/"snapshots"/(value["candidate"]+".json"))["parameters"])
    selected = []
    attempts = 0
    # First cover individual sensitivities; subsequent rounds combine changes
    # around successful parents and retain occasional broad exploration.
    singles = [(key, value) for key, values in GRID.items() for value in values if value != base.get(key)]
    rng.shuffle(singles)
    while len(selected) < config["screen_count"] and attempts < 1000:
        attempts += 1
        params = dict(rng.choice(seeds) if rng.random() < .8 else base)
        if state["round"] == 0 and attempts <= len(singles):
            key, value = singles[attempts-1]
            params[key] = value
        else:
            for key in rng.sample(list(GRID), rng.choice([1, 2, 2, 3, 4])):
                params[key] = rng.choice(GRID[key])
        params["refine_samples"] = max(params["samples"], params["refine_samples"])
        params["future_cap"] = True
        params["river_mix"] = 0.
        params["river_range"] = False
        identity = "c"+sha(canonical(params))[:16]
        if identity in state["candidates"] or params == base:
            continue
        payload = dict(base_payload, parameters=params)
        atomic(directory/"snapshots"/(identity+".json"), payload)
        state["candidates"][identity] = {"parameters": params, "created_round": state["round"]}
        selected.append(identity)
    return selected


def interval(matches, alpha=.025):
    """Card-cluster and whole-seed intervals, each at 1-alpha confidence."""
    import numpy as np
    from scipy.stats import norm, t
    clusters = {}
    for row in matches:
        for i, value in enumerate(row["paired_bb"]):
            clusters.setdefault((row["seed"], i), []).append(value)
    by_seed = {}
    for (seed, index), values in sorted(clusters.items()):
        by_seed.setdefault(seed, []).append(float(np.mean(values)))
    all_values = [x for values in by_seed.values() for x in values]
    mean = float(np.mean(all_values))
    se = float(np.std(all_values, ddof=1)/np.sqrt(len(all_values)))
    z = float(norm.ppf(1-alpha/2))
    seed_values = [float(np.mean(values)) for values in by_seed.values()]
    seed_se = float(np.std(seed_values, ddof=1)/np.sqrt(len(seed_values))) if len(seed_values)>1 else math.inf
    critical = float(t.ppf(1-alpha/2, len(seed_values)-1)) if len(seed_values)>1 else math.inf
    return {"mean": mean, "card_ci": [mean-z*se, mean+z*se],
            "seed_ci": [mean-critical*seed_se, mean+critical*seed_se], "alpha": alpha,
            "seeds": len(seed_values), "clusters": len(all_values)}


def final_gate(paths, candidate, profiles, seeds, deals):
    rows = [read(p) for p in paths]
    expected = {(p, s) for p in profiles for s in seeds}
    if len(rows) != len(expected) or {(r["task"]["opponent"], r["task"]["seed"]) for r in rows} != expected:
        return {"passed": False, "reason": "Final suite incomplete; preserve current", "complete": False}
    if any(r["task"]["candidate"] != candidate or r["task"]["deals"] != deals for r in rows):
        raise ValueError("Final suite candidate/deal mismatch")
    summary = metrics(paths, candidate)
    field = [r["difference"] for r in rows if r["task"]["opponent"] not in ("current", "released978")]
    head = [r["candidate"] for r in rows if r["task"]["opponent"] == "current"]
    # Exactly one pre-locked candidate is tested. Bonferroni allocates .025
    # two-sided alpha to each of the field and H2H claims (family <= .05).
    field_ci, head_ci = interval(field), interval(head)
    downsides = []
    for profile in profiles:
        if profile == "released978":
            continue
        selected = [r for r in rows if r["task"]["opponent"] == profile]
        delta = interval([r["difference"] for r in selected], .05/max(1, len(profiles)-1))
        raw = interval([r["candidate"] for r in selected], .05/max(1, len(profiles)-1))
        if delta["mean"] < -.50 or min(delta["card_ci"][1], delta["seed_ci"][1]) < -.10 or min(raw["card_ci"][1], raw["seed_ci"][1]) < 0:
            downsides.append({"profile": profile, "delta": delta, "raw": raw})
    checks = {"field_card_lower_positive": field_ci["card_ci"][0] > 0,
              "field_seed_lower_positive": field_ci["seed_ci"][0] > 0,
              "h2h_card_lower_positive": head_ci["card_ci"][0] > 0,
              "h2h_seed_lower_positive": head_ci["seed_ci"][0] > 0,
              "pot_frequency_at_most_1_05_current": summary["candidate_rates"]["pot"] <= 1.05*summary["current_rates"]["pot"],
              "total_raise_frequency_controlled": summary["candidate_rates"]["raise"] <= 1.20*summary["current_rates"]["raise"]+.02,
              "no_material_profile_downside": not downsides,
              "no_errors": summary["candidate_rates"]["errors"] == 0,
              "direct_runtime_proxy_within_6_5_seconds_per_1000": summary["candidate_rates"]["max_direct_seconds_per_1000"] <= 6.5}
    return {"passed": all(checks.values()), "complete": True, "checks": checks,
            "field": field_ci, "h2h_current": head_ci, "downsides": downsides, "summary": summary,
            "caveat": "One independent locked final candidate; many development trials are selection-biased. No LLM/tournament superiority claim. Direct timing excludes HTTP and is not official-container certification."}


def lock_selection(directory, state, manifest):
    lock = directory/"selection-lock.json"
    if lock.exists():
        result = read(lock)
        if result["manifest_sha256"] != state["manifest_sha256"]:
            raise RuntimeError("Selection-lock provenance mismatch")
        if result["selected_snapshot_sha256"] != sha((directory/"snapshots"/(result["selected"]+".json")).read_bytes()):
            raise RuntimeError("Locked candidate snapshot changed")
        return result
    # Historical best scores came from different cards and are optimistic.
    # Select only among the latest complete common-card long-stage ranking.
    candidates = [v for v in state.get("latest_long_ranking", []) if v["feasible"] and v["score"] > 0]
    selected = candidates[0]["candidate"] if candidates else "current"
    result = {"selected": selected, "locked_at": utc(), "manifest_sha256": state["manifest_sha256"],
              "selected_snapshot_sha256": sha((directory/"snapshots"/(selected+".json")).read_bytes()),
              "reason": "Best positive downside-penalized score in latest complete common-card long stage; preserve current if none feasible",
              "development_evidence": candidates[:5], "final_seed_commitment": manifest["config"]["final_seed_commitment"],
              "rule": "Irrevocable. Final failure/timeout =>current; never select or test a runner-up."}
    atomic(lock, result)  # Must reach disk before protected seeds are opened.
    status(directory, state, phase="final", selected_current=selected, selection_locked_at=result["locked_at"])
    log(directory, f"Irrevocably selected {selected}; protected seed commitment is now permitted to open")
    return result


def open_final_seeds(directory, state, manifest):
    if not (directory/"selection-lock.json").exists():
        raise RuntimeError("Protected final seeds cannot be opened before selection lock")
    raw = (directory/"protected-final-seeds.json").read_bytes()
    if sha(raw) != manifest["config"]["final_seed_commitment"]:
        raise RuntimeError("Protected final seed commitment mismatch")
    return json.loads(raw)


def verify_chosen_snapshot(directory, manifest, selected, gate):
    verify_frozen(directory, manifest)
    chosen = selected if gate.get("passed") else "current"
    path = directory/"snapshots"/(chosen+".json")
    if chosen != "current":
        lock = read(directory/"selection-lock.json")
        if lock["selected"] != chosen or lock["selected_snapshot_sha256"] != sha(path.read_bytes()):
            raise RuntimeError("Review bundle differs from locked/tested candidate")
    elif sha(path.read_bytes()) != manifest["files"]["snapshots/current.json"]:
        raise RuntimeError("Frozen current fallback snapshot changed")
    return chosen, read(path)


def package(directory, manifest, selected, gate):
    chosen, payload = verify_chosen_snapshot(directory, manifest, selected, gate)
    output = directory/"output/release-candidate"
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in RUNTIME_FILES:
        (output/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(directory/"frozen"/name, output/name)
    (output/"strategy.py").write_text(payload["source"])
    (output/"opponent_model.py").write_text(payload["model_source"])
    for source, asset, name in (("network_source", "network_weights_b64", "equity_network"), ("river_source", "river_weights_b64", "river_solver")):
        if payload.get(source):
            (output/(name+".py")).write_text(payload[source])
            asset_name = "equity_network.npz" if name == "equity_network" else "river_policy.npz"
            (output/asset_name).write_bytes(base64.b64decode(payload[asset]))
    if chosen != "current":
        policy = read(output/"policy.json")
        policy["parameters"] = payload["parameters"]
        policy["selection"]["overnight"] = {"candidate": chosen, "current_control_commit": manifest["current_git_commit"],
                                              "final_gate_passed": True, "requires_organizer_runtime_review": True}
        atomic(output/"policy.json", policy)
    result = {"chosen": chosen, "tested_candidate": selected, "final_gate_passed": bool(gate.get("passed")),
              "ready_for_review": True, "statistics_gates_passed": bool(gate.get("passed")), "runtime_verified": False,
              "auto_promoted": False, "original_runtime_modified": False,
              "current_control_commit": manifest["current_git_commit"], "gate": gate,
              "files": {str(p.relative_to(output)): sha(p.read_bytes()) for p in sorted(output.rglob("*")) if p.is_file()}}
    atomic(output/"MANIFEST.json", result)
    (output/"README.md").write_text("# Offline review bundle\n\n"+
        ("The locked candidate passed the predeclared independent final gates.\n" if gate.get("passed") else
         "No new candidate passed all independent final gates. This bundle preserves the frozen current neural bot.\n")+
        "\nThe deployed repository was not changed. No automatic promotion or GitHub push occurred.\n"
        "No completed new Astra matchup was available. Synthetic/public-probe results do not establish LLM superiority or tournament win probability.\n"
        "Direct timing excludes HTTP; check the organizer Docker startup and thinking bank before submission.\n"
        "\nBuild base: docker build -f docker/Dockerfile.base -t poker-agent-base:local .\n"
        "Build bot: docker build -f docker/Dockerfile.agent -t jr007-review .\n"
        "Run: docker run --rm -p 5008:5008 jr007-review\n")
    result["files"] = {str(p.relative_to(output)): sha(p.read_bytes())
                       for p in sorted(output.rglob("*")) if p.is_file() and p.name != "MANIFEST.json"}
    atomic(output/"MANIFEST.json", result)
    archive = directory/"output/release-candidate.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                handle.write(path, str(path.relative_to(output)))
    atomic(directory/"output/decision.json", dict(result, archive_sha256=sha(archive.read_bytes())))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--deadline")
    parser.add_argument("--hours", type=float, default=6.)
    parser.add_argument("--reserve-minutes", type=float, default=50.)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--screen-count", type=int, default=16)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--final-deals", type=int, default=1000)
    parser.add_argument("--final-seeds", type=int, default=5)
    parser.add_argument("--extra-opponent-bundle", type=Path, action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read(args.run_dir/"status.json"), indent=2))
        return
    if args.workers < 1 or args.workers > 4 or args.final_seeds < 3 or args.final_deals < 2:
        parser.error("workers must be 1..4, final-seeds>=3, final-deals>=2")
    # A nonblocking process lock prevents two resumed schedulers from running
    # the same protected protocol concurrently.
    args.run_dir.mkdir(parents=True, exist_ok=True)
    directory = args.run_dir.resolve()
    lock_handle = (directory/"run.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another scheduler already owns this run")
    directory, state, manifest = initialize(args)
    SCHEDULER.update(directory=directory, manifest=manifest)
    (directory/"PID").write_text(str(os.getpid())+"\n")
    if state.get("terminal"):
        log(directory, "Run already terminal; no new games or renewed deadline")
        return
    if (directory/"selection-lock.json").exists():
        # Covers a crash after fsync(lock) but before status.phase was saved.
        state["phase"] = "final"
    def stop(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    config = manifest["config"]
    pool = cf.ProcessPoolExecutor(max_workers=config["workers"], mp_context=mp.get_context("spawn"),
                                  initializer=worker_init, initargs=(str(directory),))
    gate = None
    selected = "current"
    try:
        search_cutoff = state["deadline_epoch"]-config["reserve_seconds"]
        while state["phase"] == "development" and not STOP and time.time() < search_cutoff:
            if config["max_rounds"] and state["round"] >= config["max_rounds"]:
                break
            round_id = state["round"]
            plan_path = directory/"plans"/(f"round-{round_id:04d}.json")
            if plan_path.exists():
                plan = read(plan_path)
                ids = plan["candidates"]
            else:
                ids = propose(directory, state, manifest)
                plan = {"round": round_id, "candidates": ids}
                atomic(plan_path, plan)
                status(directory, state)
            if not ids:
                raise RuntimeError("Could not generate a new distinct candidate")
            core = [p for p in config["development_profiles"] if p in ("loose_aggressive", "draw_range", "adaptive_pressure", "current", "challenge_value", "challenge_polarized") or p.startswith("public_")]
            if config["quick"]:
                core = config["development_profiles"]
            contenders = ids
            for stage, profiles, nseeds, deals, keep in (("screen", core, 2, config["screen_deals"], 4),
                ("confirm", config["development_profiles"], 2 if config["quick"] else 3, config["confirm_deals"], 2),
                ("long", config["development_profiles"], 2 if config["quick"] else 4, config["long_deals"], 2)):
                # All candidates within a fidelity use the same fresh cards.
                seeds = [f"{config['dev_namespace']}-r{round_id}-{stage}-{i}" for i in range(nseeds)]
                if stage == "long" and state.get("development_leader"):
                    contenders = list(dict.fromkeys(contenders+[state["development_leader"]]))
                tasks = make_tasks(directory, state, contenders, f"r{round_id}-{stage}", profiles, seeds, deals)
                paths, complete = run_tasks(directory, state, manifest, pool, tasks, search_cutoff)
                if not complete:
                    break
                ranked = sorted((metrics(paths, c) for c in contenders if c not in state.get("failed_candidates", {})), key=lambda v: v["score"], reverse=True)
                if not ranked:
                    log(directory, f"No complete legal candidate remains in round {round_id}/{stage}")
                    break
                atomic(directory/"scores"/(f"round-{round_id:04d}-{stage}.json"), ranked)
                contenders = [v["candidate"] for v in ranked[:keep]]
                if stage == "long":
                    for value in ranked:
                        value["round"] = round_id
                    previous = {v["candidate"]: v for v in state["archive"]}
                    previous.update({v["candidate"]: v for v in ranked})
                    state["archive"] = sorted(previous.values(), key=lambda v: v["score"], reverse=True)[:12]
                    state["latest_long_ranking"] = ranked
                    qualified = [v for v in ranked if v["feasible"] and v["score"] > 0]
                    state["development_leader"] = qualified[0]["candidate"] if qualified else None
                    state["history"].append({"round": round_id, "scores": [{"candidate": v["candidate"], "score": v["score"], "feasible": v["feasible"]} for v in ranked]})
                    status(directory, state, selected_current=state["development_leader"] or "current")
            if not complete:
                break
            state["round"] += 1
            status(directory, state)
        if STOP:
            status(directory, state, phase=state["phase"], paused=True, message="Stopped safely; --resume retains original deadline and selection lock")
            return
        selection = lock_selection(directory, state, manifest)
        selected = selection["selected"]
        if (directory/"final-gate.json").exists():
            # A persisted final decision is never retested, including after a
            # crash between gate persistence, packaging, and terminal status.
            gate = read(directory/"final-gate.json")
        elif selected == "current":
            gate = {"passed": False, "reason": "No feasible development improvement; preserve frozen current"}
        else:
            seeds = open_final_seeds(directory, state, manifest)
            tasks = make_tasks(directory, state, [selected], "protected-final", config["final_profiles"], seeds, config["final_deals"])
            paths, complete = run_tasks(directory, state, manifest, pool, tasks, state["deadline_epoch"]-10)
            if STOP:
                status(directory, state, paused=True, message="Final candidate remains irrevocably locked; resume cannot select a runner-up")
                return
            gate = final_gate(paths, selected, config["final_profiles"], seeds, config["final_deals"]) if complete else {
                "passed": False, "complete": False, "reason": "Absolute deadline interrupted final suite; preserve current"}
        if not (directory/"final-gate.json").exists():
            atomic(directory/"final-gate.json", gate)
        # The final decision is terminal before supplementary verification.
        # Additional cases never change parameters or test a second finalist.
        result = package(directory, manifest, selected, gate)
        status(directory, state, phase="verification", selected_current=result["chosen"], tested_candidate=selected,
               final_decision=result["chosen"], final_gate_passed=result["final_gate_passed"],
               message="Independent final decision fixed; remaining CPU verifies frozen choice only, without tuning")
        check_round = state.get("verification_round", 0)
        while not config["quick"] and not STOP and time.time() < state["deadline_epoch"]-20:
            chosen = result["chosen"]
            profiles = list(config["final_profiles"])
            if check_round % 3 == 0 and len(state["archive"]) > 1:
                profiles += ["candidate:"+v["candidate"] for v in state["archive"][:3] if v["candidate"] != chosen]
            seed = f"overnight-postcheck-{state['manifest_sha256'][:12]}-{check_round}"
            tasks = make_tasks(directory, state, [chosen], f"postcheck-{check_round}", profiles, [seed], config["final_deals"])
            paths, complete = run_tasks(directory, state, manifest, pool, tasks, state["deadline_epoch"]-10)
            if complete:
                atomic(directory/"verification"/(f"check-{check_round:04d}.json"), {"candidate": chosen, "purpose": "Supplementary diagnostics only; no tuning or replacement of the locked candidate", "summary": metrics(paths, chosen)})
            if not complete:
                break
            check_round += 1
            status(directory, state, verification_round=check_round)
        status(directory, state, phase="completed", terminal=True, paused=False, selected_current=result["chosen"],
               message="Review bundle ready; original runtime untouched; no automatic push or promotion; no completed new Astra game")
        log(directory, f"Completed: chosen={result['chosen']}; independent-gates={result['final_gate_passed']}; bundle={directory/'output/release-candidate.zip'}")
    except Exception as error:
        log(directory, f"ERROR {type(error).__name__}: {error}")
        # An error after final selection cannot cause a different candidate to
        # be tried. The safe review artifact always remains current.
        if (directory/"final-gate.json").exists():
            preserved = read(directory/"final-gate.json")
            status(directory, state, phase="failed", terminal=True, final_gate_passed=preserved.get("passed", False),
                   error=str(error), message="Fixed final decision preserved; packaging/verification needs review")
        elif (directory/"selection-lock.json").exists():
            gate = {"passed": False, "reason": f"Final protocol error: {type(error).__name__}: {error}; preserve current"}
            atomic(directory/"final-gate.json", gate)
            package(directory, manifest, selected, gate)
            status(directory, state, phase="failed", terminal=True, selected_current="current", final_gate_passed=False, error=str(error))
        else:
            status(directory, state, phase="development", paused=True, error=str(error))
        raise
    finally:
        stop_pool(pool)
        reconcile_counts(directory, state)
        status(directory, state)


if __name__ == "__main__":
    main()
