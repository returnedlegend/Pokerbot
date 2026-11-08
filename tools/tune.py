#!/usr/bin/env python3
"""Reproducible small evolutionary search on DEVELOPMENT deals only.

The production strategy exports PARAMS and optionally PARAM_BOUNDS as
{parameter: [lower, upper]}. Alternatively pass --space path/to/bounds.json.
Outputs all candidates and the selected parameter JSON. Evaluate the selected
file on fresh seeds and held-out opponents with benchmark.py; tuning results
are not evidence of out-of-sample strength.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import run_match, summarize


def objective(by_opponent):
    """Prefer broad edges, capping rewards from the most exploitable probes.

    The minimum-opponent term penalizes sacrificing one matchup to farm weak
    ones. This is a search heuristic, never a tournament score estimate.
    """
    means = [r["mean_bb_per_deal"] for r in by_opponent.values()]
    return sum(max(-8.0, min(8.0, m)) for m in means) / len(means) + .7*min(means)


def evaluate(module, params, agent, opponents, seeds, deals):
    module.PARAMS.clear()
    module.PARAMS.update(params)
    matches = [run_match(agent, opponent, seed, deals)
               for opponent in opponents for seed in seeds]
    by_opponent = {o: summarize([x for m in matches if m["opponent"] == o for x in m["paired_bb"]])
                   for o in opponents}
    return {"params": dict(params), "objective": objective(by_opponent),
            "opponents": by_opponent,
            "max_strategy_seconds_per_half": max(h["agent"]["strategy_seconds"]
                                                  for m in matches for h in m["half_stats"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", default="strategy")
    parser.add_argument("--agent", default=str(ROOT / "agent.py"))
    parser.add_argument("--space", type=Path)
    parser.add_argument("--initial", type=Path, help="Initial complete parameter JSON")
    parser.add_argument("--opponents", nargs="+", default=["random", "call", "pot", "mc_loose"])
    parser.add_argument("--seeds", nargs="+", default=["tune-101", "tune-307"])
    parser.add_argument("--deals", type=int, default=150)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--search-seed", type=int, default=812)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "tuning.json")
    parser.add_argument("--params-output", type=Path, default=ROOT / "tuned_params.json")
    args = parser.parse_args()
    module = importlib.import_module(args.module)
    initial = json.loads(args.initial.read_text()) if args.initial else dict(module.PARAMS)
    initial = initial.get("parameters", initial)
    space = json.loads(args.space.read_text()) if args.space else getattr(module, "PARAM_BOUNDS", None)
    if not space:
        parser.error("Provide --space or export PARAM_BOUNDS in the strategy module")
    for key, bounds in space.items():
        if key not in initial or len(bounds) != 2 or bounds[0] > bounds[1]:
            parser.error(f"Invalid search bounds for {key}")
        if not bounds[0] <= initial[key] <= bounds[1]:
            parser.error(f"Initial parameter {key} is outside its search bounds")
    rng = random.Random(args.search_seed)
    records = []
    started = time.time()
    best = initial
    best_score = -math.inf
    keys = list(space)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.params_output.parent.mkdir(parents=True, exist_ok=True)
    for iteration in range(args.iterations+1):
        if iteration == 0:
            candidate = dict(initial)
        else:
            candidate = dict(best)
            # Vary one or two parameters together. Keep every proposal in the
            # documented bounds and shrink steps as local search progresses.
            changed = rng.sample(keys, min(len(keys), 1 if rng.random() < .65 else 2))
            scale = .22 * (.45 + .55 * (1 - iteration/max(1, args.iterations)))
            for key in changed:
                low, high = space[key]
                proposal = candidate[key] + rng.gauss(0, scale*(high-low))
                candidate[key] = min(high, max(low, proposal))
        result = evaluate(module, candidate, args.agent, args.opponents, args.seeds, args.deals)
        result["iteration"] = iteration
        records.append(result)
        improved = result["objective"] > best_score
        if improved:
            best, best_score = dict(candidate), result["objective"]
            args.params_output.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
        print(f"candidate {iteration}/{args.iterations}: objective={result['objective']:+.4f} "
              f"{'BEST' if improved else ''} "
              + " ".join(f"{o}={r['mean_bb_per_deal']:+.3f}" for o,r in result["opponents"].items()),
              flush=True)
        args.output.write_text(json.dumps({
            "purpose": "Development tuning only; requires fresh-seed validation",
            "opponents": args.opponents, "seeds": args.seeds, "deals": args.deals,
            "search_seed": args.search_seed, "space": space, "best": best,
            "best_objective": best_score, "elapsed_seconds": time.time()-started,
            "candidates": records}, indent=2) + "\n")
    module.PARAMS.clear()
    module.PARAMS.update(best)


if __name__ == "__main__":
    main()
