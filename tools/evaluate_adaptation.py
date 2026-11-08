#!/usr/bin/env python3
"""Compare a candidate strategy with a frozen Git incumbent on style probes.

Example:
  python tools/evaluate_adaptation.py --phase development --deals 300 \\
    --output results/adaptation-development.json

Each policy has its own module namespace and parameters. Each match uses
fresh memory per half and duplicate cards with seats exchanged. Comparing
candidate/incumbent against the same opponent and deals produces paired
improvement estimates. All timing is direct function time, not HTTP time-bank
usage; the official judge remains the deployment check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from benchmark import Opponent, BUILTINS, summarize
from game import Action, AgentConfig, BIG_BLIND, Game, button_for_deal, make_deal
from style_opponents import STYLES, StyleOpponent

FROZEN_REF = "50b4682550455b60bf262fe0a09c60dc6269e1a2"
DEVELOPMENT_SEEDS = ("adapt-dev1", "adapt-dev2")
VALIDATION_SEEDS = ("adapt-hold101", "adapt-hold307", "adapt-hold809")


def committed(ref, name):
    return subprocess.check_output(["git", "show", f"{ref}:{name}"], cwd=ROOT)


class StrategyFactory:
    """Immutable source snapshots with separate module state per half."""
    def __init__(self, source: bytes, parameters: dict, name: str, file: Path):
        self.source = source
        self.parameters = dict(parameters)
        self.name = name
        self.file = file

    def create(self, half):
        module = ModuleType(f"adaptation_{self.name}_{half}_{id(self)}")
        # Frozen strategy's table/config are intentionally resolved from the
        # same repository root. Its PARAMS are replaced with committed values
        # immediately after import, never shared with the candidate module.
        module.__file__ = str(self.file)
        sys.modules[module.__name__] = module
        exec(compile(self.source, str(self.file), "exec"), module.__dict__)
        module.PARAMS.clear()
        module.PARAMS.update(self.parameters)
        return module


def factory_from_candidate(source_path, override=None):
    p = Path(source_path).resolve()
    policy = json.loads((ROOT / "policy.json").read_text())["parameters"]
    # Initialize source once to discover its default keys, including new
    # policy controls absent from the older policy JSON.
    probe = ModuleType("adaptation_candidate_defaults")
    probe.__file__ = str(p)
    sys.modules[probe.__name__] = probe
    source = p.read_bytes()
    exec(compile(source, str(p), "exec"), probe.__dict__)
    params = dict(probe.PARAMS)
    params.update(policy)
    if override:
        params.update(override)
    return StrategyFactory(source, params, "candidate", p)


def load_opponent(name, incumbent, half):
    if name in STYLES:
        return StyleOpponent(name)
    if name == "incumbent":
        return incumbent.create(half+2)
    if name in BUILTINS:
        return Opponent(name)
    if name == "fast_range":
        import fast_range_opponent
        return fast_range_opponent
    raise ValueError(f"Unknown opponent: {name}")


def run_match(factory, opponent_name, incumbent, seed, deals):
    config = AgentConfig(num_hands=deals)
    halves, half_stats = [], []
    started = time.perf_counter()
    for half in range(2):
        bots = [factory.create(half), load_opponent(opponent_name, incumbent, half)]
        memories = [{}, {}]
        stats = [{"calls": 0, "seconds": 0., "max_decision_ms": 0.,
                  "actions": {a.name: 0 for a in Action},
                  "by_street": [{"calls": 0, "raises": 0} for _ in range(4)],
                  "segments": [{"calls":0,"half_raises":0,"pot_raises":0} for _ in range(2)]}
                 for _ in bots]
        for bot, memory in zip(bots, memories):
            bot.new_half(config, memory)
        nets = []
        for index in range(deals):
            game = Game(make_deal(seed, index), button_for_deal(index), hand_index=index)
            while not game.is_terminal:
                seat = game.to_act
                who = seat ^ half
                obs = game.observation(seat)
                begin = time.perf_counter()
                action = bots[who].choose_action(obs, memories[who])
                elapsed = time.perf_counter() - begin
                st = stats[who]
                st["calls"] += 1
                st["seconds"] += elapsed
                st["max_decision_ms"] = max(st["max_decision_ms"], elapsed*1000)
                if not isinstance(action, Action) or action not in obs.legal_actions:
                    raise RuntimeError(f"Illegal action {action!r} from bot {who}, half {half}, deal {index}")
                st["actions"][action.name] += 1
                segment = st["segments"][int(index >= max(40,min(400,deals//2)))]
                segment["calls"] += 1
                segment["half_raises"] += int(action == Action.RAISE_HALF)
                segment["pot_raises"] += int(action == Action.RAISE_POT)
                street_stats = st["by_street"][int(obs.street)]
                street_stats["calls"] += 1
                street_stats["raises"] += action in (Action.RAISE_HALF, Action.RAISE_POT)
                game.apply(action)
            nets.append(game.result().nets[half])
            for who, bot in enumerate(bots):
                begin = time.perf_counter()
                bot.on_hand_end(game.hand_result(who ^ half), memories[who])
                stats[who]["seconds"] += time.perf_counter()-begin
        for st in stats:
            raises = st["actions"]["RAISE_HALF"] + st["actions"]["RAISE_POT"]
            st["raise_fraction"] = raises/max(1, st["calls"])
            st["raises_per_hand"] = raises/deals
        halves.append(nets)
        half_stats.append(stats)
    paired = [(a+b)/BIG_BLIND for a,b in zip(*halves)]
    return {"strategy": factory.name, "opponent": opponent_name, "seed": seed,
            **summarize(paired), "paired_bb": paired, "half_stats": half_stats,
            "elapsed_seconds": time.perf_counter()-started,
            "phase_split": max(40,min(400,deals//2))}


def aggregate(matches):
    return {name: summarize([v for m in matches if m["opponent"] == name for v in m["paired_bb"]])
            for name in sorted(set(m["opponent"] for m in matches))}



def raising_summary(matches):
    summary = {}
    for name in sorted(set(m["opponent"] for m in matches)):
        summary[name] = {}
        for strategy in ("incumbent", "candidate"):
            selected = [m for m in matches if m["opponent"] == name and m["strategy"] == strategy]
            stats = [h[0] for m in selected for h in m["half_stats"]]
            calls = sum(h["calls"] for h in stats)
            half = sum(h["actions"]["RAISE_HALF"] for h in stats)
            pot = sum(h["actions"]["RAISE_POT"] for h in stats)
            hands = 2*sum(m["deals"] for m in selected)
            summary[name][strategy] = {"calls": calls,"half_raises":half,"pot_raises":pot,
                "raise_fraction":(half+pot)/max(1,calls),"pot_raise_fraction":pot/max(1,calls),
                "raises_per_hand":(half+pot)/max(1,hands),"pot_raises_per_hand":pot/max(1,hands)}
    return summary


def phase_aggregate(matches):
    return {name: {
        "before_switch": summarize([v for m in matches if m["opponent"] == name for v in m["paired_bb"][:m["phase_split"]]]),
        "after_switch": summarize([v for m in matches if m["opponent"] == name for v in m["paired_bb"][m["phase_split"]:]])}
        for name in sorted(set(m["opponent"] for m in matches))}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "validation"), default="development")
    parser.add_argument("--candidate", type=Path, default=ROOT / "strategy.py")
    parser.add_argument("--incumbent-ref", default=FROZEN_REF)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--deals", type=int, default=300)
    parser.add_argument("--opponents", nargs="+", default=list(STYLES)+["incumbent"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/adaptation-development.json")
    args = parser.parse_args()
    if args.deals < 2:
        parser.error("At least two deals are needed")
    seeds = args.seeds or (DEVELOPMENT_SEEDS if args.phase == "development" else VALIDATION_SEEDS)
    override = json.loads(args.params.read_text()) if args.params else None
    if override is not None:
        override = override.get("parameters", override)
    candidate = factory_from_candidate(args.candidate, override)
    old_source = committed(args.incumbent_ref, "strategy.py")
    old_params = json.loads(committed(args.incumbent_ref, "policy.json"))["parameters"]
    incumbent = StrategyFactory(old_source, old_params, "incumbent", ROOT / "strategy.py")
    fingerprints = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                    for n in ("strategy.py", "opponent_model.py", "policy.json", "equity_engine.py", "preflop_model.npz",
                              "tools/style_opponents.py", "tools/evaluate_adaptation.py")}
    all_runs, improvements = [], []
    report = {"phase": args.phase, "seed_role": "Development" if args.phase == "development" else "Fresh validation deals; profiles were previously inspected",
              "incumbent_ref": args.incumbent_ref, "incumbent_strategy_sha256": hashlib.sha256(old_source).hexdigest(),
              "candidate_strategy_sha256": hashlib.sha256(candidate.source).hexdigest(),
              "candidate_parameters": candidate.parameters, "incumbent_parameters": incumbent.parameters,
              "source_sha256_at_start": fingerprints,
              "timing_scope": "Direct function calls; excludes HTTP and no time-bank enforcement",
              "matches": all_runs, "paired_improvements": improvements}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for opponent in args.opponents:
        for seed in seeds:
            current = []
            for factory in (incumbent, candidate):
                result = run_match(factory, opponent, incumbent, seed, args.deals)
                current.append(result)
                all_runs.append(result)
                print(f"{opponent} {seed} {factory.name}: {result['mean_bb_per_deal']:+.4f} "
                      f"CI={result['ci95']} raise_fraction="
                      f"{sum(h[0]['raise_fraction'] for h in result['half_stats'])/2:.3f}", flush=True)
            difference = [a-b for a,b in zip(current[1]["paired_bb"], current[0]["paired_bb"])]
            improvement = {"opponent": opponent, "seed": seed, **summarize(difference), "paired_bb": difference, "phase_split": current[1]["phase_split"]}
            improvements.append(improvement)
            report["elapsed_seconds"] = time.perf_counter()-started
            report["candidate_aggregate"] = aggregate([m for m in all_runs if m["strategy"] == "candidate"])
            report["incumbent_aggregate"] = aggregate([m for m in all_runs if m["strategy"] == "incumbent"])
            report["improvement_aggregate"] = aggregate(improvements)
            report["raise_summary"] = raising_summary(all_runs)
            report["candidate_phase_aggregate"] = phase_aggregate([m for m in all_runs if m["strategy"] == "candidate"])
            report["incumbent_phase_aggregate"] = phase_aggregate([m for m in all_runs if m["strategy"] == "incumbent"])
            report["improvement_phase_aggregate"] = phase_aggregate(improvements)
            report["source_sha256_at_end"] = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in fingerprints}
            report["runtime_source_unchanged"] = all(report["source_sha256_at_end"][n] == fingerprints[n] for n in ("strategy.py", "opponent_model.py", "policy.json", "equity_engine.py", "preflop_model.npz"))
            args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"improvement_aggregate": report["improvement_aggregate"],
                      "candidate_aggregate": report["candidate_aggregate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
