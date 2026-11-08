#!/usr/bin/env python3
"""Paired duplicate comparison with a fully frozen strategy/opponent model.

Development uses strength-dev101/307. Validation reserves strength-hold211,
503, and 907. Opponents see observations only. Time measurements are direct
function calls, without HTTP or enforcement of the tournament time bank.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import ModuleType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"tools"))

from benchmark import Opponent, BUILTINS, summarize
from game import Action, AgentConfig, BIG_BLIND, Game, button_for_deal, make_deal
from style_opponents import STYLES, StyleOpponent
from strength_opponents import PROFILES, StrengthOpponent

FROZEN_REF = "97883009569ad2d383b8145e46ee642262390049"
DEVELOPMENT_SEEDS = ("strength-dev101", "strength-dev307")
VALIDATION_SEEDS = ("strength-hold211", "strength-hold503", "strength-hold907")
DEFAULT_OPPONENTS = (*STYLES, "fast_range", *PROFILES, "incumbent")


def committed(ref, name):
    return subprocess.check_output(["git", "show", f"{ref}:{name}"], cwd=ROOT)


def source_defaults(source):
    """Read literal defaults without accidentally loading current policy.json."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError("Strategy must define literal PARAMS defaults")


def imports_module(source, name):
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(alias.name == name for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == name:
            return True
    return False


def imports_network(source):
    return imports_module(source, "equity_network")


def comparable_manifest(manifest):
    """New optional, unused dependencies remain compatible with older reports."""
    return {name: value for name, value in manifest.items() if value is not None}


def configured_source(source, parameters):
    """Apply captured parameters before import-time feature initialization.

    Replacing PARAMS only after exec is too late for optional model warmup:
    an ablation could otherwise initialize using the working-tree config.
    Keep the source manifest unchanged, but compile its literal defaults with
    captured parameters and omit its known top-level policy.json override.
    """
    tree = ast.parse(source)
    for index, node in enumerate(tree.body):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in node.targets):
            node.value = ast.parse(repr(parameters), mode="eval").body
        elif isinstance(node, ast.If):
            updates = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Attribute) and n.func.attr == "update"
                       and isinstance(n.func.value, ast.Name) and n.func.value.id == "PARAMS"]
            reads_config = any(isinstance(n, ast.Name) and n.id == "_config" for n in ast.walk(node))
            if updates and reads_config:
                tree.body[index] = ast.copy_location(ast.Pass(), node)
    return ast.fix_missing_locations(tree)


class StrategyFactory:
    """Immutable source bytes; each half has two isolated module namespaces.

    Loading a strategy with exec alone leaves its transitive opponent_model
    import attached to the current working tree. Replace that reference with
    a separately executed snapshot, for both candidate and committed baseline.
    """
    def __init__(self, source, model_source, parameters, name, file=None, network=None, river=None):
        self.source = source
        self.model_source = model_source
        self.parameters = source_defaults(source)
        self.parameters.update(parameters)
        self.name = name
        self.file = Path(file or ROOT/"strategy.py")
        self.serial = 0
        self.network = network
        self.network_module = None
        self.river = river
        self.river_module = None

    def _frozen_river(self):
        if self.river_module is None:
            source, weights = self.river
            module = ModuleType(f"strength_{self.name}_river_solver_{id(self)}")
            module.__file__ = str(ROOT/"river_solver.py")
            sys.modules[module.__name__] = module
            exec(compile(source, module.__file__, "exec"), module.__dict__)
            original_file = module.__file__
            with tempfile.TemporaryDirectory(prefix="strength-river-", dir="/tmp") as directory:
                (Path(directory)/"river_policy.npz").write_bytes(weights)
                module.__file__ = str(Path(directory)/"river_solver.py")
                try:
                    module._model()
                finally:
                    module.__file__ = original_file
            self.river_module = module
        return self.river_module

    def _frozen_network(self):
        if self.network_module is None:
            source, weights = self.network
            module = ModuleType(f"strength_{self.name}_equity_network_{id(self)}")
            module.__file__ = str(ROOT/"equity_network.py")
            # Numba resolves function globals through their module name at
            # compilation, so retain this one immutable module per factory.
            sys.modules[module.__name__] = module
            exec(compile(source, module.__file__, "exec"), module.__dict__)
            # Run the captured production loader on captured bytes. In
            # particular, preserve its denormal-weight sanitization rather
            # than bypassing it by assigning raw arrays directly to _MODEL.
            original_file = module.__file__
            with tempfile.TemporaryDirectory(prefix="strength-nn-", dir="/tmp") as directory:
                frozen_file = Path(directory)/"equity_network.npz"
                frozen_file.write_bytes(weights)
                module.__file__ = str(Path(directory)/"equity_network.py")
                try:
                    module._load_model()
                    metadata = module.metadata()
                finally:
                    module.__file__ = original_file
            module.metadata = lambda: dict(metadata)
            self.network_module = module
        return self.network_module

    def create(self, half):
        self.serial += 1
        prefix = f"strength_{self.name}_{half}_{id(self)}_{self.serial}"
        model = ModuleType(prefix+"_opponent_model")
        model.__file__ = str(ROOT/"opponent_model.py")
        # Register only during exec for decorators that look up __module__.
        # References survive after removal; modules cannot accumulate per run.
        sys.modules[model.__name__] = model
        try:
            exec(compile(self.model_source, model.__file__, "exec"), model.__dict__)
        finally:
            sys.modules.pop(model.__name__, None)
        strategy = ModuleType(prefix+"_strategy")
        strategy.__file__ = str(self.file)
        sys.modules[strategy.__name__] = strategy
        missing = object()
        previous_model = sys.modules.get("opponent_model", missing)
        sys.modules["opponent_model"] = model
        previous_network = sys.modules.get("equity_network", missing)
        previous_river = sys.modules.get("river_solver", missing)
        try:
            if self.network:
                sys.modules["equity_network"] = self._frozen_network()
            if self.river:
                sys.modules["river_solver"] = self._frozen_river()
            exec(compile(configured_source(self.source, self.parameters), str(self.file), "exec"), strategy.__dict__)
        finally:
            sys.modules.pop(strategy.__name__, None)
            if previous_model is missing:
                sys.modules.pop("opponent_model", None)
            else:
                sys.modules["opponent_model"] = previous_model
            if self.network:
                if previous_network is missing:
                    sys.modules.pop("equity_network", None)
                else:
                    sys.modules["equity_network"] = previous_network
            if self.river:
                if previous_river is missing:
                    sys.modules.pop("river_solver", None)
                else:
                    sys.modules["river_solver"] = previous_river
        strategy.opponent_model = model
        strategy.PARAMS.clear()
        strategy.PARAMS.update(self.parameters)
        return strategy

    def manifest(self):
        return {"strategy_sha256": hashlib.sha256(self.source).hexdigest(),
                "opponent_model_sha256": hashlib.sha256(self.model_source).hexdigest(),
                "equity_network_source_sha256": hashlib.sha256(self.network[0]).hexdigest() if self.network else None,
                "equity_network_weights_sha256": hashlib.sha256(self.network[1]).hexdigest() if self.network else None,
                "river_solver_source_sha256": hashlib.sha256(self.river[0]).hexdigest() if self.river else None,
                "river_policy_weights_sha256": hashlib.sha256(self.river[1]).hexdigest() if self.river else None,
                "parameters": self.parameters}

    def save_snapshot(self, path):
        """Persist exact candidate bytes for deferred comparisons/ablations."""
        payload = {"source": self.source.decode(), "model_source": self.model_source.decode(),
                   "parameters": self.parameters, "file": str(self.file),
                   "network_source": self.network[0].decode() if self.network else None,
                   "network_weights_b64": base64.b64encode(self.network[1]).decode() if self.network else None,
                   "river_source": self.river[0].decode() if self.river else None,
                   "river_weights_b64": base64.b64encode(self.river[1]).decode() if self.river else None}
        Path(path).write_text(json.dumps(payload)+"\n")


def factory_from_snapshot(path, override=None):
    payload = json.loads(Path(path).read_text())
    network = ((payload["network_source"].encode(), base64.b64decode(payload["network_weights_b64"]))
               if payload.get("network_source") else None)
    parameters = dict(payload["parameters"])
    parameters.update(override or {})
    river = ((payload["river_source"].encode(), base64.b64decode(payload["river_weights_b64"]))
             if payload.get("river_source") else None)
    return StrategyFactory(payload["source"].encode(), payload["model_source"].encode(),
                           parameters, "candidate", payload["file"], network, river)


def factory_from_candidate(path, override=None, model_path=None):
    path = Path(path).resolve()
    model_path = Path(model_path or ROOT/"opponent_model.py").resolve()
    parameters = json.loads((ROOT/"policy.json").read_text())["parameters"]
    parameters.update(override or {})
    source = path.read_bytes()
    network = ((ROOT/"equity_network.py").read_bytes(), (ROOT/"equity_network.npz").read_bytes()) if imports_network(source) else None
    river = ((ROOT/"river_solver.py").read_bytes(), (ROOT/"river_policy.npz").read_bytes()) if imports_module(source, "river_solver") else None
    return StrategyFactory(source, model_path.read_bytes(), parameters, "candidate", path, network, river)


def factory_from_commit(ref=FROZEN_REF):
    source = committed(ref, "strategy.py")
    network = (committed(ref, "equity_network.py"), committed(ref, "equity_network.npz")) if imports_network(source) else None
    river = (committed(ref, "river_solver.py"), committed(ref, "river_policy.npz")) if imports_module(source, "river_solver") else None
    return StrategyFactory(source, committed(ref, "opponent_model.py"),
                           json.loads(committed(ref, "policy.json"))["parameters"], "incumbent", network=network, river=river)


def load_opponent(name, incumbent, half, external_sources=None):
    if name in STYLES:
        return StyleOpponent(name)
    if name in PROFILES:
        return StrengthOpponent(name)
    if name == "incumbent":
        return incumbent.create(half+2)
    if name == "fast_range":
        import fast_range_opponent
        return fast_range_opponent
    if name in BUILTINS:
        return Opponent(name)
    if external_sources and name in external_sources:
        path, source = external_sources[name]
        module = ModuleType(f"strength_external_{Path(path).stem}_{half}_{time.perf_counter_ns()}")
        module.__file__ = path
        sys.modules[module.__name__] = module
        try:
            exec(compile(source, path, "exec"), module.__dict__)
        except BaseException:
            sys.modules.pop(module.__name__, None)
            raise
        # Keep the isolated module registered for Numba's later resolution of
        # function globals. Every half still receives a distinct namespace.
        for callback in ("new_half", "choose_action", "on_hand_end"):
            if not callable(getattr(module, callback, None)):
                raise ValueError(f"{name} has no callable {callback}")
        return module
    raise ValueError(f"Unknown opponent: {name}")


def fresh_stats():
    return {"calls": 0, "seconds": 0., "max_decision_ms": 0., "errors": 0,
            "actions": {a.name: 0 for a in Action},
            "by_street": [{"calls": 0, "raises": 0, "pot_raises": 0} for _ in range(4)]}


def run_match(factory, opponent_name, incumbent, seed, deals, external_sources=None):
    config = AgentConfig(num_hands=deals)
    halves, half_stats = [], []
    start = time.perf_counter()
    for half in range(2):
        bots = [factory.create(half), load_opponent(opponent_name, incumbent, half, external_sources)]
        memories, stats = [{}, {}], [fresh_stats(), fresh_stats()]
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
                elapsed = time.perf_counter()-begin
                st = stats[who]
                st["calls"] += 1
                st["seconds"] += elapsed
                st["max_decision_ms"] = max(st["max_decision_ms"], elapsed*1000)
                if not isinstance(action, Action) or action not in obs.legal_actions:
                    raise RuntimeError(f"Illegal action {action!r}; bot={who}, half={half}, deal={index}")
                st["actions"][action.name] += 1
                street = st["by_street"][int(obs.street)]
                street["calls"] += 1
                street["raises"] += action in (Action.RAISE_HALF, Action.RAISE_POT)
                street["pot_raises"] += action == Action.RAISE_POT
                game.apply(action)
            nets.append(game.result().nets[half])
            for who, bot in enumerate(bots):
                begin = time.perf_counter()
                bot.on_hand_end(game.hand_result(who ^ half), memories[who])
                stats[who]["seconds"] += time.perf_counter()-begin
        halves.append(nets)
        half_stats.append(stats)
    paired = [(left+right)/BIG_BLIND for left, right in zip(*halves)]
    return {"strategy": factory.name, "opponent": opponent_name, "seed": seed,
            **summarize(paired), "paired_bb": paired, "half_stats": half_stats,
            "elapsed_seconds": time.perf_counter()-start}


def aggregate(matches):
    return {profile: summarize([v for m in matches if m["opponent"] == profile for v in m["paired_bb"]])
            for profile in sorted({m["opponent"] for m in matches})}


def clustered_summary(matches, excluded=()):
    """Average shared cards across profiles before estimating uncertainty.

    Deals are paired between candidate/baseline and between seat-swapped
    halves. Profiles sharing a seed/deal are not counted as independent hands.
    A moving-block bootstrap also preserves 50-deal local adaptation effects.
    """
    selected = [m for m in matches if m["opponent"] not in excluded]
    if not selected:
        return None
    clusters = {}
    for match in selected:
        for index, value in enumerate(match["paired_bb"]):
            clusters.setdefault((match["seed"], index), []).append(value)
    by_seed = {}
    for (seed, index), values in sorted(clusters.items()):
        by_seed.setdefault(seed, []).append(float(np.mean(values)))
    values = [v for group in by_seed.values() for v in group]
    result = summarize(values)
    result["profiles"] = sorted({m["opponent"] for m in selected})
    result["shared_card_clusters"] = len(values)
    result["unit"] = "bb per duplicate deal, profiles averaged within seed/deal"
    rng = np.random.default_rng(765193)
    bootstrap = np.zeros(2000)
    for sequence in by_seed.values():
        sequence = np.asarray(sequence)
        count = len(sequence)
        block = min(50, max(1, count//5))
        starts = rng.integers(0, count, size=(len(bootstrap), (count+block-1)//block))
        indices = (starts[..., None]+np.arange(block)) % count
        bootstrap += sequence[indices.reshape(len(bootstrap), -1)[:, :count]].sum(axis=1)
    bootstrap /= len(values)
    result["block_bootstrap_ci95"] = np.quantile(bootstrap, [.025, .975]).tolist()
    result["block_bootstrap_deals"] = min(50, max(1, min(map(len, by_seed.values()))//5))
    lo, hi = result["block_bootstrap_ci95"]
    result["block_bootstrap_verdict"] = "WIN" if lo > 0 else "LOSS" if hi < 0 else "CHOP"
    result["block_bootstrap_scope"] = "Conditional on the observed seed trajectories; dependencies longer than the block are not captured"
    seed_means = {seed: float(np.mean(sequence)) for seed, sequence in by_seed.items()}
    result["seed_means"] = seed_means
    result["seed_cluster_ci95"] = None
    result["seed_cluster_verdict"] = "INSUFFICIENT_SEEDS"
    if len(seed_means) >= 2:
        # Whole-half learning can induce dependence beyond a 50-deal block.
        # Treat independent match seeds as clusters too. Few seeds imply a
        # deliberately wide Student-t interval rather than false precision.
        critical = (None, 12.706205, 4.302653, 3.182446, 2.776445, 2.570582,
                    2.446912, 2.364624, 2.306004, 2.262157, 2.228139,
                    2.200985, 2.178813, 2.160369, 2.144787, 2.131450,
                    2.119905, 2.109816, 2.100922, 2.093024, 2.085963,
                    2.079614, 2.073873, 2.068658, 2.063899, 2.059539,
                    2.055529, 2.051831, 2.048407, 2.045230, 2.042272)
        df = len(seed_means)-1
        z = 1.959963984540054
        t = critical[df] if df < len(critical) else z+(z**3+z)/(4*df)+(5*z**5+16*z**3+3*z)/(96*df**2)
        mean = float(np.mean(list(seed_means.values())))
        se = float(np.std(list(seed_means.values()), ddof=1)/np.sqrt(len(seed_means)))
        result["seed_cluster_ci95"] = [mean-t*se, mean+t*se]
        lo, hi = result["seed_cluster_ci95"]
        result["seed_cluster_verdict"] = "WIN" if lo > 0 else "LOSS" if hi < 0 else "CHOP"
    return result


def action_summary(matches):
    output = {}
    for profile in sorted({m["opponent"] for m in matches}):
        output[profile] = {}
        for name in ("incumbent", "candidate"):
            selected = [m for m in matches if m["opponent"] == profile and m["strategy"] == name]
            stats = [half[0] for m in selected for half in m["half_stats"]]
            calls = sum(st["calls"] for st in stats)
            half = sum(st["actions"]["RAISE_HALF"] for st in stats)
            pot = sum(st["actions"]["RAISE_POT"] for st in stats)
            hands = 2*sum(m["deals"] for m in selected)
            output[profile][name] = {"calls": calls, "half_raises": half, "pot_raises": pot,
                "raise_fraction": (half+pot)/max(1, calls), "pot_raise_fraction": pot/max(1, calls),
                "raises_per_hand": (half+pot)/max(1, hands), "pot_raises_per_hand": pot/max(1, hands),
                "direct_seconds": sum(st["seconds"] for st in stats),
                "max_decision_ms": max((st["max_decision_ms"] for st in stats), default=0),
                "errors": sum(st["errors"] for st in stats)}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "validation"), default="development")
    parser.add_argument("--candidate", type=Path, default=ROOT/"strategy.py")
    parser.add_argument("--candidate-snapshot", type=Path,
                        help="Load exact source/model/weights/params saved by StrategyFactory.save_snapshot")
    parser.add_argument("--candidate-model", type=Path, default=ROOT/"opponent_model.py")
    parser.add_argument("--incumbent-ref", default=FROZEN_REF)
    parser.add_argument("--params", type=Path)
    parser.add_argument("--seeds", nargs="+")
    parser.add_argument("--deals", type=int, default=300)
    parser.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    parser.add_argument("--output", type=Path, default=ROOT/"results/strength-development.json")
    parser.add_argument("--resume", type=Path, help="Reuse completed pairs with identical candidate, cards, and shared evaluator")
    parser.add_argument("--resume-note", default="")
    parser.add_argument("--incumbent-cache", type=Path,
                        help="Reuse verified incumbent results on matching cards/profiles to avoid repeated baseline games")
    parser.add_argument("--allow-provenance-change", action="store_true",
                        help="Development only: explicitly acknowledge changed/missing tool or probe fingerprints")
    parser.add_argument("--provenance-note", default="")
    args = parser.parse_args()
    if args.deals < 2:
        parser.error("At least two deals are needed")
    if len(set(args.opponents)) != len(args.opponents):
        parser.error("Opponent profiles must be unique")
    external = {name: (str(Path(name).resolve()), Path(name).read_bytes())
                for name in args.opponents if Path(name).is_file()}
    unknown = set(args.opponents)-set(DEFAULT_OPPONENTS)-set(BUILTINS)-set(external)
    if unknown:
        parser.error(f"Unknown opponents: {sorted(unknown)}")
    seeds = args.seeds or (DEVELOPMENT_SEEDS if args.phase == "development" else VALIDATION_SEEDS)
    if len(set(seeds)) != len(seeds):
        parser.error("Seeds must be unique")
    if args.phase != "validation" and set(seeds).intersection(VALIDATION_SEEDS):
        parser.error("Reserved holdout seeds require --phase validation")
    override = json.loads(args.params.read_text()) if args.params else {}
    parameters = override.get("parameters", override)
    candidate = (factory_from_snapshot(args.candidate_snapshot, parameters) if args.candidate_snapshot else
                 factory_from_candidate(args.candidate, parameters, args.candidate_model))
    incumbent = factory_from_commit(args.incumbent_ref)
    files = ("strategy.py", "opponent_model.py", "policy.json", "equity_engine.py", "preflop_model.npz",
             "tools/style_opponents.py", "tools/fast_range_opponent.py", "tools/strength_opponents.py", "tools/evaluate_strength.py",
             "game.py", "evaluator.py", "tools/benchmark.py")
    files += tuple(name for name in ("equity_network.py", "equity_network.npz", "river_solver.py", "river_policy.npz") if (ROOT/name).is_file())
    fingerprint = lambda: {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}
    fingerprints = fingerprint()
    matches, improvements = [], []
    report = {"phase": args.phase, "seeds": list(seeds), "incumbent_ref": args.incumbent_ref,
              "candidate": candidate.manifest(), "incumbent": incumbent.manifest(),
              "external_opponent_sha256": {name: hashlib.sha256(source).hexdigest()
                                            for name, (_, source) in external.items()},
              "candidate_parameters": candidate.parameters, "incumbent_parameters": incumbent.parameters,
              "source_sha256_at_start": fingerprints,
              "timing_scope": "Direct calls; excludes HTTP and has no time-bank enforcement",
              "uncertainty_scope": "Synthetic profiles; shared-card clustered CI, local block bootstrap conditional on observed seed trajectories, and whole-seed Student-t CI for persistent adaptation. Few seeds limit precision. No claim against unknown tournament submissions.",
              "matches": matches, "paired_improvements": improvements}
    def check_provenance(previous, label):
        inherited = previous.get("provenance_overrides", []) + previous.get("inherited_provenance_overrides", [])
        if inherited:
            if args.phase == "validation" or not args.allow_provenance_change or not args.provenance_note:
                parser.error(f"{label} contains inherited provenance exceptions; strict validation requires fresh, compatible results")
            report.setdefault("inherited_provenance_overrides", []).append({
                "source": label, "prior_exceptions": inherited, "note": args.provenance_note})
        old = previous["source_sha256_at_start"]
        required = ("equity_engine.py", "preflop_model.npz", "game.py", "evaluator.py",
                    "tools/benchmark.py", "tools/evaluate_strength.py", "tools/style_opponents.py",
                    "tools/fast_range_opponent.py", "tools/strength_opponents.py")
        gaps = {name: {"previous": old.get(name), "current": fingerprints[name]}
                for name in required if old.get(name) != fingerprints[name]}
        changed_rules = [name for name in required[:4] if old.get(name) is not None
                         and old[name] != fingerprints[name]]
        if changed_rules:
            parser.error(f"{label} shared rules/equity mismatch: {changed_rules}")
        if gaps:
            if args.phase == "validation" or not args.allow_provenance_change or not args.provenance_note:
                parser.error(f"{label} provenance differs: {sorted(gaps)}; development reuse needs --allow-provenance-change and --provenance-note")
            report.setdefault("provenance_overrides", []).append({"source": label, "files": gaps,
                                                                  "note": args.provenance_note})
    previous_seconds = 0.
    if args.resume:
        previous = json.loads(args.resume.read_text())
        for field in ("phase", "seeds", "incumbent_ref", "candidate", "incumbent", "external_opponent_sha256"):
            old, new = previous.get(field), report[field]
            if field in ("candidate", "incumbent"):
                old, new = comparable_manifest(old), comparable_manifest(new)
            if old != new:
                parser.error(f"Resume mismatch: {field}")
        check_provenance(previous, "Resume")
        if len(previous["matches"]) != 2*len(previous["paired_improvements"]):
            parser.error("Resume report contains incomplete pairs")
        for match in previous["matches"]:
            if match["deals"] != args.deals or match["opponent"] not in args.opponents or match["seed"] not in seeds:
                parser.error("Resume cards/profile/deal-count mismatch")
        matches.extend(previous["matches"])
        improvements.extend(previous["paired_improvements"])
        previous_seconds = previous["elapsed_seconds"]
        report["resume_history"] = previous.get("resume_history", []) + [{
            "path": str(args.resume), "completed_pairs": len(improvements), "note": args.resume_note,
            "prior_source_sha256_at_start": previous["source_sha256_at_start"],
            "prior_source_sha256_at_end": previous.get("source_sha256_at_end", {})}]
        print(f"Resuming {len(improvements)} completed pairs", flush=True)
    completed = {(m["opponent"], m["seed"]) for m in improvements}
    incumbent_cache = {}
    if args.incumbent_cache:
        cached = json.loads(args.incumbent_cache.read_text())
        if comparable_manifest(cached["incumbent"]) != comparable_manifest(incumbent.manifest()) or cached["incumbent_ref"] != args.incumbent_ref:
            parser.error("Incumbent cache source/parameters mismatch")
        check_provenance(cached, "Incumbent cache")
        if cached.get("external_opponent_sha256", {}) != report["external_opponent_sha256"]:
            parser.error("Incumbent cache external opponent mismatch")
        incumbent_cache = {(m["opponent"], m["seed"], m["deals"]): m for m in cached["matches"]
                           if m["strategy"] == "incumbent"}
        report["incumbent_cache"] = {"path": str(args.incumbent_cache),
            "sha256": hashlib.sha256(args.incumbent_cache.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    resumed_hands = 2*sum(m["deals"] for m in matches)
    invocation_simulated_hands = invocation_cached_hands = 0
    def persist():
        report.update(elapsed_seconds=previous_seconds+time.perf_counter()-start,
                      total_hands=2*sum(m["deals"] for m in matches), resumed_hands=resumed_hands,
                      simulated_hands=2*sum(m["deals"] for m in matches if not m.get("reused_from_incumbent_cache")),
                      reused_incumbent_hands=2*sum(m["deals"] for m in matches if m.get("reused_from_incumbent_cache")),
                      newly_simulated_hands_this_invocation=invocation_simulated_hands,
                      reused_incumbent_hands_this_invocation=invocation_cached_hands,
                      candidate_aggregate=aggregate([m for m in matches if m["strategy"] == "candidate"]),
                      incumbent_aggregate=aggregate([m for m in matches if m["strategy"] == "incumbent"]),
                      improvement_aggregate=aggregate(improvements), clustered_improvement=clustered_summary(improvements),
                      clustered_improvement_external=clustered_summary(improvements, ("incumbent",)),
                      action_summary=action_summary(matches), source_sha256_at_end=fingerprint())
        report["runtime_source_unchanged"] = all(report["source_sha256_at_end"][n] == fingerprints[n] for n in files[:5])
        report["shared_runtime_unchanged"] = all(report["source_sha256_at_end"][n] == fingerprints[n]
                                                for n in ("equity_engine.py", "preflop_model.npz", "game.py", "evaluator.py"))
        temporary = args.output.with_suffix(args.output.suffix+".tmp")
        temporary.write_text(json.dumps(report, indent=2)+"\n")
        temporary.replace(args.output)
    for opponent in args.opponents:
        for seed in seeds:
            if (opponent, seed) in completed:
                continue
            current = []
            for factory in (incumbent, candidate):
                cache_key = (opponent, seed, args.deals)
                if factory is incumbent and cache_key in incumbent_cache:
                    match = dict(incumbent_cache[cache_key], reused_from_incumbent_cache=True)
                    invocation_cached_hands += 2*args.deals
                else:
                    match = run_match(factory, opponent, incumbent, seed, args.deals, external)
                    invocation_simulated_hands += 2*args.deals
                current.append(match)
                matches.append(match)
                suffix = " (cached)" if match.get("reused_from_incumbent_cache") else ""
                print(f"{opponent} {seed} {factory.name}: {match['mean_bb_per_deal']:+.4f} CI={match['ci95']}{suffix}", flush=True)
            differences = [a-b for a, b in zip(current[1]["paired_bb"], current[0]["paired_bb"])]
            improvements.append({"opponent": opponent, "seed": seed, **summarize(differences), "paired_bb": differences})
            persist()
    persist()
    print(json.dumps({key: report[key] for key in ("clustered_improvement", "candidate_aggregate", "improvement_aggregate")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
