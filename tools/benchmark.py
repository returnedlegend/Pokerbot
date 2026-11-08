#!/usr/bin/env python3
"""Observation-only, duplicate-deal strategy benchmark (no HTTP overhead).

Run from the repository: python tools/benchmark.py --deals 1000 --seeds 11 29
Use --opponents path/to/other_agent.py to test another implementation.
This measures Python strategy time, not tournament HTTP time-bank usage; use
judge_engine.py for the final deployment/time-bank check. No time bank is
enforced here: strategies are evaluated completely, with timing reported.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
from itertools import combinations

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator import get_table, omaha_score, warmup
from game import Action, AgentConfig, BIG_BLIND, Game, Street, button_for_deal, make_deal


class Opponent:
    """Simple probes plus independent Monte Carlo opponents for holdout tests."""

    def __init__(self, name: str):
        self.name = name

    def new_half(self, config, memory):
        memory["rng"] = random.Random(0)

    def on_hand_end(self, result, memory):
        pass

    def choose_action(self, obs, memory):
        legal = obs.legal_actions
        rng = memory["rng"]
        if self.name == "random":
            return rng.choice(legal)
        if self.name == "call":
            return Action.CHECK_CALL
        if self.name == "checkfold":
            return Action.FOLD if obs.to_call else Action.CHECK_CALL
        if self.name in ("half", "pot"):
            act = Action.RAISE_HALF if self.name == "half" else Action.RAISE_POT
            return act if act in legal else Action.CHECK_CALL

        # Independent equity model, uniformly sampled hidden cards with no
        # knowledge of actual opponent holdings or future public cards.
        known = set(obs.hole_cards + obs.board)
        deck = [c for c in range(36) if c not in known]
        wins = 0.0
        samples = 48 if self.name in ("mc_tight", "mc_range") else 32
        outcomes = []
        board_masks = [sum(1 << c for c in b) for b in combinations(obs.board, 3)]
        table = get_table()
        for _ in range(samples):
            draw = rng.sample(deck, 10 - len(obs.board))
            board = obs.board + tuple(draw[5:])
            mine = omaha_score(obs.hole_cards, board)
            theirs = omaha_score(draw[:5], board)
            outcome = (mine > theirs) + .5 * (mine == theirs)
            wins += outcome
            if self.name == "mc_range":
                # Range conditioning uses only already visible board cards.
                # In particular, future sampled board cards never determine
                # which opponent hands count as plausible bets on this street.
                if len(obs.board) >= 3:
                    strength = max(table[((1 << a) | (1 << b)) | bm]
                                   for a, b in combinations(draw[:5], 2) for bm in board_masks)
                else:
                    ranks = [c >> 2 for c in draw[:5]]
                    strength = sum(ranks) + 5*(5-len(set(ranks)))
                outcomes.append((strength, outcome))
        if self.name == "mc_range":
            aggression = sum(not a.is_me and a.action in (Action.RAISE_HALF, Action.RAISE_POT)
                             for a in obs.action_history)
            if aggression:
                outcomes.sort(key=lambda x: x[0], reverse=True)
                # One bet is a broad range; multiple bets imply more strength.
                keep = max(12, int(samples * max(.35, .8-.15*aggression)))
                equity = sum(x[1] for x in outcomes[:keep])/keep
            else:
                equity = wins / samples
        else:
            equity = wins / samples
        price = obs.to_call / max(1, obs.pot + obs.to_call)
        # Tight opponents demand a premium facing raises; loose opponents
        # protect more of their range. Neither shares our production strategy.
        tight = self.name in ("mc_tight", "mc_range")
        premium = (.13 if tight else .035) if obs.to_call > obs.big_blind else 0
        if obs.to_call and equity < price + premium and Action.FOLD in legal:
            return Action.FOLD
        threshold = (.69 if tight else .58) + .025 * obs.raises_this_street
        if equity > threshold:
            act = Action.RAISE_POT if tight else Action.RAISE_HALF
            if act in legal:
                return act
        # Occasional independent positional bluffs keep these from being
        # merely transparent value bots.
        if not tight and not obs.to_call and obs.is_button and rng.random() < .12:
            if Action.RAISE_HALF in legal:
                return Action.RAISE_HALF
        return Action.CHECK_CALL


BUILTINS = ("random", "call", "half", "pot", "checkfold", "mc_loose", "mc_tight", "mc_range")


def load_bot(source: str, unique: str):
    if source in BUILTINS:
        return Opponent(source)
    path = Path(source).resolve()
    # A new Python module per half prevents ordinary Python global state from
    # being carried across duplicate halves. Bot-owned imports may still cache
    # modules: final tournament validation must use independent processes.
    spec = importlib.util.spec_from_file_location(unique, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique] = module
    spec.loader.exec_module(module)
    return module


def summarize(values):
    n = len(values)
    mean = statistics.mean(values) if n else 0.0
    se = statistics.stdev(values) / math.sqrt(n) if n > 1 else 0.0
    lo, hi = mean - 1.96 * se, mean + 1.96 * se
    return {"deals": n, "mean_bb_per_deal": mean, "ci95": [lo, hi],
            "verdict": "WIN" if lo > 0 else "LOSS" if hi < 0 else "CHOP"}


def run_match(agent_source: str, opponent_source: str, seed, deals=1000):
    """Play two fresh-memory halves, preserving each deal and swapping seats.

    Errors and illegal actions raise immediately, so a benchmark cannot hide
    crashes behind the tournament's check/fold fallback. Both bots receive
    only Game.observation and Game.hand_result; game/deal never enter memory.
    """
    if deals < 2:
        raise ValueError("Use at least two deals for a confidence interval")
    halves, timing = [], []
    config = AgentConfig(num_hands=deals)
    started = time.perf_counter()
    for half in range(2):
        bots = [load_bot(agent_source, f"benchmark_agent_{half}"),
                load_bot(opponent_source, f"benchmark_opponent_{half}")]
        memories = [{}, {}]
        stats = [{"calls": 0, "strategy_seconds": 0.0, "max_decision_ms": 0.0,
                  "errors": 0, "action_counts": {a.name: 0 for a in Action}}
                 for _ in bots]
        for bot, memory, stat in zip(bots, memories, stats):
            begin = time.perf_counter()
            bot.new_half(config, memory)
            stat["strategy_seconds"] += time.perf_counter() - begin
        nets = []
        for d in range(deals):
            game = Game(make_deal(seed, d), button_for_deal(d), hand_index=d)
            while not game.is_terminal:
                seat = game.to_act
                who = seat ^ half
                obs = game.observation(seat)
                begin = time.perf_counter()
                action = bots[who].choose_action(obs, memories[who])
                elapsed = time.perf_counter() - begin
                stats[who]["calls"] += 1
                stats[who]["strategy_seconds"] += elapsed
                stats[who]["max_decision_ms"] = max(stats[who]["max_decision_ms"], 1000*elapsed)
                if not isinstance(action, Action) or action not in obs.legal_actions:
                    raise RuntimeError(f"Illegal action {action!r}: half={half+1}, deal={d}, bot={who}")
                stats[who]["action_counts"][action.name] += 1
                game.apply(action)
            nets.append(game.result().nets[half])
            for who, bot in enumerate(bots):
                begin = time.perf_counter()
                bot.on_hand_end(game.hand_result(who ^ half), memories[who])
                stats[who]["strategy_seconds"] += time.perf_counter() - begin
        halves.append(nets)
        timing.append({"agent": stats[0], "opponent": stats[1]})
    paired = [(a+b)/BIG_BLIND for a, b in zip(*halves)]
    return {"opponent": opponent_source, "seed": seed, **summarize(paired),
            "paired_bb": paired, "half_stats": timing,
            "elapsed_seconds": time.perf_counter() - started,
            "timing_scope": "Direct strategy calls; excludes HTTP and JSON; no time-bank enforcement"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default=str(ROOT / "agent.py"))
    parser.add_argument("--opponents", nargs="+", default=list(BUILTINS))
    parser.add_argument("--seeds", nargs="+", default=["11", "29", "73"])
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--params", type=Path, help="Override strategy.PARAMS with a parameter JSON for this run")
    args = parser.parse_args()
    parameter_override = None
    if args.params:
        parameter_override = json.loads(args.params.read_text())
        parameter_override = parameter_override.get("parameters", parameter_override)
        strategy = importlib.import_module("strategy")
        strategy.PARAMS.update(parameter_override)
    warmup()
    matches = []
    for opponent in args.opponents:
        for seed in args.seeds:
            match = run_match(args.agent, opponent, seed, args.deals)
            matches.append(match)
            lo, hi = match["ci95"]
            times = [h["agent"]["strategy_seconds"] for h in match["half_stats"]]
            print(f"{opponent} seed={seed}: {match['mean_bb_per_deal']:+.3f} bb/deal "
                  f"95%CI [{lo:+.3f}, {hi:+.3f}] {match['verdict']}; "
                  f"strategy seconds/half={times[0]:.3f},{times[1]:.3f}", flush=True)
    aggregate = {o: summarize([v for m in matches if m["opponent"] == o for v in m["paired_bb"]])
                 for o in args.opponents}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        fingerprints = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (ROOT / "agent.py", ROOT / "strategy.py", ROOT / "equity_engine.py",
                                  ROOT / "preflop_model.npz", Path(__file__).resolve()) if p.exists()}
        loaded_strategy = sys.modules.get("strategy")
        effective_parameters = dict(loaded_strategy.PARAMS) if loaded_strategy and hasattr(loaded_strategy, "PARAMS") else None
        args.json.write_text(json.dumps({"matches": matches, "aggregate": aggregate,
                                        "parameter_override": parameter_override,
                                        "effective_parameters": effective_parameters,
                                        "source_sha256": fingerprints}, indent=2) + "\n")
    print(json.dumps({"aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
