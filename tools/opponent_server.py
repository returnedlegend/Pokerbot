#!/usr/bin/env python3
"""Serve benchmark opponents through the unchanged official HTTP plumbing.

Example: python tools/opponent_server.py --opponent mc_range --port 5009
Then use judge_engine.py for the production time-bank/deployment check.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark import BUILTINS, Opponent
from evaluator import warmup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", choices=BUILTINS, default="mc_loose")
    parser.add_argument("--port", type=int, default=5009)
    args = parser.parse_args()
    # Reuse official server endpoints/token checks; replace only the strategy
    # functions that those endpoints resolve in the template module globals.
    path = ROOT / "baseline-bots" / "baseline_random.py"
    spec = importlib.util.spec_from_file_location("benchmark_http_opponent", path)
    template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(template)
    bot = Opponent(args.opponent)
    template.new_half = bot.new_half
    template.choose_action = bot.choose_action
    template.on_hand_end = bot.on_hand_end
    template.AGENT_NAME = f"Probe-{args.opponent}"
    warmup()
    import uvicorn
    uvicorn.run(template.app, host="0.0.0.0", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
