#!/usr/bin/env python3
"""Serve a frozen Git strategy or an independent style through event HTTP routes.

Examples (run from repository root):
  python tools/serve_strategy.py --revision 50b4682 --port 5009
  python tools/serve_strategy.py --style tight_passive --port 5010

Only the strategy callbacks are replaced. The organizer template retains its
request parsing, token checks, fresh-half memory reset, and result delivery.
Frozen strategies are compiled in their own namespace with committed parameters;
this helper does not import the current strategy module.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from equity_engine import warmup
from evaluate_adaptation import StrategyFactory, committed
from style_opponents import STYLES, StyleOpponent


def configured_template(*, revision: str | None = None, style: str | None = None):
    if (revision is None) == (style is None):
        raise ValueError("Specify exactly one revision or style")
    if revision is not None:
        full_ref = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
            cwd=ROOT, text=True).strip()
        parameters = json.loads(committed(full_ref, "policy.json"))["parameters"]
        factory = StrategyFactory(committed(full_ref, "strategy.py"), parameters,
                                  "frozen_http", ROOT / "strategy.py")
        bot = factory.create(0)
        name = f"Frozen-{full_ref[:12]}"
    else:
        bot = StyleOpponent(style)
        name = f"Style-{style}"
    warmup()
    path = ROOT / "baseline-bots" / "baseline_random.py"
    spec = importlib.util.spec_from_file_location("strategy_http_template", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load organizer template: {path}")
    template = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = template
    spec.loader.exec_module(template)
    template.new_half = bot.new_half
    template.choose_action = bot.choose_action
    template.on_hand_end = bot.on_hand_end
    template.PARTICIPANT = "Development evaluation probe"
    template.AGENT_NAME = name
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--revision", help="Git commit or ref with strategy.py and policy.json")
    source.add_argument("--style", choices=STYLES, help="Independent observation-only style policy")
    parser.add_argument("--port", type=int, default=5009)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    template = configured_template(revision=args.revision, style=args.style)
    import uvicorn
    uvicorn.run(template.app, host="0.0.0.0", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
