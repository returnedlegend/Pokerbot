"""Independent, observation-only style probes for adaptation evaluation.

These are deliberately small equity policies, not other teams' submissions.
They share the exact event card sampler but do not import production strategy,
its learned preflop table, or its opponent model. Their profiles are fixed
before candidate validation. Uniform random legal hidden cards are sampled;
actual opponent cards and future game cards are never available here.
"""
from __future__ import annotations

import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from equity_engine import sample_equities
from game import Action

STYLES = ("tight_passive", "loose_passive", "selective_aggressive",
          "loose_aggressive", "changing", "changing_reverse")
RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


class StyleOpponent:
    def __init__(self, style: str):
        if style not in STYLES:
            raise ValueError(style)
        self.style = style

    def new_half(self, config, memory):
        memory.update(rng=random.Random(98341), hand=-1, cache={},
                      switch_after=max(40, min(400, config.num_hands // 2)))

    def on_hand_end(self, result, memory):
        pass

    def effective_style(self, obs, memory):
        if self.style == "changing":
            return "tight_passive" if obs.hand_index < memory["switch_after"] else "loose_aggressive"
        if self.style == "changing_reverse":
            return "loose_aggressive" if obs.hand_index < memory["switch_after"] else "tight_passive"
        return self.style

    def choose_action(self, obs, memory):
        rng = memory["rng"]
        if memory["hand"] != obs.hand_index:
            memory["hand"] = obs.hand_index
            memory["cache"] = {}
        board_key = tuple(obs.hole_cards) + tuple(obs.board)
        if board_key not in memory["cache"]:
            memory["cache"][board_key] = sample_equities(
                obs.hole_cards, obs.board, n=128,
                seed=rng.randrange(1, 2**31), include_history=False)
        particles = memory["cache"][board_key]
        wins = particles["wins"]
        style = self.effective_style(obs, memory)
        prior_raises = sum(not a.is_me and a.action in RAISES for a in obs.action_history)
        # Passive/loose probes make broad calls, while the selective opponent
        # independently narrows its perceived range after observing raises.
        if style == "selective_aggressive" and prior_raises:
            scores = particles["villain_scores"] if obs.board else particles["villain_preflop"]
            keep = max(32, int(len(wins) * max(.35, .80-.12*prior_raises)))
            equity = float(wins[np.argsort(scores, kind="stable")[-keep:]].mean())
        else:
            equity = float(wins.mean())
        price = obs.to_call / max(1, obs.pot + obs.to_call)
        premium = {"tight_passive": .17, "loose_passive": -.10,
                   "selective_aggressive": .065, "loose_aggressive": -.025}[style]
        # No probe folds a free check, and blind-completion prices are handled
        # broadly enough that preflop does not determine the entire matchup.
        if obs.to_call > obs.big_blind and equity < max(.07, price+premium):
            if Action.FOLD in obs.legal_actions:
                return Action.FOLD
        threshold = {"tight_passive": .84, "loose_passive": .85,
                     "selective_aggressive": .63, "loose_aggressive": .53}[style]
        threshold += .025*obs.raises_this_street
        raise_action = Action.RAISE_HALF if style == "loose_passive" else Action.RAISE_POT
        if equity > threshold and raise_action in obs.legal_actions:
            return raise_action
        if style == "loose_aggressive":
            # Includes unprompted stabs and occasional re-raises with weaker
            # holdings, rather than being an always-pot caricature.
            bluff_rate = .23 if not obs.to_call else .10
            if rng.random() < bluff_rate and obs.raises_this_street < 3:
                if Action.RAISE_HALF in obs.legal_actions:
                    return Action.RAISE_HALF
        return Action.CHECK_CALL


def main():
    """Serve one fixed independent style through organizer HTTP plumbing."""
    import argparse
    import importlib.util
    import os
    import uvicorn
    from equity_engine import warmup
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", choices=STYLES, default="loose_aggressive")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5009")))
    args = parser.parse_args()
    warmup()
    spec = importlib.util.spec_from_file_location("style_probe_http", ROOT/"baseline-bots/baseline_random.py")
    template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(template)
    opponent = StyleOpponent(args.style)
    template.new_half = opponent.new_half
    template.choose_action = opponent.choose_action
    template.on_hand_end = opponent.on_hand_end
    template.PARTICIPANT = "Independent style probe"
    template.AGENT_NAME = args.style
    uvicorn.run(template.app, host="0.0.0.0", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
