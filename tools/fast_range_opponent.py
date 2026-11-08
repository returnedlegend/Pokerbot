#!/usr/bin/env python3
"""Independent, fast equity/range stress opponent (development tools only).

Shares the exact card evaluator/Monte Carlo sampler, but none of ShortStack's
policy, fitted preflop model, parameters, or adaptation logic. Unlike the slow
Python mc_range probe, this can be checked under the official HTTP time bank.

Run: PORT=5009 python tools/fast_range_opponent.py
Benchmark: python tools/benchmark.py --opponents tools/fast_range_opponent.py
"""
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from equity_engine import sample_equities, warmup
from game import Action

warmup()


def new_half(config, memory):
    memory['rng'] = random.Random(411)


def on_hand_end(result, memory):
    pass


def choose_action(obs, memory):
    sample = sample_equities(obs.hole_cards, obs.board, n=192,
                             seed=memory['rng'].randrange(1, 2**31),
                             include_history=False)
    wins = sample['wins']
    raises = sum(not a.is_me and a.action in (Action.RAISE_HALF, Action.RAISE_POT)
                 for a in obs.action_history)
    if raises:
        # A hard broad range cut differs from ShortStack's soft likelihood
        # weighting. Only revealed-board strength enters range selection.
        scores = sample['villain_scores'] if obs.board else sample['villain_preflop']
        keep = max(48, int(len(wins)*max(.35, .8-.12*raises)))
        selected = np.argsort(scores, kind='stable')[-keep:]
        equity = float(wins[selected].mean())
    else:
        equity = float(wins.mean())
    price = obs.to_call / max(1, obs.pot+obs.to_call)
    premium = .06 if obs.to_call > obs.big_blind else 0.
    if obs.to_call and equity < price+premium and Action.FOLD in obs.legal_actions:
        return Action.FOLD
    if equity > .65+.015*obs.raises_this_street:
        if Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
    return Action.CHECK_CALL


def main():
    import importlib.util
    import os
    import uvicorn
    spec = importlib.util.spec_from_file_location('fast_range_http', ROOT/'baseline-bots'/'baseline_random.py')
    template = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(template)
    template.new_half = new_half
    template.choose_action = choose_action
    template.on_hand_end = on_hand_end
    template.PARTICIPANT = 'Independent development probe'
    template.AGENT_NAME = 'FastRange'
    uvicorn.run(template.app, host='0.0.0.0', port=int(os.environ.get('PORT','5009')), access_log=False)


if __name__ == '__main__':
    main()
