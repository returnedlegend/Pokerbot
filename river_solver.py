"""Optional lookup policy for an offline-solved, abstract river game.

This is a small CFR+ model, not a solved strategy for five-card short-deck PLO.
Its private types are independent strength quantiles. A supplied opponent root
histogram selects a precomputed approximate range-conditioned endgame; the hero
root range remains uniform. Real hands have card removal, board-dependent ties,
and correlated ranges that this abstraction does not represent. Keep it experimental until an
independent match evaluation demonstrates an improvement.

The training implementation and reproducible command are in
``tools/train_river_policy.py``. Runtime needs only NumPy and a small NPZ file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

from game import Action, Observation, Street


@lru_cache(maxsize=1)
def _model() -> dict[str, np.ndarray]:
    with np.load(Path(__file__).with_name("river_policy.npz"), allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _prior_index(model: dict[str, np.ndarray], opponent_prior: Sequence[float] | None) -> int | None:
    if opponent_prior is None:
        return 0
    prior = np.asarray(opponent_prior, dtype=np.float64)
    buckets = model["strategy"].shape[1]
    if (prior.shape != (buckets,) or not np.all(np.isfinite(prior))
            or np.any(prior < 0) or prior.sum() <= 0):
        return None
    # CDF distance is stable with modest particle counts and respects the order
    # of strength buckets. No sample-size confidence is inferred here: callers
    # should hedge low-effective-sample-size histograms toward uniform first.
    cdf = np.cumsum(prior / prior.sum())
    template_cdf = np.cumsum(model["prior_templates"], axis=1)
    return int(np.argmin(np.sum((template_cdf - cdf) ** 2, axis=1)))


def policy(obs: Observation, equity_uniform: float,
           opponent_prior: Sequence[float] | None = None) -> dict[Action, float] | None:
    """Return legal mixed-action probabilities, or None for an unmappable state.

    ``equity_uniform`` must be river win-plus-half-tie equity against a uniform
    opponent, NOT equity conditional on the opponent's betting range. The
    strategy already conditions on the river action history within its toy game.
    The caller can blend this with an independently evaluated opponent model.

    ``opponent_prior`` optionally supplies 12 strength-bucket masses at the
    START of the river, before ANY river actions. It must not already condition
    on river bets, which the policy handles itself. A nearest-CDF template is
    selected; the caller should hedge uncertain histograms toward uniform. None
    preserves the original uniform-range behavior.
    """
    if obs.street != Street.RIVER or len(obs.board) != 5:
        return None
    if not np.isfinite(equity_uniform) or not 0.0 <= equity_uniform <= 1.0:
        return None
    river = [a for a in obs.action_history if a.street == Street.RIVER]
    investments = [0, 0]
    for action in river:
        investments[0 if action.is_me else 1] += action.amount
    root_pot = obs.pot - sum(investments)
    root_stacks = (obs.my_stack + investments[0], obs.opp_stack + investments[1])
    if root_pot <= 0 or min(root_stacks) <= 0:
        return None
    # The event starts each hand with equal stacks, so river roots should match.
    # Unequal roots would require an additional effective-stack abstraction.
    if root_stacks[0] != root_stacks[1]:
        return None
    try:
        model = _model()
    except (OSError, ValueError):
        return None
    spr = root_stacks[0] / root_pot
    grid = model["spr_grid"]
    grid_index = int(np.argmin(np.abs(np.log(grid / spr))))
    hero = 1 if obs.is_button else 0  # OOP is the first river actor.
    if int(model.get("format_version", 1)) >= 2:
        template = _prior_index(model, opponent_prior)
        if template is None:
            return None
        node = int(model["roots"][grid_index, hero, template])
    else:
        # Keep old frozen model snapshots usable with their original behavior.
        if opponent_prior is not None:
            return None
        node = int(model["roots"][grid_index])
    for action in river:
        recorded_actor = hero if action.is_me else 1 - hero
        if int(model["actor"][node]) != recorded_actor:
            return None
        node = int(model["children"][node, int(action.action)])
        if node < 0:
            return None
    if int(model["actor"][node]) != hero:
        return None
    buckets = model["strategy"].shape[1]
    bucket = min(buckets - 1, int(equity_uniform * buckets))
    probabilities = np.array(model["strategy"][node, bucket], dtype=np.float64)
    legal = set(obs.legal_actions)
    for action in Action:
        if action not in legal:
            probabilities[int(action)] = 0.0
    # Equal all-in sizes are strategically one action; avoid changing behavior
    # simply because the organizer spells the same raise HALF or POT.
    if (Action.RAISE_HALF in legal and Action.RAISE_POT in legal
            and obs.raise_to[Action.RAISE_HALF] == obs.raise_to[Action.RAISE_POT]):
        probabilities[Action.RAISE_HALF] += probabilities[Action.RAISE_POT]
        probabilities[Action.RAISE_POT] = 0.0
    total = float(probabilities.sum())
    if total <= 0.0:
        return None
    return {action: float(probabilities[int(action)] / total)
            for action in obs.legal_actions if probabilities[int(action)] > 0.0}


def choose_action(obs: Observation, equity_uniform: float,
                  opponent_prior: Sequence[float] | None = None) -> Action | None:
    """Sample a policy using visible-card hashing, stable under duplicate deals.

    Neither seat number nor the match seed enters the draw. Sampling is therefore
    repeatable if the same visible information is replayed in the opposite half.
    """
    probabilities = policy(obs, equity_uniform, opponent_prior)
    if probabilities is None:
        return None
    value = 2166136261
    for token in (*sorted(obs.hole_cards), 37, *obs.board, 41, obs.pot, obs.to_call):
        value = ((value ^ int(token)) * 16777619) & 0xFFFFFFFF
    for record in obs.action_history:
        if record.street == Street.RIVER:
            for token in (int(record.action), record.amount, int(record.is_me)):
                value = ((value ^ token) * 16777619) & 0xFFFFFFFF
    draw = (value + 0.5) / 4294967296.0
    cumulative = 0.0
    for action, probability in probabilities.items():
        cumulative += probability
        if draw < cumulative:
            return action
    return next(reversed(probabilities))
