"""Train a small independent-quantile river abstraction using vanilla CFR+.

Run from the repository root, for example::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
      .venv/bin/python tools/train_river_policy.py --iterations 20000

No external solver source is copied. This implements counterfactual regret
minimization directly; CFR+ uses nonnegative cumulative regrets, alternating
updates, and a linearly weighted average strategy after a delay.

Players each receive an independent integer strength bucket from an explicit
root range. Higher bucket wins; equal buckets tie. They bet with the event's
integer half-pot/pot actions and four-raise cap. Payoffs include half the initial
pot plus the opponent's future investment for wins, or minus half the initial
pot and own investment for losses. This artificial game is NOT real PLO, and
its exploitability numbers apply only to this game.

Range-conditioned endgame solving uses public-history-conditioned ranges at
the start of a subgame; see Brown and Sandholm, "Safe and Nested Subgame
Solving for Imperfect-Information Games" (NeurIPS 2017),
https://arxiv.org/abs/1705.02955. This implementation
does not have that paper's safety guarantees: it replaces real card-dependent
joint ranges with independent bucket priors, assumes a uniform hero root
range, and selects a nearby precomputed opponent template. These choices
make it a cheap, explicitly approximate experiment rather than real PLO GTO.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from game import Action


@dataclass
class Tree:
    pot: int
    stack: int
    buckets: int
    actor: np.ndarray
    children: np.ndarray
    legal: np.ndarray
    utility: np.ndarray
    bets: np.ndarray
    targets: np.ndarray


def build_tree(pot: int = 8, stack: int = 8, buckets: int = 12) -> Tree:
    """Build a topologically ordered tree, merging identical-size raises."""
    if pot < 2 or pot % 2 or stack < 1 or buckets < 2:
        raise ValueError("need even positive pot, positive stack and >=2 buckets")
    actors, children, legals, utilities, all_bets, targets = [], [], [], [], [], []
    strengths = np.arange(buckets)
    showdown = np.sign(strengths[:, None] - strengths[None, :])

    def add(bets, acted, actor, raises, increment, winner=None, terminal=False):
        index = len(actors)
        actors.append(-1 if terminal else actor)
        children.append([-1] * 4)
        legals.append([False] * 4)
        all_bets.append(bets)
        targets.append([-1] * 4)
        if terminal:
            if winner is None:
                utilities.append(showdown * (pot / 2 + bets[0]))
            else:
                payoff = pot / 2 + bets[1] if winner == 0 else -pot / 2 - bets[0]
                utilities.append(np.full((buckets, buckets), payoff))
            return index
        utilities.append(np.zeros((buckets, buckets)))
        opp = 1 - actor
        call = max(0, bets[opp] - bets[actor])
        remaining = stack - bets[actor]
        actions = [1] if call == 0 else [0, 1]
        if raises < 4 and remaining > call and stack > bets[opp]:
            actions += [2, 3]
        duplicate = {}
        for action in actions:
            if action == 0:
                legals[index][action] = True
                children[index][action] = add(bets, acted, opp, raises, increment,
                                              winner=opp, terminal=True)
                continue
            next_bets = list(bets)
            next_acted = list(acted)
            next_raises = raises
            next_increment = increment
            if action == 1:
                next_bets[actor] += min(call, remaining)
            else:
                current_pot = pot + sum(bets)
                maximum = bets[opp] + current_pot + call
                target = bets[opp] + ((current_pot + call) // 2 if action == 2
                                      else current_pot + call)
                target = min(max(target, bets[opp] + max(increment, 2)),
                             maximum, stack)
                targets[index][action] = target
                if target in duplicate:
                    children[index][action] = children[index][duplicate[target]]
                    continue
                duplicate[target] = action
                next_bets[actor] = target
                next_increment = target - bets[opp]
                next_raises += 1
                next_acted[opp] = False
            legals[index][action] = True
            next_acted[actor] = True
            terminal = all(next_acted) and next_bets[0] == next_bets[1]
            children[index][action] = add(tuple(next_bets), tuple(next_acted), opp,
                                          next_raises, next_increment, terminal=terminal)
        return index

    add((0, 0), (False, False), 0, 0, 2)
    return Tree(pot, stack, buckets, np.asarray(actors, dtype=np.int8),
                np.asarray(children, dtype=np.int32), np.asarray(legals),
                np.asarray(utilities, dtype=np.float64), np.asarray(all_bets),
                np.asarray(targets))


@njit(cache=True)
def _strategy(regrets, legal):
    strategy = np.zeros_like(regrets)
    for n in range(len(legal)):
        count = np.sum(legal[n])
        if count == 0:
            continue
        for b in range(regrets.shape[1]):
            total = np.sum(regrets[n, b])
            for a in range(4):
                if legal[n, a]:
                    strategy[n, b, a] = regrets[n, b, a] / total if total > 0 else 1.0 / count
    return strategy


@njit(cache=True)
def _reach(actor, children, legal, strategy):
    n, buckets = len(actor), strategy.shape[1]
    reach = np.ones((n, 2, buckets))
    for node in range(n):
        player = actor[node]
        if player < 0:
            continue
        for action in range(4):
            if not legal[node, action]:
                continue
            child = children[node, action]
            for b in range(buckets):
                reach[child, player, b] = reach[node, player, b] * strategy[node, b, action]
                reach[child, 1 - player, b] = reach[node, 1 - player, b]
    return reach


@njit(cache=True)
def _values(actor, children, legal, utility, strategy):
    values = utility.copy()
    buckets = strategy.shape[1]
    for node in range(len(actor) - 1, -1, -1):
        player = actor[node]
        if player < 0:
            continue
        for r in range(buckets):
            for c in range(buckets):
                total = 0.0
                for action in range(4):
                    if legal[node, action]:
                        bucket = r if player == 0 else c
                        total += strategy[node, bucket, action] * values[children[node, action], r, c]
                values[node, r, c] = total
    return values


@njit(cache=True)
def _run(actor, children, legal, utility, priors, iterations, delay):
    buckets = utility.shape[1]
    uniform_priors = np.max(np.abs(priors - 1 / buckets)) < 1e-14
    regrets = np.zeros((len(actor), buckets, 4))
    sums = np.zeros_like(regrets)
    for iteration in range(1, iterations + 1):
        for update in range(2):
            strategy = _strategy(regrets, legal)
            reach = _reach(actor, children, legal, strategy)
            values = _values(actor, children, legal, utility, strategy)
            for node in range(len(actor)):
                if actor[node] != update:
                    continue
                for b in range(buckets):
                    for action in range(4):
                        if not legal[node, action]:
                            continue
                        child = children[node, action]
                        delta = 0.0
                        for opponent in range(buckets):
                            if update == 0:
                                difference = values[child, b, opponent] - values[node, b, opponent]
                            else:
                                difference = values[node, opponent, b] - values[child, opponent, b]
                            if uniform_priors:
                                # Preserve the original model's operation order.
                                # Tiny rounding changes can move mixtures between
                                # equally good equilibria in this discrete game.
                                delta += reach[node, 1 - update, opponent] * difference / buckets
                            else:
                                delta += (reach[node, 1 - update, opponent] * difference
                                          * priors[1 - update, opponent])
                        regrets[node, b, action] = max(0.0, regrets[node, b, action] + delta)
            # The reach is the same player's realization probability before this
            # decision, so this is the standard realization-weighted average.
            if iteration > delay:
                weight = iteration - delay
                for node in range(len(actor)):
                    if actor[node] == update:
                        for b in range(buckets):
                            for action in range(4):
                                sums[node, b, action] += (weight * reach[node, update, b]
                                                          * strategy[node, b, action])
    # _strategy normalizes any nonnegative weights, not only regrets.
    return _strategy(sums, legal)


def _priors(tree: Tree, priors=None) -> np.ndarray:
    if priors is None:
        return np.full((2, tree.buckets), 1 / tree.buckets)
    result = np.asarray(priors, dtype=np.float64)
    if (result.shape != (2, tree.buckets) or not np.all(np.isfinite(result))
            or np.any(result < 0) or np.any(result.sum(axis=1) <= 0)):
        raise ValueError("priors must be two nonnegative, nonempty bucket histograms")
    return result / result.sum(axis=1, keepdims=True)


def train(tree: Tree, iterations: int = 20000, delay: int = 100,
          priors=None) -> np.ndarray:
    return _run(tree.actor, tree.children, tree.legal, tree.utility,
                _priors(tree, priors), iterations, delay)


@njit(cache=True)
def _best_response(actor, children, legal, utility, strategy, priors, player):
    """Exact information-set best response, never revealing the other bucket."""
    buckets = strategy.shape[1]
    opponent_reach = np.ones((len(actor), buckets))
    for node in range(len(actor)):
        if actor[node] < 0:
            continue
        for action in range(4):
            if legal[node, action]:
                child = children[node, action]
                for b in range(buckets):
                    probability = strategy[node, b, action] if actor[node] != player else 1.0
                    opponent_reach[child, b] = opponent_reach[node, b] * probability
    values = np.zeros((len(actor), buckets))
    for node in range(len(actor) - 1, -1, -1):
        for b in range(buckets):
            if actor[node] < 0:
                for opp in range(buckets):
                    payoff = utility[node, b, opp] if player == 0 else -utility[node, opp, b]
                    values[node, b] += opponent_reach[node, opp] * payoff * priors[1 - player, opp]
            elif actor[node] == player:
                best = -1e100
                for action in range(4):
                    if legal[node, action]:
                        best = max(best, values[children[node, action], b])
                values[node, b] = best
            else:
                for action in range(4):
                    if legal[node, action]:
                        values[node, b] += values[children[node, action], b]
    return np.sum(values[0] * priors[player])


def evaluate(tree: Tree, strategy: np.ndarray, priors=None) -> dict[str, float]:
    """Report value/NashConv in chips of this toy game, not real poker."""
    priors = _priors(tree, priors)
    chance = priors[0, :, None] * priors[1, None, :]
    value = float(np.sum(_values(tree.actor, tree.children, tree.legal, tree.utility, strategy)[0] * chance))
    br0 = float(_best_response(tree.actor, tree.children, tree.legal, tree.utility, strategy, priors, 0))
    br1 = float(_best_response(tree.actor, tree.children, tree.legal, tree.utility, strategy, priors, 1))
    report = {"value_p0": value, "br_p0": br0, "br_p1": br1,
              "nash_conv": br0 + br1, "exploitability": (br0 + br1) / 2}
    random_policy = _strategy(np.zeros_like(strategy), tree.legal)
    call_policy = np.zeros_like(strategy)
    call_policy[:, :, 1] = 1.0
    for name, opponent in (("random", random_policy), ("always_call", call_policy)):
        results = []
        for hero in range(2):
            mixed = opponent.copy()
            mixed[tree.actor == hero] = strategy[tree.actor == hero]
            ev = float(np.sum(_values(tree.actor, tree.children, tree.legal, tree.utility, mixed)[0] * chance))
            results.append(ev if hero == 0 else -ev)
        report[f"vs_{name}_seat_average"] = float(np.mean(results))
    return report


def prior_templates(buckets: int) -> tuple[list[str], np.ndarray]:
    """Exact equal-width bucket probabilities under six Beta distributions."""
    edges = np.linspace(0, 1, buckets + 1)
    cdfs = (edges, edges ** 2, edges ** 4, 2 * edges - edges ** 2,
            2 / np.pi * np.arcsin(np.sqrt(edges)), 3 * edges ** 2 - 2 * edges ** 3)
    names = ["uniform", "strong", "nuts", "weak", "polar", "capped"]
    templates = np.asarray([np.diff(cdf) for cdf in cdfs])
    templates[0] = 1 / buckets
    return names, templates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--prior-iterations", type=int, default=10000,
                        help="iterations for each nonuniform opponent range")
    parser.add_argument("--delay", type=int, default=100)
    parser.add_argument("--buckets", type=int, default=12)
    parser.add_argument("--pot", type=int, default=8)
    parser.add_argument("--spr", type=float, nargs="+", default=[.25, .5, 1, 2, 4, 8, 16])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "river_policy.npz")
    args = parser.parse_args()
    arrays, reports = [], []
    names, templates = prior_templates(args.buckets)
    roots = np.zeros((len(args.spr), 2, len(names)), dtype=np.int32)
    offset = 0
    for grid_index, spr in enumerate(args.spr):
        tree = build_tree(args.pot, round(spr * args.pot), args.buckets)
        for template in range(len(names)):
            for hero in range(2):
                if template == 0 and hero == 1:
                    roots[grid_index, hero, template] = roots[grid_index, 0, 0]
                    continue
                started = time.monotonic()
                priors = np.vstack((templates[0], templates[0]))
                priors[1 - hero] = templates[template]
                iterations = args.iterations if template == 0 else args.prior_iterations
                strategy = train(tree, iterations, args.delay, priors)
                report = {"spr": spr, "template": names[template], "hero": hero,
                          "nodes": len(tree.actor), "iterations": iterations,
                          **evaluate(tree, strategy, priors), "seconds": time.monotonic() - started}
                print(json.dumps(report), flush=True)
                reports.append(report)
                roots[grid_index, hero, template] = offset
                children = tree.children.copy()
                children[children >= 0] += offset
                arrays.append((tree.actor, children, strategy.astype(np.float32)))
                offset += len(tree.actor)
    np.savez_compressed(args.output, format_version=np.asarray(2),
                        spr_grid=np.asarray(args.spr), roots=np.asarray(roots),
                        actor=np.concatenate([a[0] for a in arrays]),
                        children=np.concatenate([a[1] for a in arrays]),
                        strategy=np.concatenate([a[2] for a in arrays]),
                        prior_names=np.asarray(names), prior_templates=templates,
                        training_pot=np.asarray(args.pot),
                        reports_json=np.asarray(json.dumps(reports)))
    print(f"Saved {args.output} ({args.output.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
