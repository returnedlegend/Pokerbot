"""Rules, information-set solver math, and optional runtime mapping checks."""

from dataclasses import replace

import numpy as np
import pytest

from game import Action, Game, Street, make_deal
import river_solver
from tools.train_river_policy import build_tree, evaluate, prior_templates, train


def river_game(pot=8, stack=8):
    game = Game(make_deal("river-policy-test", 0), button_seat=1)
    game.street = Street.RIVER
    game.stacks = [stack, stack]
    game.contributed = [pot // 2, pot // 2]
    game.bets = [0, 0]
    game.acted = [False, False]
    game.to_act = 0
    game.raises_this_street = 0
    game.last_increment = 2
    game.history = []
    return game


@pytest.mark.parametrize("stack", [2, 4, 8, 16, 32, 64, 128])
def test_abstract_tree_matches_event_betting_rules(stack):
    tree = build_tree(pot=8, stack=stack, buckets=3)
    pending = [(0, river_game(stack=stack))]
    while pending:
        node, game = pending.pop()
        if game.is_terminal:
            assert tree.actor[node] == -1
            result = game.result()
            if not result.went_to_showdown:
                assert np.all(tree.utility[node] == result.nets[0])
            else:
                assert tree.utility[node, 2, 0] == 4 + game.bets[1]
                assert tree.utility[node, 0, 2] == -4 - game.bets[0]
                assert tree.utility[node, 1, 1] == 0
            continue
        assert tree.actor[node] == game.to_act
        assert tuple(tree.bets[node]) == tuple(game.bets)
        for action in Action:
            child = tree.children[node, action]
            assert (child >= 0) == (action in game.legal_actions())
            if child < 0:
                continue
            if action in (Action.RAISE_HALF, Action.RAISE_POT):
                assert tree.targets[node, action] == game.raise_to(action)
            # The duplicate edge leads to the same public betting state.
            if not tree.legal[node, action]:
                assert action == Action.RAISE_POT
                assert child == tree.children[node, Action.RAISE_HALF]
                continue
            following = game.clone()
            following.apply(action)
            pending.append((child, following))


def test_best_response_does_not_see_opponent_bucket():
    # Two types, each occurring 1/2. Against always-check/call, value-betting
    # the high type gains 1 chip conditionally, or 1/2 chip before the deal.
    tree = build_tree(pot=8, stack=2, buckets=2)
    always_call = np.zeros((len(tree.actor), 2, 4))
    always_call[:, :, Action.CHECK_CALL] = 1
    report = evaluate(tree, always_call)
    assert report["value_p0"] == 0
    assert report["br_p0"] == .5
    assert report["br_p1"] == .5
    assert report["nash_conv"] == 1


def test_nonuniform_best_response_uses_both_chance_priors():
    tree = build_tree(pot=8, stack=2, buckets=2)
    always_call = np.zeros((len(tree.actor), 2, 4))
    always_call[:, :, Action.CHECK_CALL] = 1
    # At showdown P0 wins .25*.25 and loses .75*.75. Each player
    # best-responds by value-betting its high type and checking its low type.
    report = evaluate(tree, always_call, [[.75, .25], [.25, .75]])
    assert report["value_p0"] == -2
    assert report["br_p0"] == -1.875
    assert report["br_p1"] == 3.125
    assert report["nash_conv"] == 1.25


def test_cfr_converges_in_the_toy_game():
    tree = build_tree(pot=8, stack=16, buckets=6)
    short = evaluate(tree, train(tree, iterations=20, delay=0))
    trained = evaluate(tree, train(tree, iterations=2000, delay=100))
    assert trained["nash_conv"] < .005
    assert trained["nash_conv"] < short["nash_conv"] / 10
    assert trained["vs_random_seat_average"] > 0
    assert trained["vs_always_call_seat_average"] > 0


@pytest.mark.parametrize("hero", [0, 1])
def test_cfr_converges_with_nonuniform_chance_priors(hero):
    tree = build_tree(pot=8, stack=16, buckets=6)
    _, templates = prior_templates(6)
    priors = np.vstack((templates[0], templates[0]))
    priors[1 - hero] = templates[2]  # Beta(4,1): a much stronger root range.
    short = evaluate(tree, train(tree, iterations=20, delay=0, priors=priors), priors)
    trained = evaluate(tree, train(tree, iterations=3000, delay=100, priors=priors), priors)
    assert trained["nash_conv"] < .005
    assert trained["nash_conv"] < short["nash_conv"] / 10


def test_saved_model_is_converged_in_its_own_abstraction():
    model = river_solver._model()
    for index, spr in enumerate(model["spr_grid"]):
        tree = build_tree(pot=int(model["training_pot"]), stack=round(float(spr) * 8), buckets=12)
        for hero in range(2):
            for template in range(len(model["prior_templates"])):
                start = int(model["roots"][index, hero, template])
                strategy = model["strategy"][start:start + len(tree.actor)].astype(np.float64)
                priors = np.vstack((model["prior_templates"][0], model["prior_templates"][0]))
                priors[1 - hero] = model["prior_templates"][template]
                assert evaluate(tree, strategy, priors)["nash_conv"] < .001


def test_runtime_maps_every_exact_grid_history_and_masks_actions():
    pending = [river_game()]
    while pending:
        game = pending.pop()
        if game.is_terminal:
            continue
        obs = game.observation(game.to_act)
        for equity in (0, .31, .8, 1):
            probabilities = river_solver.policy(obs, equity)
            assert probabilities is not None
            assert sum(probabilities.values()) == pytest.approx(1)
            assert set(probabilities) <= set(obs.legal_actions)
            assert river_solver.choose_action(obs, equity) in obs.legal_actions
            assert river_solver.choose_action(obs, equity) == river_solver.choose_action(obs, equity)
        for action in game.legal_actions():
            following = game.clone()
            following.apply(action)
            pending.append(following)


def test_runtime_root_reconstruction_and_invalid_inputs():
    game = river_game()
    game.apply(Action.RAISE_HALF)
    obs = game.observation(1)
    assert obs.pot == 12 and obs.my_stack == 8 and obs.opp_stack == 4
    assert river_solver.policy(obs, .6) is not None
    assert river_solver.policy(replace(obs, street=Street.TURN), .6) is None
    assert river_solver.policy(replace(obs, my_stack=7), .6) is None
    assert river_solver.policy(obs, float("nan")) is None
    assert river_solver.policy(obs, 1.01) is None


def test_duplicate_allin_raises_have_one_canonical_probability():
    obs = river_game(stack=2).observation(0)
    assert obs.raise_to[Action.RAISE_HALF] == obs.raise_to[Action.RAISE_POT]
    probabilities = river_solver.policy(obs, 1)
    assert probabilities is not None
    assert Action.RAISE_POT not in probabilities


def test_exact_prior_templates_and_uniform_backward_compatibility():
    model = river_solver._model()
    names, templates = prior_templates(12)
    assert names == list(model["prior_names"])
    assert np.all(templates > 0)
    assert np.allclose(templates.sum(axis=1), 1)
    for index, template in enumerate(templates):
        assert river_solver._prior_index(model, template) == index
    obs = river_game().observation(0)
    assert river_solver.policy(obs, .6) == river_solver.policy(obs, .6, templates[0])
    assert river_solver.choose_action(obs, .6) == river_solver.choose_action(obs, .6, templates[0])
    assert river_solver.policy(obs, .6, templates[2]) is not None
    assert river_solver.policy(obs, .6, [-1] + [1] * 11) is None
    assert river_solver.policy(obs, .6, [0] * 12) is None
    assert river_solver.policy(obs, .6, [1] * 11) is None


def test_strong_opponent_prior_changes_strategy_with_valid_probabilities():
    model = river_solver._model()
    obs = river_game().observation(0)
    uniform = river_solver.policy(obs, .6)
    strong = river_solver.policy(obs, .6, model["prior_templates"][2])
    assert strong is not None and uniform is not None
    assert sum(strong.values()) == pytest.approx(1)
    assert set(strong) <= set(obs.legal_actions)
    # A hand barely above median has less value when the opponent often has
    # the nuts. Test behavior, in addition to the independent chance math.
    assert strong.get(Action.RAISE_HALF, 0) < uniform.get(Action.RAISE_HALF, 0)
