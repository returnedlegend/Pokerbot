"""Information boundaries, legal play and reset behavior of fixed probes."""
from dataclasses import replace
import random
from types import SimpleNamespace

import numpy as np
import pytest

from game import Action, AgentConfig, Deal, Game, make_deal
from tools.challenge_opponents import ChallengeOpponent, PROFILES, SETTINGS


def play(profile, hands=8):
    bot = ChallengeOpponent(profile, seed=817249)
    memory = {}
    bot.new_half(AgentConfig(num_hands=hands), memory)
    opponent = random.Random(918273)
    answers, nets = [], []
    for index in range(hands):
        game = Game(make_deal("challenge-tests-only", index), index % 2, index)
        while not game.is_terminal:
            if game.to_act == 0:
                obs = game.observation(0)
                action = bot.choose_action(obs, memory)
                assert action in obs.legal_actions
                assert action != Action.FOLD or obs.to_call > 0
                answers.append(action)
            else:
                action = opponent.choice(game.legal_actions())
            game.apply(action)
        bot.on_hand_end(game.hand_result(0), memory)
        nets.append(game.result().nets[0])
    return bot, memory, answers, nets


@pytest.mark.parametrize("profile", PROFILES)
def test_actual_legal_games_and_repeatable_half_reset(profile):
    bot, memory, answers, nets = play(profile)
    _, replay_memory, replay_answers, replay_nets = play(profile)
    assert replay_answers == answers
    assert replay_nets == nets
    assert memory["finished"] == 7
    np.testing.assert_array_equal(memory["fold_trials"], replay_memory["fold_trials"])
    bot.new_half(AgentConfig(num_hands=8), memory)
    assert memory["hand"] == -1 and memory["finished"] == -1
    assert memory["seen"] == 0 and memory["cache"] == {}
    assert not np.any(memory["actions"]) and not np.any(memory["fold_trials"])


def flop_observations_with_different_hidden_cards():
    own, flop = (0, 5, 10, 15, 20), (24, 29, 34)
    remaining = [c for c in range(36) if c not in own+flop]
    deals = (Deal((own, tuple(remaining[:5])), flop+tuple(remaining[5:7])),
             Deal((own, tuple(remaining[7:12])), flop+tuple(remaining[12:14])))
    observations = []
    for deal in deals:
        game = Game(deal, button_seat=0)
        # Complete the blind, check the big blind, then check OOP on the flop.
        for _ in range(3):
            game.apply(Action.CHECK_CALL)
        observations.append(game.observation(0))
    assert observations[0] == observations[1]
    return observations


@pytest.mark.parametrize("profile", PROFILES)
def test_actual_opponent_cards_and_future_board_cannot_change_decision(profile):
    left, right = flop_observations_with_different_hidden_cards()
    bot = ChallengeOpponent(profile)
    a, b = {}, {}
    bot.new_half(AgentConfig(), a)
    bot.new_half(AgentConfig(), b)
    assert bot.choose_action(left, a) == bot.choose_action(right, b)
    assert a["diagnostics"] == b["diagnostics"]
    for ((sample_a, ranks_a), (sample_b, ranks_b)) in zip(a["cache"].values(), b["cache"].values()):
        np.testing.assert_array_equal(sample_a["wins"], sample_b["wins"])
        for st in ranks_a:
            np.testing.assert_array_equal(ranks_a[st], ranks_b[st])


def test_visible_actions_count_once_and_terminal_fold_once_without_hole_cards():
    bot, memory = ChallengeOpponent("challenge_adaptive"), {}
    bot.new_half(AgentConfig(), memory)
    game = Game(make_deal("challenge-terminal-test", 0), button_seat=0)
    obs = game.observation(0)
    # Deliver an actual legal raise followed by a terminal fold. Results expose
    # only the three public fields this callback needs; no hole/board property.
    bot._observe(obs, memory)
    memory["last_action"] = Action.RAISE_HALF
    game.apply(Action.RAISE_HALF)
    game.apply(Action.FOLD)
    public = SimpleNamespace(hand_index=0, winner="me", went_to_showdown=False)
    bot.on_hand_end(public, memory)
    before = memory["fold_trials"].copy()
    bot.on_hand_end(public, memory)
    np.testing.assert_array_equal(before, memory["fold_trials"])
    assert memory["fold_trials"][0, 0] == 1
    assert memory["folds"][0, 0] == 1
    next_game = Game(make_deal("challenge-terminal-test", 1), button_seat=0, hand_index=1)
    next_game.apply(Action.CHECK_CALL)
    obs = next_game.observation(1)
    bot._observe(obs, memory)
    before = memory["actions"].copy()
    bot._observe(obs, memory)
    np.testing.assert_array_equal(before, memory["actions"])


def test_style_shift_uses_only_public_hand_counter_and_fixed_config():
    bot, memory = ChallengeOpponent("challenge_shifting"), {}
    bot.new_half(AgentConfig(num_hands=400), memory)
    obs = Game(make_deal("challenge-shift-test", 0), button_seat=0).observation(0)
    assert [bot.effective_profile(replace(obs, hand_index=index), memory) for index in (0, 100, 200, 300)] == [
        "challenge_value", "challenge_polarized", "challenge_station", "challenge_checkraise"]
    assert all(128 <= settings.samples <= 384 for settings in SETTINGS.values())
    with pytest.raises(ValueError, match="Unknown challenge"):
        ChallengeOpponent("imaginary")
