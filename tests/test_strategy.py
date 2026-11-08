"""Behavioral invariants beyond the organizer's HTTP contract tests."""

import builtins
import copy
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent
from game import Action, AgentConfig, Deal, Game, HandResult, ObservedAction, Street, make_deal


def initialized_memory():
    memory = {}
    agent.new_half(AgentConfig(), memory)
    return memory


def test_fresh_half_reproduces_decisions():
    """A duplicate half must not depend on knowledge retained in globals."""
    observations = [Game(make_deal("repeatability", i), i % 2, i).observation(i % 2)
                    for i in range(12)]
    first = initialized_memory()
    actions = [agent.choose_action(obs, first) for obs in observations]
    second = initialized_memory()
    assert actions == [agent.choose_action(obs, second) for obs in observations]


def test_unrevealed_cards_cannot_affect_action():
    """Two states with identical information must give identical decisions."""
    deal = make_deal("hidden-information", 1)
    visible = set(deal.hole_cards[0])
    available = [card for card in range(36) if card not in visible]
    alternate = Deal(hole_cards=(deal.hole_cards[0], tuple(available[:5])),
                     board=tuple(available[5:10]))
    first = Game(deal, button_seat=0, hand_index=1).observation(0)
    second = Game(alternate, button_seat=0, hand_index=1).observation(0)
    assert first == second
    memory = initialized_memory()
    assert agent.choose_action(first, copy.deepcopy(memory)) == agent.choose_action(second, copy.deepcopy(memory))


def test_decisions_and_learning_need_no_writable_repository(monkeypatch):
    """After import/warmup, play and result handling must not write files."""
    original_open = builtins.open

    def read_only_open(file, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in ("w", "a", "x", "+")), str(file)
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", read_only_open)
    memory = initialized_memory()
    for index in range(8):
        game = Game(make_deal("readonly", index), button_seat=index % 2, hand_index=index)
        while not game.is_terminal:
            observation = game.observation(game.to_act)
            action = agent.choose_action(observation, memory) if game.to_act == 0 else Action.CHECK_CALL
            assert action in observation.legal_actions
            game.apply(action)
        agent.on_hand_end(game.hand_result(0), memory)


def test_legal_when_raise_cap_or_stack_prevents_raise():
    """Exercise capped raises and short stacks in both seats."""
    found_restricted = False
    for our_seat in (0, 1):
        memory = initialized_memory()
        for index in range(25):
            game = Game(make_deal("stack-edges", index), button_seat=index % 2, hand_index=index)
            while not game.is_terminal:
                observation = game.observation(game.to_act)
                if game.to_act == our_seat:
                    action = agent.choose_action(observation, memory)
                    assert action in observation.legal_actions
                    if Action.RAISE_POT not in observation.legal_actions:
                        found_restricted = True
                # Drive the game to edge states independently of strategy choice.
                forced = Action.RAISE_POT if Action.RAISE_POT in observation.legal_actions else Action.CHECK_CALL
                game.apply(forced)
            agent.on_hand_end(game.hand_result(our_seat), memory)
    assert found_restricted


@pytest.mark.parametrize("showdown,winner,expected_fold", [(False, "me", 1), (True, "opp", 0), (True, "split", 0)])
def test_terminal_opponent_response_is_counted(showdown, winner, expected_fold):
    memory = initialized_memory()
    observation = Game(make_deal("terminal", 0), button_seat=0, hand_index=0).observation(0)
    memory.update(last_obs=observation, last_action=Action.RAISE_POT)
    result = HandResult(0, 0, 160, showdown, (), None, winner)
    agent.on_hand_end(result, memory)
    assert memory["faced"] == 1
    assert memory["folded"] == expected_fold
    assert memory["opp_actions"] == 1


def test_seen_opponent_call_is_not_counted_again_at_hand_end():
    import strategy

    memory = initialized_memory()
    observation = Game(make_deal("seen-call", 0), button_seat=0, hand_index=0).observation(0)
    observation = replace(observation, street=Street.RIVER, to_call=0,
                          action_history=(
                              ObservedAction(Street.TURN, True, Action.RAISE_POT, 10, 10),
                              ObservedAction(Street.TURN, False, Action.CHECK_CALL, 10, 10)))
    strategy._observe(observation, memory)
    # Our final action was a river check, not another raise.
    memory["last_action"] = Action.CHECK_CALL
    before = memory["faced"]
    agent.on_hand_end(HandResult(0, 0, 20, True, (), None, "split"), memory)
    assert before == 1
    assert memory["faced"] == before
    assert memory["folded"] == 0
