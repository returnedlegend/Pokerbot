"""Behavioral checks for evidence-based, per-half opponent adaptation.

Use public observations and hand-end results only. The tests avoid injecting
synthetic model counters: differing styles must be learned through the same
interface used in a match.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import opponent_model as model
from game import Action, Game, HandResult, ObservedAction, Street, make_deal


def fresh():
    memory = {}
    model.init(memory)
    return memory


def observation(hand, street, history, *, is_button=True, to_call=0):
    obs = Game(make_deal("opponent-model", hand), button_seat=0,
               hand_index=hand).observation(0)
    board = () if street == Street.PREFLOP else (6, 11, 18)[:3]
    if street == Street.TURN:
        board += (21,)
    elif street == Street.RIVER:
        board += (21, 30)
    return replace(obs, street=street, board=board,
                   action_history=tuple(history), is_button=is_button,
                   to_call=to_call)


def action(street, mine, choice, amount=0, total=0):
    return ObservedAction(street, mine, choice, amount, total)


def result(hand, *, showdown=True, winner="split"):
    return HandResult(hand, 0, 20, showdown, (), None, winner)


def learn_open_actions(memory, actions, start=0, street=Street.FLOP):
    for hand, choice in enumerate(actions, start):
        raising = choice in (Action.RAISE_HALF, Action.RAISE_POT)
        obs = observation(hand, street, [action(street, False, choice,
                                               10 if raising else 0,
                                               10 if raising else 0)],
                          is_button=True, to_call=10 if raising else 0)
        model.observe(obs, memory)
        memory['last_action'] = Action.CHECK_CALL
        model.finish(result(hand), memory)


def learn_facing_actions(memory, actions, start=0, street=Street.FLOP):
    for hand, choice in enumerate(actions, start):
        # Our flop raise and the opponent's visible response. A raise-back
        # keeps action alive, while this public-history fixture is sufficient
        # for checking context classification independently of card strength.
        history = [action(street, True, Action.RAISE_HALF, 10, 10),
                   action(street, False, choice,
                          20 if choice == Action.RAISE_POT else 10,
                          20 if choice == Action.RAISE_POT else 10)]
        obs = observation(hand, street, history,
                          to_call=10 if choice == Action.RAISE_POT else 0)
        model.observe(obs, memory)
        memory['last_action'] = Action.CHECK_CALL
        model.finish(result(hand), memory)


def test_prior_is_finite_and_does_not_assume_an_extreme_style():
    memory = fresh()
    for street in Street:
        for facing in (False, True):
            profile = model.profile(memory, street, facing=facing)
            assert 0 < profile['aggression'] < 1
            assert 0 < profile['fold_rate'] < 1
            assert 0 <= profile['confidence'] <= 1


def test_repeated_public_history_is_counted_once():
    memory = fresh()
    obs = observation(0, Street.FLOP,
                      [action(Street.FLOP, False, Action.RAISE_POT, 10, 10)],
                      to_call=10)
    model.observe(obs, memory)
    once = deepcopy(memory)
    model.observe(obs, memory)
    assert memory == once


def test_single_raise_is_shrunk_instead_of_labeling_opponent_a_maniac():
    memory = fresh()
    before = model.profile(memory, Street.FLOP)
    learn_open_actions(memory, [Action.RAISE_POT])
    after = model.profile(memory, Street.FLOP)
    assert before['aggression'] < after['aggression'] < .8
    assert after['confidence'] < .5


def test_learns_different_passive_and_aggressive_styles():
    passive, aggressive = fresh(), fresh()
    learn_open_actions(passive, [Action.CHECK_CALL]*80)
    learn_open_actions(aggressive, [Action.RAISE_POT]*80)
    low = model.profile(passive, Street.FLOP)
    high = model.profile(aggressive, Street.FLOP)
    assert high['aggression'] > low['aggression'] + .45
    assert high['confidence'] > .5
    assert low['confidence'] > .5


def test_open_bets_and_raise_backs_are_distinct_evidence():
    memory = fresh()
    learn_open_actions(memory, [Action.RAISE_POT]*80)
    learn_facing_actions(memory, [Action.CHECK_CALL]*80, start=80)
    opening = model.profile(memory, Street.FLOP, facing=False)
    facing = model.profile(memory, Street.FLOP, facing=True)
    assert opening['aggression'] > facing['aggression'] + .35


def test_preflop_aggression_does_not_erase_passive_river_evidence():
    memory = fresh()
    learn_open_actions(memory, [Action.RAISE_POT]*80, street=Street.PREFLOP)
    learn_open_actions(memory, [Action.CHECK_CALL]*80, start=80,
                       street=Street.RIVER)
    preflop = model.profile(memory, Street.PREFLOP)
    river = model.profile(memory, Street.RIVER)
    assert preflop['aggression'] > river['aggression'] + .35


def test_prior_street_raise_does_not_turn_flop_check_into_call():
    memory = fresh()
    for hand in range(40):
        obs = observation(hand, Street.FLOP, [
            action(Street.PREFLOP, True, Action.RAISE_POT, 5, 6),
            action(Street.PREFLOP, False, Action.CHECK_CALL, 4, 6),
            action(Street.FLOP, False, Action.CHECK_CALL),
        ])
        model.observe(obs, memory)
        memory['last_action'] = Action.CHECK_CALL
        model.finish(result(hand), memory)
    opening = model.profile(memory, Street.FLOP, facing=False)
    facing = model.profile(memory, Street.FLOP, facing=True)
    assert opening['confidence'] > facing['confidence']
    assert opening['aggression'] < facing['aggression']


@pytest.mark.parametrize('winner', ['me', 'opp', 'split'])
def test_terminal_all_in_call_counts_as_a_call_not_a_fold(winner):
    memory = fresh()
    for hand in range(40):
        obs = observation(hand, Street.RIVER, [], is_button=False)
        model.observe(obs, memory)
        memory['last_action'] = Action.RAISE_POT
        model.finish(result(hand, showdown=True, winner=winner), memory)
    learned = model.profile(memory, Street.RIVER, facing=True)
    assert learned['fold_rate'] < .15
    assert learned['confidence'] > .5


def test_terminal_opponent_folds_are_learned():
    memory = fresh()
    for hand in range(40):
        obs = observation(hand, Street.RIVER, [], is_button=False)
        model.observe(obs, memory)
        memory['last_action'] = Action.RAISE_HALF
        model.finish(result(hand, showdown=False, winner='me'), memory)
    assert model.profile(memory, Street.RIVER, facing=True)['fold_rate'] > .75


def test_seen_call_is_not_counted_again_as_terminal_fold():
    memory = fresh()
    obs = observation(0, Street.RIVER, [
        action(Street.TURN, True, Action.RAISE_POT, 10, 10),
        action(Street.TURN, False, Action.CHECK_CALL, 10, 10),
        action(Street.RIVER, False, Action.CHECK_CALL),
    ], is_button=True)
    model.observe(obs, memory)
    before = model.profile(memory, Street.TURN, facing=True)
    memory['last_action'] = Action.CHECK_CALL
    model.finish(result(0), memory)
    after = model.profile(memory, Street.TURN, facing=True)
    assert after == before


def test_recent_style_switch_changes_model_without_unlearning_history():
    memory = fresh()
    learn_open_actions(memory, [Action.CHECK_CALL]*100)
    passive = model.profile(memory, Street.FLOP)['aggression']
    learn_open_actions(memory, [Action.RAISE_POT]*100, start=100)
    switched = model.profile(memory, Street.FLOP)['aggression']
    assert switched > passive + .35
    assert switched < 1


def test_reinitialization_clears_previous_opponent_information():
    memory = fresh()
    learn_open_actions(memory, [Action.RAISE_POT]*40)
    model.init(memory)
    assert model.profile(memory, Street.FLOP) == model.profile(fresh(), Street.FLOP)


def test_blind_price_and_voluntary_raise_are_distinguished():
    obs = observation(0, Street.FLOP, [
        action(Street.PREFLOP, True, Action.RAISE_POT, 5, 6),
        action(Street.PREFLOP, False, Action.CHECK_CALL, 4, 6),
        action(Street.FLOP, False, Action.CHECK_CALL),
    ], is_button=True)
    contexts = model.context_history(obs)
    assert contexts[0] == {'facing': True, 'price': .25, 'voluntary': False}
    assert contexts[1]['facing'] and contexts[1]['voluntary']
    assert contexts[1]['price'] == pytest.approx(1/3)
    assert contexts[2] == {'facing': False, 'price': 0., 'voluntary': False}


def test_blind_call_does_not_count_as_response_to_our_voluntary_bet():
    memory = fresh()
    obs = observation(0, Street.PREFLOP, [
        action(Street.PREFLOP, False, Action.CHECK_CALL, 1, 2),
    ], is_button=False)
    model.observe(obs, memory)
    assert memory['opp_actions'] == 1
    assert memory['faced'] == 0
    assert model.profile(memory, Street.PREFLOP, facing=True)['confidence'] > 0
    assert model.profile(memory, Street.PREFLOP, facing=True)['fold_confidence'] == 0


def test_size_specific_fold_evidence_uses_matching_price_and_uncertainty():
    memory = fresh()
    for hand in range(80):
        obs = observation(hand, Street.RIVER, [], is_button=False)
        obs = replace(obs, pot=36, my_bet=0, opp_bet=0,
                      my_stack=62, opp_stack=62,
                      raise_to={Action.RAISE_HALF: 18, Action.RAISE_POT: 36})
        model.observe(obs, memory)
        folding = hand < 40
        memory['last_action'] = Action.RAISE_POT if folding else Action.RAISE_HALF
        model.finish(result(hand, showdown=not folding,
                            winner='me' if folding else 'split'), memory)
    large = model.profile(memory, Street.RIVER, facing=True, price=1/3)
    half = model.profile(memory, Street.RIVER, facing=True, price=.25)
    assert large['fold_rate'] > half['fold_rate'] + .5
    for profile in (large, half):
        assert 0 <= profile['fold_lower'] <= profile['fold_rate'] <= 1
        assert profile['fold_confidence'] > .5


def test_terminal_result_is_idempotent():
    memory = fresh()
    obs = observation(0, Street.RIVER, [], is_button=False)
    model.observe(obs, memory)
    memory['last_action'] = Action.RAISE_POT
    done = result(0, showdown=False, winner='me')
    model.finish(done, memory)
    once = deepcopy(memory)
    model.finish(done, memory)
    assert memory == once


def test_global_fallback_has_finite_priors_and_is_read_only():
    memory = fresh()
    before = deepcopy(memory)
    for prior in (.15, .37, .8):
        rates = model.global_rates(memory, fold_prior=prior)
        assert 0 < rates['aggression'] < 1
        assert 0 < rates['fold_rate'] < 1
    assert memory == before


def test_global_aggression_forgets_old_style_without_rewriting_diagnostics():
    memory = fresh()
    model.init(memory, decay=.99)
    learn_open_actions(memory, [Action.RAISE_POT]*80)
    aggressive = model.global_rates(memory)['aggression']
    lifetime = {k: memory[k] for k in ('opp_actions', 'opp_raises', 'faced', 'folded')}
    # An interval with no observed opponent actions reduces certainty, rather
    # than fabricating new passive actions or rewriting lifetime diagnostics.
    model.observe(observation(1000, Street.FLOP, []), memory)
    assert {k: memory[k] for k in lifetime} == lifetime
    assert model.global_rates(memory)['aggression'] < aggressive
    learn_open_actions(memory, [Action.CHECK_CALL]*40, start=1001)
    recent = model.global_rates(memory)['aggression']
    lifetime_rate = (memory['opp_raises']+3)/(memory['opp_actions']+12)
    assert recent < .5*lifetime_rate
    assert memory['opp_actions'] == lifetime['opp_actions']+40
    assert memory['opp_raises'] == lifetime['opp_raises']


def test_global_folding_tracks_recent_calls_without_rewriting_lifetime_folds():
    memory = fresh()
    model.init(memory, decay=.99)
    for hand in range(80):
        model.observe(observation(hand, Street.RIVER, [], is_button=False), memory)
        memory['last_action'] = Action.RAISE_HALF
        model.finish(result(hand, showdown=False, winner='me'), memory)
    lifetime = {k: memory[k] for k in ('opp_actions', 'opp_raises', 'faced', 'folded')}
    model.observe(observation(1000, Street.RIVER, []), memory)
    assert {k: memory[k] for k in lifetime} == lifetime
    for hand in range(1001, 1041):
        model.observe(observation(hand, Street.RIVER, [], is_button=False), memory)
        memory['last_action'] = Action.RAISE_HALF
        model.finish(result(hand, showdown=True, winner='split'), memory)
    recent = model.global_rates(memory)['fold_rate']
    lifetime_rate = (memory['folded']+8*.37)/(memory['faced']+8)
    assert recent < .5*lifetime_rate
    assert memory['faced'] == lifetime['faced']+40
    assert memory['folded'] == lifetime['folded']
