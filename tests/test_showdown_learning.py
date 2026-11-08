"""Revealed cards can calibrate bet types; folded hands cannot invent labels."""
import numpy as np
import pytest
import opponent_model as model
import strategy
from game import Action,AgentConfig,Game,Street,make_deal


def fresh():
    memory={};strategy.new_half(AgentConfig(),memory);return memory


def test_revealed_bet_types_are_separate_and_regularized(monkeypatch):
    memory=fresh()
    ranks=np.linspace(0,1,100)
    weights=np.full(100,.01)
    base=.1+.9*strategy._sigmoid((ranks-.75)*12.)
    context={'facing':False}
    for _ in range(40):model.record_bet_quality(memory,Street.RIVER,False,Action.RAISE_HALF,.05)
    half=strategy._revealed_likelihood(memory,Street.RIVER,context,Action.RAISE_HALF,ranks,weights,base)
    pot=strategy._revealed_likelihood(memory,Street.RIVER,context,Action.RAISE_POT,ranks,weights,base)
    assert half[0]>half[-1] # Confirmed weak half-bets are not credited as nuts.
    np.testing.assert_array_equal(pot,base) # No learned evidence for pot bets.
    assert np.all(half>0) and np.isfinite(half).all()


def test_conditional_quantiles_use_prior_not_eventual_winners():
    ranks=np.array([0.,0.,1.,2.])
    prior=np.array([.1,.1,.2,.6])
    quantile=strategy._conditional_quantiles(ranks,prior)
    np.testing.assert_allclose(quantile,[.1,.1,.3,.7])


@pytest.mark.parametrize('draw_model',[False,True])
def test_only_terminal_called_raise_supplies_one_legal_label(monkeypatch,draw_model):
    monkeypatch.setitem(strategy.PARAMS,'showdown_learning',True)
    monkeypatch.setitem(strategy.PARAMS,'draw_model',draw_model)
    game=Game(make_deal('legal-showdown-label',0),button_seat=1)
    for action in (Action.RAISE_POT,Action.RAISE_POT,Action.RAISE_POT,Action.CHECK_CALL):
        game.apply(action)
    assert game.street==Street.FLOP and game.to_act==0
    game.apply(Action.CHECK_CALL)
    game.apply(Action.RAISE_POT)
    obs=game.observation(0)
    assert obs.opp_stack==0 and obs.to_call==obs.my_stack
    memory=fresh();strategy._observe(obs,memory)
    memory['particle_target']=64
    strategy._particles(obs,memory)
    memory['last_action']=Action.CHECK_CALL
    game.apply(Action.CHECK_CALL)
    result=game.hand_result(0)
    assert result.went_to_showdown and result.opp_hole is not None
    strategy.on_hand_end(result,memory)
    counts=model.bet_quality(memory,Street.FLOP,False,Action.RAISE_POT)
    assert sum(counts)==1
    strategy.on_hand_end(result,memory)
    assert model.bet_quality(memory,Street.FLOP,False,Action.RAISE_POT)==counts


def test_folded_hand_never_supplies_an_imaginary_label(monkeypatch):
    monkeypatch.setitem(strategy.PARAMS,'showdown_learning',True)
    game=Game(make_deal('fold-not-label',0),button_seat=1)
    game.apply(Action.RAISE_POT)
    obs=game.observation(0)
    memory=fresh();strategy._observe(obs,memory)
    memory['last_action']=Action.FOLD
    game.apply(Action.FOLD)
    result=game.hand_result(0)
    assert result.opp_hole is None
    strategy.on_hand_end(result,memory)
    total=sum(sum(counts) for street in memory['opponent_model']['revealed_bets'] for facing in street for counts in facing)
    assert total==0
