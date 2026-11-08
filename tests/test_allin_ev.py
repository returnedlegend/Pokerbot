"""An all-in call must use showdown equity without future-street penalties."""
from dataclasses import replace
import numpy as np
import pytest
import strategy
from game import Action, AgentConfig, Game, Street, make_deal


@pytest.mark.parametrize('street,board', [(Street.FLOP,(5,10,15)),(Street.TURN,(5,10,15,20)),(Street.RIVER,(5,10,15,20,25))])
def test_profitable_allin_is_called_even_with_thin_edge(monkeypatch,street,board):
    monkeypatch.setitem(strategy.PARAMS,'neural_anchor',False)
    monkeypatch.setitem(strategy.PARAMS,'draw_model',False)
    base=Game(make_deal('allin-ev-test',0),0).observation(0)
    obs=replace(base,street=street,board=board,pot=116,to_call=44,my_stack=44,
                opp_stack=0,legal_actions=(Action.FOLD,Action.CHECK_CALL),raise_to={},
                action_history=())
    # Needs27.5%;29% is profitable. There are no future bets after calling.
    sample={'wins':np.array([1.]*29+[0.]*71),
            'villain_holes':np.tile(np.array([1,2,3,4,6]),(100,1)),
            'villain_preflop':np.full(100,.5), 'villain_scores':np.zeros(100),
            'villain_scores_by_street':np.zeros((100,3))}
    monkeypatch.setattr(strategy,'_particles',lambda obs,memory:sample)
    memory={};strategy.new_half(AgentConfig(),memory)
    assert strategy.choose_action(obs,memory)==Action.CHECK_CALL


@pytest.mark.parametrize('bounded,expected',[(False,Action.FOLD),(True,Action.CHECK_CALL)])
def test_future_risk_cannot_exceed_effective_chips_left(monkeypatch,bounded,expected):
    from equity_engine import sample_equities
    deal=make_deal('one-chip-future',0)
    obs=replace(Game(deal,0).observation(0),street=Street.TURN,board=deal.board[:4],
                pot=115,to_call=43,my_stack=44,opp_stack=1,my_bet=34,opp_bet=77,
                raises_this_street=4,legal_actions=(Action.FOLD,Action.CHECK_CALL),
                raise_to={},action_history=())
    p=sample_equities(obs.hole_cards,obs.board,n=100,seed=511)
    p['wins']=np.array([1.]*28+[0.]*72)
    monkeypatch.setattr(strategy,'_particles',lambda obs,memory:p)
    for k,v in {'neural_anchor':False,'draw_model':False,'future_cap':bounded,'refine_samples':100}.items():
        monkeypatch.setitem(strategy.PARAMS,k,v)
    memory={};strategy.new_half(AgentConfig(),memory)
    # Call EV before future betting is .28*158-43=1.24chips. Only one chip
    # remains, so always calling later guarantees a positive lower bound.
    assert strategy.choose_action(obs,memory)==expected
