"""A learned visible-card expectation reduces MC noise with bounded trust."""
from dataclasses import replace
import numpy as np
import pytest
import strategy
from game import AgentConfig,Game,Street,make_deal


def observation(street=Street.TURN):
    deal=make_deal('anchor-test',0)
    obs=Game(deal,0).observation(0)
    return replace(obs,street=street,board=deal.board[:4 if street==Street.TURN else 5],action_history=())


def sample(n):
    return {'wins':np.tile([0.,1.],n//2),'hero_score':0}


def setup(monkeypatch):
    monkeypatch.setitem(strategy.PARAMS,'draw_model',False)
    monkeypatch.setattr(strategy,'_strength_ranks',lambda obs,p:(np.full(len(p['wins']),.5),
                           {s:np.full(len(p['wins']),.5) for s in (1,2,3)}))
    monkeypatch.setattr(strategy.equity_network,'batchpredict',lambda hands,board,scores=None:np.full(len(hands),.6))


def test_anchor_moves_noisy_estimate_toward_model_without_changing_range(monkeypatch):
    setup(monkeypatch)
    obs=observation();memory={};strategy.new_half(AgentConfig(),memory)
    monkeypatch.setitem(strategy.PARAMS,'neural_anchor',False)
    baseline=strategy._posterior(obs,memory,sample(192))
    monkeypatch.setitem(strategy.PARAMS,'neural_anchor',True)
    anchored=strategy._posterior(obs,memory,sample(192))
    assert baseline[2]==pytest.approx(.5)
    assert .5<anchored[2]<.6
    assert anchored[4]<baseline[4]
    np.testing.assert_array_equal(anchored[1],baseline[1])


def test_more_precise_sampling_gets_less_model_correction(monkeypatch):
    setup(monkeypatch);monkeypatch.setitem(strategy.PARAMS,'neural_anchor',True)
    obs=observation();memory={};strategy.new_half(AgentConfig(),memory)
    small=strategy._posterior(obs,memory,sample(192))[2]
    large=strategy._posterior(obs,memory,sample(3072))[2]
    assert .5<large<small<.6


def test_untrained_river_is_not_sent_to_equity_network(monkeypatch):
    setup(monkeypatch);monkeypatch.setitem(strategy.PARAMS,'neural_anchor',True)
    monkeypatch.setattr(strategy.equity_network,'batchpredict',lambda *a,**k:(_ for _ in ()).throw(AssertionError('river has no network model')))
    memory={};strategy.new_half(AgentConfig(),memory)
    assert strategy._posterior(observation(Street.RIVER),memory,sample(192))[2]==pytest.approx(.5)
