"""Sampling decisions must be precise without leaking sampled future cards."""
from dataclasses import replace
import numpy as np
import pytest
import strategy
from equity_engine import sample_equities
from game import Action,AgentConfig,Game,Street,make_deal


def fixture_observation(street=Street.TURN):
    deal=make_deal('sampling-test',0)
    obs=Game(deal,0).observation(0)
    return replace(obs,street=street,board=deal.board[:{Street.PREFLOP:0,Street.FLOP:3,Street.TURN:4}[street]],
                   pot=116,to_call=44,my_stack=44,opp_stack=0,
                   legal_actions=(Action.FOLD,Action.CHECK_CALL),raise_to={},action_history=())


@pytest.mark.parametrize('draw_model',[False,True])
def test_range_rank_does_not_use_sampled_winner(monkeypatch,draw_model):
    obs=fixture_observation()
    particles=sample_equities(obs.hole_cards,obs.board,n=32,seed=701)
    other={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in particles.items()}
    other['wins']=1.-other['wins']
    monkeypatch.setitem(strategy.PARAMS,'draw_model',draw_model)
    a=strategy._strength_ranks(obs,particles)
    b=strategy._strength_ranks(obs,other)
    np.testing.assert_array_equal(a[0],b[0])
    for street in (1,2): np.testing.assert_array_equal(a[1][street],b[1][street])


def toy_particles(n,equity):
    wins=np.zeros(n)
    wins[:round(n*equity)]=1.
    return {'wins':wins}


def configure_sampling(monkeypatch):
    monkeypatch.setitem(strategy.PARAMS,'neural_anchor',False)
    monkeypatch.setitem(strategy.PARAMS,'draw_model',False)
    for k,v in {'samples':256,'refine_samples':768,'refine_z':1.25,'refine_min_pot':12,'ess_floor':48}.items():
        monkeypatch.setitem(strategy.PARAMS,k,v)
    monkeypatch.setattr(strategy,'_strength_ranks',lambda obs,p:(np.full(len(p['wins']),.5),
                        {s:np.full(len(p['wins']),.5) for s in (1,2,3)}))


def test_close_terminal_call_gets_more_samples(monkeypatch):
    configure_sampling(monkeypatch)
    requests=[]
    def sampling(obs,memory):
        n=memory['particle_target'];requests.append(n)
        return toy_particles(n,.27 if n==256 else .30)
    monkeypatch.setattr(strategy,'_particles',sampling)
    memory={};strategy.new_half(AgentConfig(),memory)
    assert strategy.choose_action(fixture_observation(),memory)==Action.CHECK_CALL
    assert requests==[256,768]
    assert memory['refinements']==1
    assert memory['last_equity']['samples']==768


def test_obvious_fold_does_not_spend_refinement_budget(monkeypatch):
    configure_sampling(monkeypatch)
    requests=[]
    def sampling(obs,memory):
        requests.append(memory['particle_target'])
        return toy_particles(256,.10)
    monkeypatch.setattr(strategy,'_particles',sampling)
    memory={};strategy.new_half(AgentConfig(),memory)
    assert strategy.choose_action(fixture_observation(),memory)==Action.FOLD
    assert requests==[256]
    assert memory['refinements']==0


def test_uniform_preflop_control_variate_has_no_live_sampling_noise():
    obs=fixture_observation(Street.PREFLOP)
    p=sample_equities(obs.hole_cards,(),n=64,seed=991)
    memory={};strategy.new_half(AgentConfig(),memory)
    a=strategy._posterior(obs,memory,p)
    other={k:v for k,v in p.items() if k!='rank_cache'}
    other['wins']=1.-p['wins']
    b=strategy._posterior(obs,memory,other)
    assert abs(a[2]-b[2])<1.e-12
    assert a[4]<1.e-12 and b[4]<1.e-12


def test_refined_particles_keep_seed_prefix_and_are_not_downgraded():
    obs=fixture_observation()
    memory={};strategy.new_half(AgentConfig(),memory)
    memory['particle_target']=64
    first=strategy._particles(obs,memory)
    memory['particle_target']=128
    refined=strategy._particles(obs,memory)
    np.testing.assert_array_equal(first['wins'],refined['wins'][:64])
    np.testing.assert_array_equal(first['villain_holes'],refined['villain_holes'][:64])
    memory['particle_target']=64
    assert strategy._particles(obs,memory) is refined
