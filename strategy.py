"""Range-aware adaptation of PyPokerEngine's Monte Carlo equity policy.

No pretrained Hold'em weights: hidden cards are sampled using the event's
exact deck/evaluator. Betting and range parameters are tuned offline.
"""
from pathlib import Path
import json
from math import comb
import numpy as np
from game import Action, Street
from equity_engine import sample_equities, warmup

PARAMS = {
    'samples': 192,
    'preflop_table': True,
    'range_weight': 0.75,
    'call_margin': 0.025,
    'realization': 0.94,
    'raise_margin': 0.035,
    'fold_prior': 0.32,
    'bluff_scale': 0.65,
    'preflop_raise': 0.56,
}
_config = Path(__file__).with_name('policy.json')
if _config.exists():
    PARAMS.update(json.loads(_config.read_text())['parameters'])
warmup()
# Offline Monte Carlo fit, generated without observing evaluation deals.
with np.load(Path(__file__).with_name('preflop_model.npz')) as _data:
    _PREFLOP = _data['equity']
_COMB = np.array([[comb(c, k) if c >= k else 0 for k in range(6)]
                  for c in range(36)], dtype=np.int64)


def _lookup_preflop(hands):
    cards = np.sort(np.asarray(hands), axis=-1)
    return _PREFLOP[np.sum(_COMB[cards, np.arange(1, 6)], axis=-1)]


def new_half(config, memory):
    memory.update(hands_played=0, opp_actions=0, opp_raises=0,
                  faced=0, folded=0, last_hand=-1, seen=0,
                  last_obs=None, last_action=None, cache={})


def _observe(obs, memory):
    if memory['last_hand'] != obs.hand_index:
        memory['last_hand'] = obs.hand_index
        memory['seen'] = 0
        memory['cache'] = {}
    history = obs.action_history
    for i in range(memory['seen'], len(history)):
        a = history[i]
        if a.is_me:
            continue
        memory['opp_actions'] += 1
        if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
            memory['opp_raises'] += 1
        previous = history[i-1] if i else None
        if previous and previous.is_me and previous.action in (Action.RAISE_HALF, Action.RAISE_POT):
            memory['faced'] += 1
            if a.action == Action.FOLD:
                memory['folded'] += 1
    memory['seen'] = len(history)
    memory['last_obs'] = obs


def _ranks(values):
    # Midrank ties matter: paired boards produce many equal Omaha hands.
    _, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends[inv] - .5*counts[inv]) / len(values)


def _sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(x, -30, 30)))


def _particles(obs, memory):
    key = tuple(obs.board)
    if key not in memory['cache']:
        seed = 2166136261
        for c in (*obs.hole_cards, *obs.board):
            seed = ((seed ^ (c+1)) * 16777619) & 0x7fffffff
        memory['cache'][key] = sample_equities(obs.hole_cards, obs.board,
                                               n=int(PARAMS['samples']), seed=seed)
    return memory['cache'][key]


def choose_action(obs, memory):
    if not memory:
        # Useful for standalone analysis; the server always calls new_half.
        from game import AgentConfig
        new_half(AgentConfig(), memory)
    _observe(obs, memory)
    if len(obs.legal_actions) == 1:
        memory['last_action'] = obs.legal_actions[0]
        return obs.legal_actions[0]
    particles = _particles(obs, memory)
    wins = particles['wins']
    n = len(wins)
    use_table = PARAMS['preflop_table']
    pre_rank = _ranks(_lookup_preflop(particles['villain_holes']) if use_table else particles['villain_preflop'])
    if use_table and obs.street == Street.PREFLOP:
        # A control variate anchors noisy conditional estimates to a much
        # better sampled unconditional equity. Range effects still use live
        # legal samples, rather than assuming the opponent is uniform.
        wins = wins.astype(np.float64) + float(_lookup_preflop(obs.hole_cards)) - float(wins.mean())
    current_rank = pre_rank
    board_ranks = {}
    scores_history = particles.get('villain_scores_by_street')
    for s in range(1, int(obs.street)+1):
        scores = scores_history[:, s-1] if scores_history is not None else particles['villain_scores']
        board_ranks[s] = _ranks(scores)
    if obs.street != Street.PREFLOP:
        current_rank = board_ranks[int(obs.street)]
    rate = (memory['opp_raises']+3.)/(memory['opp_actions']+12.)
    # Frequent indiscriminate aggressors should not be credited with nuts.
    selectivity = PARAMS['range_weight'] * max(0., 1. - 1.8*rate)
    weights = np.ones(n)
    for i, a in enumerate(obs.action_history):
        if a.is_me:
            continue
        rank = pre_rank if a.street == Street.PREFLOP else board_ranks[int(a.street)]
        previous = obs.action_history[i-1] if i else None
        facing_raise = previous and previous.is_me and previous.action in (Action.RAISE_HALF, Action.RAISE_POT) and previous.street == a.street
        if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
            threshold = min(.9, max(.48, 1.-rate))
            likelihood = .10 + .90*_sigmoid((rank-threshold)*12.)
            weights *= likelihood ** selectivity
        elif a.action == Action.CHECK_CALL and facing_raise:
            weights *= (.35+.65*_sigmoid((rank-.28)*9.)) ** selectivity
        elif a.action == Action.CHECK_CALL and a.street != Street.PREFLOP:
            weights *= (1.-.65*_sigmoid((rank-.78)*10.)) ** selectivity
    weights /= weights.sum()
    eq = float(np.clip(np.dot(weights, wins), 0., 1.))
    call = min(obs.to_call, obs.my_stack)
    pot = obs.pot
    price = call / max(1, pot+call)
    realization = 1. if obs.street == Street.RIVER or call == obs.my_stack else PARAMS['realization']
    # Position buys information and improves equity realization.
    if obs.is_button:
        realization = min(1., realization+.025)
    margin = PARAMS['call_margin'] if obs.street != Street.RIVER else 0.
    call_ev = (eq*realization-margin)*(pot+call)-call
    best_action = Action.CHECK_CALL
    best_ev = call_ev
    if obs.to_call and Action.FOLD in obs.legal_actions and call_ev < 0:
        best_ev, best_action = 0., Action.FOLD

    fold_rate = (memory['folded']+8.*PARAMS['fold_prior'])/(memory['faced']+8.)
    for action in (Action.RAISE_HALF, Action.RAISE_POT):
        if action not in obs.legal_actions:
            continue
        investment = obs.raise_to[action]-obs.my_bet
        response = min(obs.opp_stack, max(0, obs.raise_to[action]-obs.opp_bet))
        if response <= 0:
            continue
        # Smoothed observed folds determine continuation range, not a fixed
        # assumption that every opponent plays rationally.
        fraction = response / max(1, pot+investment+response)
        cutoff = np.clip(fold_rate + .40*(fraction-.25), .015, .92)
        continuation = .025+.975*_sigmoid((current_rank-cutoff)*12.)
        # Very low observed folding: exploit stations by value betting, with
        # little imagined fold equity. High aggression also suggests this.
        continuation = 1.-(1.-continuation)*min(1.,fold_rate/.20)
        probability_call = float(np.dot(weights, continuation))
        weighted_win = float(np.clip(np.dot(weights*continuation, wins), 0., probability_call))
        fold_prob = (1.-probability_call)*PARAMS['bluff_scale']
        # Discount speculative fold equity; conserve value of checking.
        ev = fold_prob*pot + weighted_win*(pot+investment+response) - probability_call*investment
        risk_margin = PARAMS['raise_margin']*(pot+investment)
        if obs.street == Street.PREFLOP:
            # No bloated preflop pots merely on noisy 50/50 estimates.
            if eq < PARAMS['preflop_raise'] - max(0., fold_rate-.45)*.5:
                continue
        if ev > best_ev + risk_margin:
            best_ev, best_action = ev, action
    memory['last_action'] = best_action
    return best_action if best_action in obs.legal_actions else Action.CHECK_CALL


def on_hand_end(result, memory):
    obs = memory.get('last_obs')
    # A final opponent fold is absent from the last decision observation.
    if obs is not None and obs.hand_index == result.hand_index:
        last = memory.get('last_action')
        if last in (Action.RAISE_HALF, Action.RAISE_POT):
            memory['faced'] += 1
            memory['opp_actions'] += 1
            if result.winner == 'me' and not result.went_to_showdown:
                memory['folded'] += 1
        elif last == Action.CHECK_CALL and not obs.to_call and not obs.is_button and obs.street == Street.RIVER:
            memory['opp_actions'] += 1
    memory['hands_played'] += 1
