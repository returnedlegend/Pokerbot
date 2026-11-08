"""Range-aware adaptation of PyPokerEngine's Monte Carlo equity policy.

No pretrained Hold'em weights: hidden cards are sampled using the event's
exact deck/evaluator. Betting and range parameters are tuned offline.
"""
from pathlib import Path
from dataclasses import replace
import json
import hashlib
from math import comb
import numpy as np
from game import Action, Street
import opponent_model
import equity_network
import river_solver
from equity_engine import sample_equities, score_hand, preflop_strength, warmup

PARAMS = {
    'samples': 256,
    'refine_samples': 768,
    'refine_z': 1.25,
    'refine_min_pot': 12,
    'ess_floor': 48,
    'draw_model': False,
    'neural_anchor': False,
    'river_mix': 0.,
    'river_range': False,
    'anchor_error': .022,
    'sensible_checks': True,
    'showdown_learning': False,
    'quality_prior': 12.,
    'adaptation': 1.0,
    'caution': .006,
    'style_decay': .99,
    'preflop_table': True,
    'range_weight': 0.75,
    'call_margin': 0.025,
    'future_cap': False,
    'realization': 0.94,
    'raise_margin': 0.035,
    'pot_margin': None,
    'fold_prior': 0.32,
    'bluff_scale': 0.65,
    'preflop_raise': 0.56,
}
_config = Path(__file__).with_name('policy.json')
if _config.exists():
    PARAMS.update(json.loads(_config.read_text())['parameters'])
warmup()
if PARAMS['draw_model'] or PARAMS['neural_anchor']:
    equity_network.warmup()
if PARAMS['river_mix'] > 0:
    river_solver._model()
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
                  last_obs=None, last_action=None, cache={}, refinements=0, river_uses=0)
    opponent_model.init(memory, decay=PARAMS['style_decay'])


def _observe(obs, memory):
    opponent_model.observe(obs, memory)


def _ranks(values):
    # Midrank ties matter: paired boards produce many equal Omaha hands.
    _, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends[inv] - .5*counts[inv]) / len(values)


def _sigmoid(x):
    return 1. / (1. + np.exp(-np.clip(x, -30, 30)))


def _particles(obs, memory):
    key = (tuple(obs.hole_cards), tuple(obs.board))
    target = int(memory.get('particle_target', PARAMS['samples']))
    existing = memory['cache'].get(key)
    if existing is None or len(existing['wins']) < target:
        seed = 2166136261
        for c in (*obs.hole_cards, *obs.board):
            seed = ((seed ^ (c+1)) * 16777619) & 0x7fffffff
        memory['cache'][key] = sample_equities(obs.hole_cards, obs.board,
                                               n=target, seed=seed)
    return memory['cache'][key]


def _strength_ranks(obs, particles):
    key = (bool(PARAMS['preflop_table']), bool(PARAMS['draw_model']))
    cache = particles.setdefault('rank_cache', {})
    if key not in cache:
        pre_values = (_lookup_preflop(particles['villain_holes'])
                      if PARAMS['preflop_table'] else particles['villain_preflop'])
        pre_rank = _ranks(pre_values)
        board_ranks, board_values = {}, {}
        histories = particles.get('villain_scores_by_street')
        for street in range(1, int(obs.street)+1):
            scores = histories[:, street-1] if histories is not None else particles['villain_scores']
            if PARAMS['draw_model'] and street in (int(Street.FLOP), int(Street.TURN)):
                # Rank betting hands by independently learned visible-card
                # equity, including draws, never by the sampled future outcome.
                board = obs.board[:3 if street == int(Street.FLOP) else 4]
                strengths = equity_network.batchpredict(particles['villain_holes'], board, scores=scores)
                board_values[street] = strengths
                board_ranks[street] = _ranks(strengths)
            else:
                board_values[street] = scores
                board_ranks[street] = _ranks(scores)
        particles.setdefault('strength_values', {})[key] = pre_values, board_values
        cache[key] = pre_rank, board_ranks
    return cache[key]


def _conditional_quantiles(values, weights):
    _, inverse = np.unique(values, return_inverse=True)
    masses = np.bincount(inverse, weights=weights)
    ends = np.cumsum(masses)
    return ((ends-.5*masses)/ends[-1])[inverse]


def _revealed_likelihood(memory, street, context, action, ranks, weights, base):
    counts = np.asarray(opponent_model.bet_quality(memory, street, context['facing'], action))
    count = float(counts.sum())
    if count == 0: return base
    # The PIT coordinate removes earlier action selection: ranks are measured
    # within the current pre-bet posterior, rather than counted twice.
    quantiles = _conditional_quantiles(ranks, weights)
    bins = np.minimum(4, (quantiles*5).astype(np.int32))
    density = (counts+.5)/(count+2.5)*5.
    learned = density[bins]
    base = base/max(1.e-12, float(np.dot(weights,base)/weights.sum()))
    learned = learned/max(1.e-12, float(np.dot(weights,learned)/weights.sum()))
    trust = count/(count+PARAMS['quality_prior'])
    return (1.-trust)*base+trust*learned


def _posterior(obs, memory, particles):
    wins = particles['wins']
    pre_rank, board_ranks = _strength_ranks(obs, particles)
    if PARAMS['preflop_table'] and obs.street == Street.PREFLOP:
        wins = wins.astype(np.float64) + float(_lookup_preflop(obs.hole_cards)) - float(wins.mean())
    pooled = opponent_model.global_rates(memory, PARAMS['fold_prior'])
    rate = pooled['aggression']
    contexts = opponent_model.context_history(obs)
    weights = np.ones(len(wins))
    for i, action in enumerate(obs.action_history):
        if action.is_me: continue
        rank = pre_rank if action.street == Street.PREFLOP else board_ranks[int(action.street)]
        context = contexts[i]
        tendency = opponent_model.profile(memory, action.street, context['facing'], context['price'])
        trust = tendency['confidence']*PARAMS['adaptation']
        context_rate = (1.-trust)*rate + trust*tendency['aggression']
        selectivity = PARAMS['range_weight']*max(0., 1.-1.8*context_rate)
        if action.action in (Action.RAISE_HALF, Action.RAISE_POT):
            threshold = min(.9, max(.48, 1.-context_rate))
            likelihood = (.10+.90*_sigmoid((rank-threshold)*12.)) ** selectivity
            terminal_wager = (action.street == Street.RIVER or
                              (i == len(obs.action_history)-1 and
                               (obs.to_call >= obs.my_stack or obs.opp_stack == 0)))
            if PARAMS['showdown_learning'] and terminal_wager:
                likelihood = _revealed_likelihood(memory, action.street, context,
                                                   action.action, rank, weights, likelihood)
            weights *= likelihood
        elif action.action == Action.CHECK_CALL and context['voluntary']:
            weights *= (.35+.65*_sigmoid((rank-.28)*9.)) ** selectivity
        elif action.action == Action.CHECK_CALL and action.street != Street.PREFLOP:
            if PARAMS['sensible_checks']:
                # A passive opponent checks strong hands too. The likelihood
                # of a check must complement their estimated betting policy.
                threshold = min(.9, max(.48, 1.-context_rate))
                curve = (.10+.90*_sigmoid((rank-threshold)*12.)) ** selectivity
                normalizer = float(np.dot(weights, curve)/weights.sum())
                bet_probability = np.clip(context_rate*curve/max(1.e-12, normalizer), 0., .98)
                weights *= 1.-bet_probability
            else:
                weights *= (1.-.65*_sigmoid((rank-.78)*10.)) ** selectivity
    weights /= weights.sum()
    anchor_alpha = 0.
    if PARAMS['neural_anchor'] and obs.street in (Street.FLOP, Street.TURN):
        if 'hero_anchor' not in particles:
            score = particles['hero_score'] if 'hero_score' in particles else score_hand(obs.hole_cards, obs.board)
            particles['hero_anchor'] = float(equity_network.batchpredict(
                np.asarray([obs.hole_cards]), obs.board, scores=np.asarray([score]))[0])
        uniform = float(wins.mean())
        raw_range = float(np.dot(weights, wins))
        variance_mean = float(np.mean((wins-uniform)**2)/len(wins))
        covariance = float(np.dot(weights*(wins-raw_range), wins-uniform)/len(wins))
        anchor_alpha = float(np.clip(covariance/(variance_mean+PARAMS['anchor_error']**2), 0., 1.))
        wins = wins.astype(np.float64) + anchor_alpha*(particles['hero_anchor']-uniform)
    raw_equity = float(np.dot(weights, wins))
    equity = float(np.clip(raw_equity, 0., 1.))
    effective = float(1./np.dot(weights, weights))
    influence = weights*(wins-raw_equity)
    if PARAMS['preflop_table'] and obs.street == Street.PREFLOP:
        # The preflop control variate uses the same samples: account for its
        # covariance instead of treating the subtracted sample mean as fixed.
        influence = influence-(wins-float(wins.mean()))/len(wins)
    elif anchor_alpha:
        influence = influence-anchor_alpha*(wins-float(wins.mean()))/len(wins)
    error = float(np.sqrt(np.dot(influence, influence)+(anchor_alpha*PARAMS['anchor_error'])**2))
    current = pre_rank if obs.street == Street.PREFLOP else board_ranks[int(obs.street)]
    return wins, weights, equity, effective, error, current, pooled


def choose_action(obs, memory):
    if not memory:
        # Useful for standalone analysis; the server always calls new_half.
        from game import AgentConfig
        new_half(AgentConfig(), memory)
    _observe(obs, memory)
    if len(obs.legal_actions) == 1:
        memory['last_action'] = obs.legal_actions[0]
        return obs.legal_actions[0]
    memory['particle_target'] = int(PARAMS['samples'])
    particles = _particles(obs, memory)
    wins, weights, eq, effective, error, current_rank, pooled = _posterior(obs, memory, particles)
    call = min(obs.to_call, obs.my_stack)
    pot = obs.pot
    price = call / max(1, pot+call)
    realization = 1. if obs.street == Street.RIVER or call == obs.my_stack else PARAMS['realization']
    # Position buys information and improves equity realization.
    if obs.is_button:
        realization = min(1., realization+.025)
    # Once a call commits the stack, there are no later decisions to
    # justify a future-street realization penalty. Use actual pot odds.
    margin = PARAMS['call_margin'] if obs.street != Street.RIVER and call < obs.my_stack else 0.
    threshold = (price+margin)/max(.01, realization)
    future_limit = max(0, min(obs.my_stack-call, obs.opp_stack))
    if PARAMS['future_cap']:
        # Always calling later can cost at most the effective chips left.
        # A future-risk discount cannot exceed that enforceable lower bound.
        threshold = min(threshold, price+future_limit/max(1,pot+call))
    close_call = bool(call and pot+call >= PARAMS['refine_min_pot']
                      and abs(eq-threshold) <= PARAMS['refine_z']*error)
    low_effective = bool(call and pot+call >= 32 and effective < PARAMS['ess_floor'])
    if (close_call or low_effective) and len(particles['wins']) < PARAMS['refine_samples']:
        old_count = len(particles['wins'])
        memory['particle_target'] = int(PARAMS['refine_samples'])
        particles = _particles(obs, memory)
        wins, weights, eq, effective, error, current_rank, pooled = _posterior(obs, memory, particles)
        memory['refinements'] += int(len(particles['wins']) > old_count)
    memory['last_equity'] = {'equity': eq, 'mc_error': error, 'effective_samples': effective,
                             'samples': len(particles['wins'])}
    call_ev = (eq*realization-margin)*(pot+call)-call
    if PARAMS['future_cap']:
        call_ev = max(call_ev, eq*(pot+call)-call-future_limit)
    best_action = Action.CHECK_CALL
    best_ev = call_ev
    if obs.to_call and Action.FOLD in obs.legal_actions and call_ev < 0:
        best_ev, best_action = 0., Action.FOLD

    recent_fold = pooled['fold_rate']
    current_style = opponent_model.profile(memory, obs.street, bool(obs.to_call), price)
    fold_rate = recent_fold
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
        response_style = opponent_model.profile(memory, obs.street, True, fraction)
        trust_fold = response_style['fold_confidence']*PARAMS['adaptation']
        fold_rate = (1.-trust_fold)*recent_fold + trust_fold*response_style['fold_lower']
        cutoff = np.clip(fold_rate + .40*(fraction-.25), .015, .92)
        continuation = .025+.975*_sigmoid((current_rank-cutoff)*12.)
        # Very low observed folding: exploit stations by value betting, with
        # little imagined fold equity. High aggression also suggests this.
        continuation = 1.-(1.-continuation)*min(1.,fold_rate/.20)
        probability_call = float(np.dot(weights, continuation))
        weighted_win = float(np.clip(np.dot(weights*continuation, wins), 0., probability_call))
        fold_prob = (1.-probability_call)*PARAMS['bluff_scale']
        called_equity = weighted_win/max(1.e-9, probability_call)
        # Discount speculative fold equity; conserve value of checking.
        ev = fold_prob*pot + weighted_win*(pot+investment+response) - probability_call*investment
        caution = PARAMS['caution']*(1.-current_style['confidence'])
        if obs.to_call and called_equity < .65:
            caution += PARAMS['caution']
        if action == Action.RAISE_POT:
            caution += .5*PARAMS['caution']
        action_margin = PARAMS['raise_margin']
        if action == Action.RAISE_POT and PARAMS['pot_margin'] is not None:
            action_margin = PARAMS['pot_margin']
        risk_margin = (action_margin+caution)*(pot+investment)
        if obs.street == Street.PREFLOP:
            # No bloated preflop pots merely on noisy 50/50 estimates.
            if eq < PARAMS['preflop_raise'] - max(0., fold_rate-.45)*.5:
                continue
        if ev > best_ev + risk_margin:
            best_ev, best_action = ev, action
    if obs.street == Street.RIVER and PARAMS['river_mix'] > 0:
        # A compact solved betting abstraction supplies a robust mixed river
        # baseline. It is not an equilibrium solution for the actual card game.
        public = repr((obs.hole_cards, obs.board, obs.pot, obs.to_call,
                       tuple((int(a.action), a.amount, a.is_me) for a in obs.action_history)))
        draw = int.from_bytes(hashlib.blake2b(public.encode(), digest_size=8,
                                             person=b'river-mix').digest(), 'little')/2**64
        if draw < PARAMS['river_mix']:
            prior = None
            if PARAMS['river_range']:
                # Condition only on streets BEFORE this river. The solved
                # policy itself handles the river betting history.
                root_obs = replace(obs, action_history=tuple(a for a in obs.action_history
                                                             if a.street != Street.RIVER))
                _, prior_weights, _, _, _, root_ranks, _ = _posterior(root_obs, memory, particles)
                buckets = river_solver._model()['strategy'].shape[1]
                cells = np.minimum(buckets-1, (root_ranks*buckets).astype(np.int32))
                prior = np.bincount(cells, weights=prior_weights, minlength=buckets)
                prior /= prior.sum()
            mixed = river_solver.choose_action(obs, float(particles['wins'].mean()),
                                                opponent_prior=prior)
            if mixed is not None:
                best_action = mixed
                memory['river_uses'] += 1
    memory['last_action'] = best_action
    return best_action if best_action in obs.legal_actions else Action.CHECK_CALL


def _learn_terminal_bet(result, memory):
    obs = memory.get('last_obs')
    if (not result.went_to_showdown or result.opp_hole is None or obs is None
            or obs.hand_index != result.hand_index or memory.get('last_action') != Action.CHECK_CALL
            or not obs.to_call or not obs.action_history):
        return
    terminal_call = (obs.street == Street.RIVER or obs.to_call >= obs.my_stack or obs.opp_stack == 0)
    action = obs.action_history[-1]
    if not terminal_call or action.is_me or action.action not in (Action.RAISE_HALF, Action.RAISE_POT):
        return
    particles = _particles(obs, memory)
    before = replace(obs, action_history=obs.action_history[:-1])
    _, weights, _, _, _, _, _ = _posterior(before, memory, particles)
    key = (bool(PARAMS['preflop_table']), bool(PARAMS['draw_model']))
    pre, boards = particles['strength_values'][key]
    if obs.street == Street.PREFLOP:
        reference = pre
        actual = float(_lookup_preflop(result.opp_hole)) if PARAMS['preflop_table'] else preflop_strength(result.opp_hole)
    elif PARAMS['draw_model'] and obs.street in (Street.FLOP, Street.TURN):
        reference = boards[int(obs.street)]
        score = score_hand(result.opp_hole, obs.board)
        actual = float(equity_network.batchpredict(np.asarray([result.opp_hole]), obs.board,
                                                   scores=np.asarray([score]))[0])
    else:
        reference = boards[int(obs.street)]
        actual = score_hand(result.opp_hole, obs.board)
    quantile = float(np.clip(np.dot(weights, (reference < actual)+.5*(reference == actual)), 0., 1.))
    context = opponent_model.context_history(obs)[-1]
    opponent_model.record_bet_quality(memory, obs.street, context['facing'], action.action, quantile)


def on_hand_end(result, memory):
    # Only a terminal wager we called yields an uncensored direct label. A
    # hand that later folded cannot supply imaginary weak/strong hole cards.
    if PARAMS['showdown_learning'] and result.hand_index > memory['opponent_model']['finished']:
        _learn_terminal_bet(result, memory)
    opponent_model.finish(result, memory)
