"""Small, uncertainty-aware opponent model from public betting actions only.

Beta/Dirichlet smoothing and an exponential recent-history window prevent a
few hands from defining an opponent. Context means street and whether a wager
is already outstanding, not a fixed personality label. See docs/OPPONENT_RESEARCH.md.
"""
from math import sqrt
from game import Action, Street

DECAY = .996


def init(memory, decay=DECAY):
    for key in ('hands_played', 'opp_actions', 'opp_raises', 'faced', 'folded', 'seen'):
        memory.setdefault(key, 0)
    memory.setdefault('last_hand', -1)
    memory.setdefault('last_obs', None)
    memory.setdefault('last_action', None)
    memory.setdefault('cache', {})
    memory['opponent_model'] = {
        'counts': [[[0., 0., 0.] for _ in range(2)] for _ in range(4)],
        'responses': [[[0., 0.] for _ in range(3)] for _ in range(4)],
        'last_decay': -1, 'finished': -1, 'decay': float(decay),
        'revealed_bets': [[[[0.]*5 for _ in range(2)] for _ in range(2)] for _ in range(4)],
    }


def context_history(obs):
    """Replay only visible actions to distinguish a check from a priced call."""
    small = max(1, obs.big_blind // 2)
    bets = [small if obs.is_button else obs.big_blind, obs.big_blind if obs.is_button else small]
    voluntary = [False, False]
    street = Street.PREFLOP
    pot = small + obs.big_blind
    contexts = []
    for action in obs.action_history:
        if action.street != street:
            street = action.street
            bets = [0, 0]
            voluntary = [False, False]
        actor = 0 if action.is_me else 1
        call = max(0, bets[1-actor]-bets[actor])
        contexts.append({'facing': bool(call), 'price': call/max(1, pot+call),
                         'voluntary': bool(call and voluntary[1-actor])})
        bets[actor] = action.street_total
        pot += action.amount
        if action.action in (Action.RAISE_HALF, Action.RAISE_POT):
            voluntary[actor] = True
    return contexts


def _bucket(price):
    return 0 if price <= .20 else (1 if price <= .30 else 2)


def _decay(memory, hand):
    model = memory['opponent_model']
    previous = model['last_decay']
    if hand > previous:
        factor = model['decay'] ** max(0, hand-max(0, previous))
        for rows in model['counts']:
            for row in rows:
                for i in range(3): row[i] *= factor
        for rows in model['responses']:
            for row in rows:
                for i in range(2): row[i] *= factor
        for street in model['revealed_bets']:
            for facing in street:
                for counts in facing:
                    for i in range(5): counts[i] *= factor
        model['last_decay'] = hand


def _record(memory, street, action, facing, price, voluntary):
    model = memory['opponent_model']
    raised = action in (Action.RAISE_HALF, Action.RAISE_POT)
    folded = action == Action.FOLD
    model['counts'][int(street)][int(facing)][2 if raised else (0 if folded else 1)] += 1.
    memory['opp_actions'] += 1
    memory['opp_raises'] += int(raised)
    if facing and voluntary:
        pair = model['responses'][int(street)][_bucket(price)]
        pair[0] += int(folded)
        pair[1] += 1.
        memory['faced'] += 1
        memory['folded'] += int(folded)


def observe(obs, memory):
    if 'opponent_model' not in memory: init(memory)
    _decay(memory, obs.hand_index)
    if memory['last_hand'] != obs.hand_index:
        memory['last_hand'] = obs.hand_index
        memory['seen'] = 0
        memory['cache'] = {}
    contexts = context_history(obs)
    for i in range(memory['seen'], len(obs.action_history)):
        action = obs.action_history[i]
        if not action.is_me:
            ctx = contexts[i]
            _record(memory, action.street, action.action, **ctx)
    memory['seen'] = len(obs.action_history)
    memory['last_obs'] = obs


def finish(result, memory):
    if 'opponent_model' not in memory: init(memory)
    model = memory['opponent_model']
    if result.hand_index <= model['finished']: return
    _decay(memory, result.hand_index)
    obs = memory.get('last_obs')
    last = memory.get('last_action')
    if obs is not None and obs.hand_index == result.hand_index:
        if last in (Action.RAISE_HALF, Action.RAISE_POT):
            target = obs.raise_to.get(last, obs.opp_bet + obs.big_blind)
            investment = max(0, target-obs.my_bet)
            response = min(obs.opp_stack, max(0, target-obs.opp_bet))
            folded = not result.went_to_showdown and result.winner == 'me'
            _record(memory, obs.street, Action.FOLD if folded else Action.CHECK_CALL,
                    True, response/max(1, obs.pot+investment+response), True)
        elif last == Action.CHECK_CALL and not obs.to_call and not obs.is_button and obs.street == Street.RIVER:
            _record(memory, obs.street, Action.CHECK_CALL, False, 0., False)
    model['finished'] = result.hand_index
    memory['hands_played'] += 1


def profile(memory, street, facing=False, price=.25):
    model = memory['opponent_model']
    rows = model['counts']
    # Pool the same decision context across streets before shrinking toward it.
    pooled = [sum(rows[s][int(facing)][i] for s in range(4)) for i in range(3)]
    pooled_rate = (pooled[2]+4.)/(sum(pooled)+16.)
    row = rows[int(street)][int(facing)]
    count = sum(row)
    aggression = (row[2]+8.*pooled_rate)/(count+8.)
    responses = model['responses'][int(street)]
    folds = sum(p[0] for p in responses)
    faced = sum(p[1] for p in responses)
    # Match the actual wager's price only after observing enough size examples.
    bucket = responses[_bucket(price)]
    prior = (folds+8.*.37)/(faced+8.)
    mean = (bucket[0]+12.*prior)/(bucket[1]+12.)
    effective = min(faced, bucket[1]+12.)
    error = sqrt(max(0., mean*(1.-mean)/(effective+9.)))
    return {'aggression': aggression, 'fold_rate': mean,
            'fold_lower': max(0., mean-error), 'confidence': count/(count+24.),
            'fold_confidence': faced/(faced+24.), 'effective_count': count}


def global_rates(memory, fold_prior=.37):
    """Recent pooled fallback; diagnostics' lifetime totals must not drive it."""
    model = memory['opponent_model']
    actions = sum(sum(row) for street in model['counts'] for row in street)
    raises = sum(row[2] for street in model['counts'] for row in street)
    folded = sum(row[0] for street in model['responses'] for row in street)
    faced = sum(row[1] for street in model['responses'] for row in street)
    return {'aggression': (raises+3.)/(actions+12.),
            'fold_rate': (folded+8.*fold_prior)/(faced+8.)}


def record_bet_quality(memory, street, facing, action, quantile):
    """Record only legally revealed terminal betting-hand quality.

    The strategy supplies a percentile within the pre-bet conditional range,
    not the eventual winning outcome or an unrevealed opponent holding.
    """
    if action not in (Action.RAISE_HALF, Action.RAISE_POT):
        raise ValueError('Expected a betting action')
    if not 0. <= quantile <= 1.:
        raise ValueError('Expected a percentile in [0,1]')
    kind = 0 if action == Action.RAISE_HALF else 1
    bucket = min(4, int(quantile*5))
    memory['opponent_model']['revealed_bets'][int(street)][int(facing)][kind][bucket] += 1.


def bet_quality(memory, street, facing, action):
    kind = 0 if action == Action.RAISE_HALF else 1
    return tuple(memory['opponent_model']['revealed_bets'][int(street)][int(facing)][kind])
