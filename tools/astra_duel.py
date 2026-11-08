"""Untimed, capability-isolated exhibition referee for direct LLM decisions.

The poker agent is unchanged. Each player endpoint exposes only that player's
observation and legally revealed hand results. Two fresh LLM contexts play
opposite seats on duplicate deals. Full traces are written only after both
halves finish, preventing cross-half information leakage through saved logs.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import secrets
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi import FastAPI, HTTPException
import uvicorn
from evaluator import cards_to_str, category_name
from equity_engine import score_hand
from game import Action, AgentConfig, Game, button_for_deal, make_deal
import strategy

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
MATCH = None


class Half:
    def __init__(self, match, number):
        self.match = match
        self.number = number
        self.bot_seat = number-1
        self.memory = {}
        strategy.new_half(AgentConfig(num_hands=match.count), self.memory)
        self.index = 0
        self.game = None
        self.request = 0
        self.pending = []
        self.nets = []
        self.traces = []
        self.astra_actions = []
        self.bot_seconds = 0.
        self.bot_calls = 0
        self.invalid_requests = 0
        self.started = time.monotonic()

    def state(self):
        while self.index < self.match.count:
            if self.game is None:
                self.game = Game(self.match.deals[self.index], button_for_deal(self.index), self.index)
            game = self.game
            if game.is_terminal:
                result = game.result()
                self.nets.append(result.nets[self.bot_seat])
                self.pending.append(game.hand_result(1-self.bot_seat).to_dict())
                strategy.on_hand_end(game.hand_result(self.bot_seat), self.memory)
                self.traces.append({'hand_index': self.index, 'bot_seat': self.bot_seat,
                                    'deal': asdict(self.match.deals[self.index]), 'result': asdict(result)})
                self.index += 1
                self.game = None
                continue
            if game.to_act == self.bot_seat:
                obs = game.observation(self.bot_seat)
                begin = time.perf_counter()
                action = strategy.choose_action(obs, self.memory)
                self.bot_seconds += time.perf_counter()-begin
                self.bot_calls += 1
                if action not in obs.legal_actions:
                    raise RuntimeError('Bot returned illegal action')
                game.apply(action)
                continue
            obs = game.observation(1-self.bot_seat)
            visible = obs.to_dict()
            visible['hole_text'] = cards_to_str(obs.hole_cards)
            visible['board_text'] = cards_to_str(obs.board)
            if len(obs.board) >= 3:
                visible['current_made_hand_category'] = category_name(score_hand(obs.hole_cards, obs.board))
            return {'finished': False, 'half': self.number,
                    'request_id': f'{self.index}:{len(game.history)}',
                    'observation': visible, 'completed_hands': self.pending,
                    'astra_net_chips_so_far': -sum(self.nets),
                    'note': 'Untimed exhibition. No future cards or other-half information supplied.'}
        self.match.save_if_complete()
        return {'finished': True, 'half': self.number, 'hands_played': len(self.nets),
                'completed_hands': self.pending, 'astra_net_chips': -sum(self.nets),
                'bot_net_chips': sum(self.nets), 'decisions': len(self.astra_actions)}

    def act(self, body):
        current = self.state()
        if current['finished']: return current
        if body.get('request_id') != current['request_id']:
            raise HTTPException(409, 'Stale request ID; GET next state.')
        try:
            action = Action[body['action']]
            rationale = str(body.get('rationale', ''))[:300]
            if not rationale: raise ValueError('Short decision reason required')
            if action not in self.game.observation(1-self.bot_seat).legal_actions:
                raise ValueError('Action is not legal')
        except (ValueError, KeyError) as error:
            self.invalid_requests += 1
            raise HTTPException(400, str(error))
        self.pending = []
        self.astra_actions.append({'request_id': current['request_id'],
                                   'observation': current['observation'],
                                   'action': action.name, 'reason': rationale})
        self.game.apply(action)
        return self.state()


class Match:
    def __init__(self, count, output):
        self.count = count
        hidden_seed = secrets.token_hex(32)
        self.deals = [make_deal(hidden_seed, i) for i in range(count)]
        self.players = {secrets.token_urlsafe(18): Half(self, i) for i in (1, 2)}
        self.admin = secrets.token_urlsafe(24)
        self.output = output
        self.saved = False

    def save_if_complete(self):
        if self.saved or not all(len(h.nets)==self.count for h in self.players.values()): return
        halves = list(self.players.values())
        paired = [(x+y)/2 for x,y in zip(halves[0].nets, halves[1].nets)]
        import statistics, math
        mean = statistics.mean(paired)
        se = statistics.stdev(paired)/math.sqrt(len(paired)) if len(paired)>1 else 0.
        report = {'kind': 'untimed_direct_Astra_exhibition', 'astra_model': 'gpt-6-astra',
                  'deals_per_half': self.count, 'total_hands': 2*self.count,
                  'payout_units': 'bb per duplicate deal; positive is bot',
                  'bot_mean_bb_per_deal': mean, 'ci95_normal_small_sample': [mean-1.96*se, mean+1.96*se],
                  'paired_bb_bot': paired,
                  'fairness': 'Separate fresh Astra contexts, private unguessable deal seed, separate endpoints; completed-hand cards only at legal showdown. Full logs saved after both halves.',
                  'time_bank_enforced': False,
                  'policy_parameters': dict(strategy.PARAMS),
                  'astra_helpers': 'Public card strings and exact current made-hand category; no Monte Carlo equity or automated decision policy.',
                  'limitations': 'Small direct-LLM exhibition; not tournament latency, not a reliable ranking or win probability.',
                  'source_sha256': {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
                                    ['agent.py','strategy.py','opponent_model.py','equity_engine.py','equity_network.py',
                                     'equity_network.npz','policy.json','preflop_model.npz','river_solver.py','river_policy.npz']},
                  'halves': [{'half': h.number, 'bot_seat': h.bot_seat, 'bot_nets_chips': h.nets,
                              'bot_net_chips': sum(h.nets), 'astra_net_chips': -sum(h.nets),
                              'astra_decisions': len(h.astra_actions), 'invalid_requests': h.invalid_requests,
                              'bot_calls': h.bot_calls, 'bot_strategy_seconds': h.bot_seconds,
                              'astra_actions': h.astra_actions, 'hands': h.traces} for h in halves]}
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(report,indent=2)+'\n')
        self.saved = True


@app.get('/')
def health():
    return {'ready': True, 'exhibition': 'direct LLM actions, untimed'}


@app.get('/player/{token}/next')
def next_state(token: str):
    if token not in MATCH.players: raise HTTPException(404)
    return MATCH.players[token].state()


@app.post('/player/{token}/action')
def action(token: str, body: dict):
    if token not in MATCH.players: raise HTTPException(404)
    return MATCH.players[token].act(body)


@app.get('/admin/{token}')
def status(token: str):
    if token != MATCH.admin: raise HTTPException(404)
    return {'saved': MATCH.saved, 'halves': [
        {'half': h.number, 'hands': len(h.nets), 'astra_decisions': len(h.astra_actions),
         'bot_net_chips': sum(h.nets), 'invalid_requests': h.invalid_requests}
        for h in MATCH.players.values()]}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deals', type=int, default=20)
    p.add_argument('--port', type=int, default=5028)
    p.add_argument('--output', type=Path, default=ROOT/'results/astra-exhibition.json')
    args=p.parse_args()
    if not 2 <= args.deals <= 100: p.error('Use 2..100 deals per half')
    MATCH = Match(args.deals,args.output)
    print(json.dumps({'port':args.port, 'players':list(MATCH.players), 'admin':MATCH.admin}),flush=True)
    uvicorn.run(app,host='127.0.0.1',port=args.port,access_log=False,log_level='error')
