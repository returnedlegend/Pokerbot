"""Replay and audit a completed direct-LLM exhibition, without hidden state."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluator import cards_to_str,category_name
from equity_engine import score_hand
from game import Action,AgentConfig,Deal,Game,button_for_deal
import strategy


def audit(path, doc=None):
    data=json.loads(path.read_text())
    count=data['deals_per_half']
    totals={'bot':Counter(),'astra':Counter()}
    samples=[]
    for half in data['halves']:
        memory={}
        strategy.new_half(AgentConfig(num_hands=count),memory)
        decisions={x['request_id']:x for x in half['astra_actions']}
        assert len(decisions)==half['astra_decisions']
        used=set()
        assert len(half['hands'])==count
        for hand in half['hands']:
            index=hand['hand_index']
            d=hand['deal']
            deal=Deal(tuple(tuple(x) for x in d['hole_cards']),tuple(d['board']))
            game=Game(deal,button_for_deal(index),index)
            for record in hand['result']['history']:
                assert game.to_act==record['seat']
                obs=game.observation(game.to_act)
                choice=Action(record['action'])
                assert choice in obs.legal_actions
                if game.to_act==half['bot_seat']:
                    assert strategy.choose_action(obs,memory)==choice,'Recorded bot action differs from unchanged policy'
                    totals['bot'][choice.name]+=1
                else:
                    request_id=f'{index}:{len(game.history)}'
                    supplied=decisions[request_id]
                    visible=supplied['observation']
                    for key,value in obs.to_dict().items():
                        assert visible[key]==value,(request_id,key)
                    assert visible['hole_text']==cards_to_str(obs.hole_cards)
                    assert visible['board_text']==cards_to_str(obs.board)
                    if len(obs.board)>=3:
                        assert visible['current_made_hand_category']==category_name(score_hand(obs.hole_cards,obs.board))
                    assert set(visible)==set(obs.to_dict())|{'hole_text','board_text'}|({'current_made_hand_category'} if len(obs.board)>=3 else set())
                    assert supplied['action']==choice.name and supplied['reason']
                    used.add(request_id)
                    totals['astra'][choice.name]+=1
                game.apply(choice)
            assert game.is_terminal
            result=game.result()
            assert list(result.nets)==hand['result']['nets']
            assert result.pot==hand['result']['pot']
            assert result.nets[half['bot_seat']]==half['bot_nets_chips'][index]
            assert sum(result.nets)==0
            strategy.on_hand_end(game.hand_result(half['bot_seat']),memory)
            samples.append({'half':half['half'],'hand_index':index,
                            'bot_net_chips':result.nets[half['bot_seat']],
                            'pot':result.pot,'showdown':result.went_to_showdown})
        assert used==set(decisions)
    for name,digest in data['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    bot_net=sum(h['bot_net_chips'] for h in data['halves'])
    astra_net=sum(h['astra_net_chips'] for h in data['halves'])
    assert bot_net+astra_net==0
    assert bot_net==sum(data['paired_bb_bot'])*2
    summary={'opponent_model':data['astra_model'],'hands':data['total_hands'],
             'paired_deals':count,'bot_net_chips':bot_net,'astra_net_chips':astra_net,
             'bot_bb_per_duplicate_deal':data['bot_mean_bb_per_deal'],
             'approximate_ci95':data['ci95_normal_small_sample'],
             'astra_decisions':sum(h['astra_decisions'] for h in data['halves']),
             'submitted_invalid_actions':sum(h['invalid_requests'] for h in data['halves']),
             'actions':{k:dict(v) for k,v in totals.items()},
             'replay_audit_passed':True,'every_public_observation_verified':True,
             'every_bot_action_reproduced':True,'time_bank_enforced':False,
             'showdown_hands':sum(x['showdown'] for x in samples),
             'mean_pot_chips':statistics.mean(x['pot'] for x in samples),
             'limitations':data['limitations'],
             'largest_pots':sorted(samples,key=lambda x:x['pot'],reverse=True)[:5]}
    out=path.with_name('astra-exhibition-summary.json')
    out.write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Bot versus direct Astra decisions','',
           'This is an untimed exhibition against actual `gpt-6-astra` model agents,',
           'not another scripted poker bot. Two fresh isolated contexts played',
           f'opposite seats on {count} duplicate deals ({2*count} actual hands). Each model',
           'selected every action itself through a private observation endpoint.',
           'No automated betting policy or Monte Carlo equity tool was supplied to',
           'Astra. A current-made-hand category helper used visible cards only.', '',
           f'**Bot: {bot_net:+d} chips. Astra: {astra_net:+d} chips.**',
           f"Bot average: {data['bot_mean_bb_per_deal']:+.3f} bb per duplicate deal.",
           f"Approximate small-sample 95% interval: {summary['approximate_ci95']}.", '',
           'The tournament time bank was deliberately not enforced: language-model',
           'decision latency is incompatible with 8 seconds per 1,000 hands. This',
           'small strategic exhibition cannot establish a reliable ranking or',
           'tournament win probability. Do not conflate the chip lead with strong',
           'statistical evidence of superiority.', '',
           f"Astra made {summary['astra_decisions']} decisions. Invalid submitted actions: {summary['submitted_invalid_actions']}.",
           'The replay audit reproduced every bot action from the unchanged policy,',
           'verified every Astra observation against the official game state,',
           'validated legal actions and zero-sum payouts, and checked that the',
           'runtime source hashes remained unchanged. Neither half could access',
           "the other half's endpoint; full trace files were saved only after",
           'both halves finished. The private deal seed was not exposed.', '',
           'Evidence: `results/astra-exhibition.json` and',
           '`results/astra-exhibition-summary.json`. No bot training or tuning',
           'was performed during this experiment.', '']
    if doc is not None:
        Path(doc).write_text('\n'.join(lines))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('result',nargs='?',type=Path,default=ROOT/'results/astra-exhibition.json')
    p.add_argument('--doc',type=Path,help='Optional Markdown output; omit to preserve historical reports')
    args=p.parse_args()
    print(json.dumps(audit(args.result,args.doc),indent=2))
