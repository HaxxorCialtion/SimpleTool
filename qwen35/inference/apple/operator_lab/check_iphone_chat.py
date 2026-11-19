"""Same-checkpoint Python reference for the App's actual native multi-turn path."""
import argparse,json
from pathlib import Path
import mlx.core as mx
from simpletool.loading import load_model
from simpletool.decode import HEADS,prompt_ids,prefill,batched_generate
from simpletool.stream_grouped_qmv import install_stream_grouped_qmv
from simpletool.fused_swiglu_int4 import install_fused_swiglu_int4
from simpletool.compiled import CompiledStep

p=argparse.ArgumentParser();p.add_argument('--model',choices=('2b','4b'),required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
fixture=json.loads((root/'reports/iphone_chat_fixture_20261008.json').read_text())
native=json.loads((root/f'reports/iphone_chat_{a.model}_native_20261008.json').read_text())
mx.set_cache_limit(128*1024**2)
model,tok=load_model(root/f'artifacts/{a.model}_int4_compact',kernel='custom',compact_metadata=True)
install_stream_grouped_qmv(model);install_fused_swiglu_int4(model)
step=CompiledStep(model)
messages=[{'role':'system','content':fixture['system']}]
checks=[]
for user,actual in zip(fixture['inputs'],native):
 messages.append({'role':'user','content':user})
 ids=prompt_ids(tok,messages);cache,_=prefill(model,ids,chunk_size=64)
 result=batched_generate(model,tok,ids,48,prefix_cache=cache,compiled_step=step)
 tokens={h:c['token_ids'] for h,c in result['channels'].items()}
 texts={h:c['text'] for h,c in result['channels'].items()}
 parity={h:tokens[h]==actual['tokenIDs'][h] for h in HEADS}
 checks.append(dict(user=user,prompt_token_count=len(ids),native_cache_hit_tokens=actual['cacheHitTokens'],
                    token_parity=parity,python_token_ids=tokens,python_text=texts,native_stops=actual['stops']))
 messages.append({'role':'assistant','content':''.join(f'<{h}>{texts[h]}</{h}>' for h in HEADS)})
 print(a.model,user,parity,flush=True)
receipt=dict(model=a.model,checks=checks,all_passed=all(all(c['token_parity'].values()) for c in checks),
             reference='Full recompute at each turn with matching compact int4 weights; native App uses incremental prefix state.')
(root/f'reports/iphone_chat_{a.model}_parity_20261008.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
assert receipt['all_passed']
