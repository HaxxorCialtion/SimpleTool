"""Real instruction parity for the explicit full-model deferred-state candidate."""
import argparse,json,hashlib
from pathlib import Path
import mlx.core as mx
from simpletool.loading import load_model
from simpletool.deferred_model import prepare_deferred_model
from simpletool.fixed_history_model import CompiledFixedHistoryStep
from simpletool.decode import prefill,batched_generate
from simpletool.cache import cache_arrays
import numpy as np
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--size',choices=('2b','4b'),default='4b');p.add_argument('--interval',type=int,default=4)
p.add_argument('--cases',type=int,default=5);p.add_argument('--chunk',type=int,default=1);p.add_argument('--fused',action='store_true');p.add_argument('--dispatch',choices=('baseline','head','value'),default='baseline');a=p.parse_args()
from simpletool.locality_gdn import STATES
State=STATES[a.dispatch]
mx.set_cache_limit(256*1024**2)
model,tokenizer=load_model(ROOT/f'artifacts/{a.size}_int4_compact',kernel='custom',compact_metadata=True)
prepare_deferred_model(model);compiled=CompiledFixedHistoryStep(model,a.interval,fused=a.fused,state_type=State);rows=[]
stem=f'{a.dispatch+"_" if a.dispatch!="baseline" else ""}fixed_history_model_{a.size}_i{a.interval}_c{a.chunk}{"_fused" if a.fused else ""}_20261008'
def digest(arrays):return [hashlib.sha256(np.asarray(t).tobytes()).hexdigest() for t in arrays]
for i in range(a.cases):
    path=ROOT/f'reports/swift_suite_{a.size}_{i}_20261008.packet.json';packet=json.loads(path.read_text())
    source,_=prefill(model,packet['prompt_ids']);before=digest(cache_arrays(source))
    results={};equal={}
    for mode in ['serial','batch4']:
        groups=[[h] for h in packet['heads']] if mode=='serial' else [packet['heads']]
        channels={};traces=[];stats=[]
        for heads in groups:
            result=batched_generate(model,tokenizer,packet['prompt_ids'],packet['max_tokens'],heads=heads,
                prefix_cache=source,compiled_step=compiled,decode_chunk_size=a.chunk)
            channels.update(result['channels']);traces+=result['stats']['active_batch_by_step'];stats.append(result['stats'])
        results[mode]=dict(channels=channels,active_batch_by_step=traces,stats=stats)
        equal[mode]={h:channels[h]['token_ids']==packet['gold'][h] for h in packet['heads']}
    immutable=before==digest(cache_arrays(source));passed=immutable and all(v for vals in equal.values() for v in vals.values())
    out=ROOT/f'reports/{stem}_case{i}.json';out.write_text(json.dumps(dict(results=results,gold_token_equal=equal,
        prefix_unchanged=immutable,weight_bits=4,compute='float32',interval=a.interval,chunk=a.chunk,input_shape_signatures=len(compiled.input_signatures)),ensure_ascii=False,indent=2)+'\n')
    rows.append(dict(case=i,result=str(out.relative_to(ROOT)),passed=passed))
    (ROOT/f'reports/{stem}.json').write_text(json.dumps(rows,indent=2)+'\n');print(a.size,a.interval,a.chunk,i,passed,equal,flush=True)
    if not passed:raise SystemExit(2)
