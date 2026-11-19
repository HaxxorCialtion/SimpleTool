"""Same-model, one-prefill A/B for complete int4 compiled autoregressive decode."""
import argparse,json,time,statistics
from pathlib import Path
import mlx.core as mx
from simpletool.loading import load_model
from simpletool.deferred_model import prepare_deferred_model,CompiledDeferredStep
from simpletool.compiled import CompiledStep
from simpletool.decode import prefill
from simpletool.cache import fork_batch
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--size',choices=('2b','4b'),default='4b');p.add_argument('--output',type=Path,required=True)
p.add_argument('--tokens',type=int,default=96);p.add_argument('--rounds',type=int,default=5);a=p.parse_args()
mx.set_cache_limit(256*1024**2)
model,tokenizer=load_model(ROOT/f'artifacts/{a.size}_int4_compact',kernel='custom',compact_metadata=True)
prepare_deferred_model(model)
packet=json.loads((ROOT/f'reports/swift_{a.size}_packet_20261008.json').read_text());source,pp=prefill(model,packet['prompt_ids'])
heads=packet['heads'];keys=[];runners={};samples={};first={};parity={};reference={}
for interval in [1,2,4,8,16]:
    compiled=CompiledStep(model) if interval==1 else CompiledDeferredStep(model,interval)
    for batch in [1,4]:
        key=f'{interval}.{"single" if batch==1 else "batch4"}';keys.append(key);selected=heads[:batch]
        markers=[packet['marker_ids'][h] for h in selected];assert len({len(m) for m in markers})==1
        caches=fork_batch(source,batch);initial=compiled.initialize(caches,batch,a.tokens+len(markers[0]))
        runners[key]=(compiled,initial,markers);samples[key]=[]
def run(key):
    compiled,states,markers=runners[key];used=0;raw=[[] for _ in markers];start=time.perf_counter()
    for position in range(len(markers[0])):
        prediction,states=compiled.forward(mx.array([[m[position]] for m in markers]),states,mx.array(used,mx.int32));used+=1;mx.eval(prediction,states)
    for i in range(a.tokens):
        chosen=prediction.tolist()
        for row,token in enumerate(chosen):raw[row].append(token)
        if i+1<a.tokens:
            prediction,states=compiled.forward(mx.array(chosen)[:,None],states,mx.array(used,mx.int32));used+=1;mx.eval(prediction,states)
    return time.perf_counter()-start,raw
for key in keys:
    seconds,raw=run(key);first[key]=seconds;mode=key.split('.')[1]
    if key.startswith('1.'):reference[mode]=raw
    parity[key]=raw==reference[mode];print('warm',key,seconds,parity[key],flush=True)
for round_index in range(a.rounds):
    shift=round_index%len(keys)
    for key in keys[shift:]+keys[:shift]:
        seconds,raw=run(key);samples[key].append(seconds);parity[key]&=raw==reference[key.split('.')[1]]
    print('round',round_index+1,flush=True)
med={k:statistics.median(v) for k,v in samples.items()}
result=dict(size=a.size,median_s=med,samples_s=samples,first_run_s=first,token_equal_to_baseline=parity,
    efficiency={str(n):med[f'{n}.single']/med[f'{n}.batch4'] for n in [1,2,4,8,16]},
    weight_bits=4,compute='float32',prefill_count=1,prefill_s=pp,peak_bytes=mx.get_peak_memory(),fixed_tokens_each_head=a.tokens,
    scope='one model and common prefix, 10 runners; rotating five warm rounds including all complete marker positions',
    limitations='stops ignored; peak includes all initial runner state; Python MLX runtime, not Swift/iPhone; uncontrolled competing load')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(med,flush=True)
if not all(parity.values()):raise SystemExit(2)
