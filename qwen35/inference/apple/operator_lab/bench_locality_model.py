"""Rotating complete-model A/B: shared seed dispatch order vs identical fixed/fused history."""
import argparse
import json
import statistics
import time
from pathlib import Path
import mlx.core as mx
from simpletool.loading import load_model
from simpletool.deferred_model import prepare_deferred_model,CompiledDeferredStep
from simpletool.fixed_history_model import CompiledFixedHistoryStep
from simpletool.locality_gdn import STATES
from simpletool.compiled import CompiledStep
from simpletool.decode import prefill
from simpletool.cache import fork_batch

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--size',choices=('2b','4b'),default='4b')
p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,default=96)
p.add_argument('--rounds',type=int,default=5);a=p.parse_args()
mx.set_cache_limit(256*1024**2)
m,t=load_model(ROOT/f'artifacts/{a.size}_int4_compact',kernel='custom',compact_metadata=True);prepare_deferred_model(m)
packet=json.loads((ROOT/f'reports/swift_{a.size}_packet_20261008.json').read_text());source,pp=prefill(m,packet['prompt_ids'])
policies=[('baseline',1),('baseline',4),('head',4),('value',4),('baseline',16),('head',16),('value',16)]
runners={};keys=[];samples={};first={};reference={};parity={};signatures={}
for policy,interval in policies:
 for batch in (1,4):
    key=f'{policy}{interval}.{"single" if batch==1 else "batch4"}';keys.append(key)
    c=CompiledStep(m) if interval==1 else CompiledFixedHistoryStep(m,interval,fused=True,state_type=STATES[policy])
    markers=[packet['marker_ids'][h] for h in packet['heads'][:batch]]
    initial=c.initialize(fork_batch(source,batch),batch,a.tokens+len(markers[0]))
    runners[key]=(c,initial,markers);samples[key]=[];signatures[key]=set()
def run(key):
    c,states,markers=runners[key];used=0;raw=[[] for _ in markers];start=time.perf_counter()
    def forward(tokens):
        nonlocal used,states
        signatures[key].add(tuple(tuple(tuple(x.shape) for x in s) for s in states))
        pred,states=c.forward(tokens,states,mx.array(used,mx.int32));used+=1;mx.eval(pred,states)
        return pred
    for pos in range(len(markers[0])):prediction=forward(mx.array([[v[pos]] for v in markers]))
    for i in range(a.tokens):
        chosen=prediction.tolist()
        for row,token in enumerate(chosen):raw[row].append(token)
        if i+1<a.tokens:prediction=forward(mx.array(chosen)[:,None])
    return time.perf_counter()-start,raw
for key in keys:
    seconds,raw=run(key);first[key]=seconds;mode=key.split('.')[1]
    if key.startswith('baseline1.'):reference[mode]=raw
    parity[key]=raw==reference[mode];print('first',key,seconds,parity[key],flush=True)
for i in range(a.rounds):
    shift=(3*i)%len(keys)
    for key in keys[shift:]+keys[:shift]:
        seconds,raw=run(key);samples[key].append(seconds);parity[key]&=raw==reference[key.split('.')[1]]
    print('round',i+1,flush=True)
med={k:statistics.median(v) for k,v in samples.items()}
signature_counts={k:len(v) for k,v in signatures.items()}
function_signature_counts={k:(len(runners[k][0].input_signatures) if not k.startswith('baseline1.') else signature_counts[k]) for k in keys}
out=dict(size=a.size,median_s=med,samples_s=samples,first_run_s=first,token_equal_to_baseline=parity,
    observed_state_shape_signatures=signature_counts,observed_shape_plus_write_mode_signatures=function_signature_counts,
    fixed_tokens_each_head=a.tokens,weight_bits=4,compute='float32',prefill_count=1,prefill_s=pp,
    peak_bytes=mx.get_peak_memory(),
    scope='one int4 weight set, one prefix, dense baseline plus fixed/fused I4/I16 baseline/head/value dispatch; 14 runners; rotating warm rounds, all complete marker positions',
    limits='Observed input signatures are not internal compiler or GPU pipeline counters; ignore stops; first-run comparisons share prewarm; peak includes all runners; external load uncontrolled; not Swift/iPhone')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(med,signature_counts,function_signature_counts,flush=True)
if not all(parity.values()):raise SystemExit(2)
