"""Real int4 checkpoint token and canonical full-cache parity across flushes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mlx.core as mx
from simpletool.loading import load_model
from simpletool.deferred_model import prepare_deferred_model
from simpletool.fixed_history_model import CompiledFixedHistoryStep
from simpletool.fixed_history_gdn import FixedHistoryState
from simpletool.compiled import CompiledStep
from simpletool.decode import prefill
from simpletool.cache import fork_batch,cache_arrays

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--size',choices=('2b','4b'),default='4b');p.add_argument('--fused',action='store_true');p.add_argument('--dispatch',choices=('baseline','head','value'),default='baseline');a=p.parse_args()
from simpletool.locality_gdn import STATES
State=STATES[a.dispatch]
mx.set_cache_limit(256*1024**2)
m,t=load_model(ROOT/f'artifacts/{a.size}_int4_compact',kernel='custom',compact_metadata=True);prepare_deferred_model(m)
packet=json.loads((ROOT/f'reports/swift_{a.size}_packet_20261008.json').read_text());source,_=prefill(m,packet['prompt_ids'])
def digest():return [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in cache_arrays(source)]
before=digest();rows=[];out=ROOT/f'reports/{a.dispatch+"_" if a.dispatch!="baseline" else ""}fixed_history_state_{a.size}{"_fused" if a.fused else ""}_20261008.json'
def run(compiled,batch,chunk,interval):
    heads=packet['heads'][:batch];markers=[packet['marker_ids'][h] for h in heads]
    count=19;capacity=count+len(markers[0])-1
    states=compiled.initialize(fork_batch(source,batch),batch,capacity);used=0
    for j in range(len(markers[0])):
        pred,states=compiled.forward(mx.array([[v[j]] for v in markers]),states,mx.array(used,mx.int32))
        used+=1;mx.eval(pred,states)
    raw=[];generated=0
    while generated<count:
        n=min(chunk,count-generated)
        block,states=compiled.advance_pending(pred,states,mx.array(used,mx.int32),n);used+=n-1
        mx.eval(block,states);values=np.asarray(block);raw.extend(values.T.reshape(-1).tolist());generated+=n
        if generated<count:
            pred,states=compiled.forward(block[:,-1,None],states,mx.array(used,mx.int32));used+=1
    canonical=[]
    for layer,state in zip(m.layers,states):
        if len(state)==4:canonical.extend(state)
        elif interval==1:canonical.extend(state)
        else:
            d=State(state[1],batch,layer.linear_attn.num_k_heads,interval,used%interval)
            d.keys,d.gates,d.deltas=state[2:]
            canonical.extend([state[0],d.materialize()])
    mx.eval(canonical)
    return raw,canonical,used,capacity
for batch in range(1,5):
    reference=run(CompiledStep(m),batch,1,1)
    for interval in (2,4,8,16):
        for chunk in ((1,2,4,8) if interval in (2,16) else (1,8)):
            c=CompiledFixedHistoryStep(m,interval,fused=a.fused,state_type=State);result=run(c,batch,chunk,interval)
            different=[i for i,(x,y) in enumerate(zip(reference[1],result[1]))
                       if x.shape!=y.shape or x.dtype!=y.dtype or not np.array_equal(np.asarray(x).view(np.uint32),np.asarray(y).view(np.uint32))]
            equal=result[0]==reference[0];passed=equal and not different and result[2]==result[3]
            rows.append(dict(batch=batch,interval=interval,chunk=chunk,token_equal=equal,
                             different_state_indices=different,state_bitwise_equal=not different,
                             used=result[2],capacity=result[3],passed=passed))
            out.write_text(json.dumps(dict(cases=rows,source_unchanged=before==digest(),all_passed=all(r['passed'] for r in rows)),indent=2)+'\n')
            print(a.size,batch,interval,chunk,passed,flush=True)
            if not passed:raise SystemExit(2)
assert before==digest()
