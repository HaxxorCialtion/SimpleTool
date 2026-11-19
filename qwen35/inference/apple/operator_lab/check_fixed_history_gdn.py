"""Bitwise recurrence, poisoned unused slots, and branch-filter regression."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel
from simpletool.fixed_history_gdn import FixedHistoryState

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');p.add_argument('--fused',action='store_true');p.add_argument('--dispatch',choices=('baseline','head','value'),default='baseline');a=p.parse_args()
from simpletool.locality_gdn import STATES
State=STATES[a.dispatch]
name='fixed_history_gdn_quick' if a.quick else 'fixed_history_gdn_regression'
if a.fused:name+='_fused'
if a.dispatch!='baseline':name+='_'+a.dispatch
output=ROOT/f'reports/{name}_20261008.json';rows=[];receipts=[]
def bits(x,y):return np.array_equal(np.asarray(x).view(np.uint32),np.asarray(y).view(np.uint32))
def digest(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
def save():output.write_text(json.dumps(dict(cases=rows,receipts=receipts,all_passed=all(r['passed'] for r in rows)),indent=2)+'\n')
rng=np.random.default_rng(29314)
def random(shape):return mx.array(rng.standard_normal(shape).astype(np.float32))
def inputs(B,Hk,Hv,Dv,edge=False):
    q=random((B,1,Hk,128));q=q*mx.rsqrt(mx.sum(q*q,axis=-1,keepdims=True)+1e-6)*128**-.5
    k=random(q.shape);k=k*mx.rsqrt(mx.sum(k*k,axis=-1,keepdims=True)+1e-6)
    v=random((B,1,Hv,Dv))*.1
    if edge:
        g=mx.broadcast_to(mx.array([0.,1.,2**-20,1e-30],mx.float32)[mx.arange(Hv)%4],(B,1,Hv))
        beta=mx.broadcast_to(mx.array([0.,1.,.5,1.],mx.float32)[mx.arange(Hv)%4],(B,1,Hv))
    else:g=mx.exp(-mx.abs(random((B,1,Hv))));beta=mx.sigmoid(random(g.shape))
    return q,k,v,g,beta
shapes=[(16,32,128)] if a.quick else [(16,32,128),(16,16,128),(8,16,64),(8,16,256),(1,1,8)]
for Hk,Hv,Dv in shapes:
 for B in ([4] if a.quick else range(1,5)):
  for limit in ([4] if a.quick else (2,4,8,16)):
   for shared in ([True] if a.quick else (False,True)):
    base=random((1 if shared else B,Hv,Dv,128))*.01;before=digest(base)
    candidate=State(base,B,Hk,limit,fused=a.fused)
    # NaN future slots make accidental replay beyond the dynamic count visible.
    candidate.keys=mx.full(candidate.keys.shape,mx.nan,mx.float32)
    candidate.gates=mx.full(candidate.gates.shape,mx.nan,mx.float32)
    candidate.deltas=mx.full(candidate.deltas.shape,mx.nan,mx.float32)
    reference=mx.broadcast_to(base,(B,Hv,Dv,128))
    for step in range(2*limit+3):
        x=inputs(B,Hk,Hv,Dv)
        y,reference=gated_delta_kernel(*x,reference)
        cy=candidate.step(*x);dense=candidate.materialize();mx.eval(y,reference,cy,dense,candidate.arrays)
        equal_y,equal_state=bits(y,cy),bits(reference,dense)
        rows.append(dict(hk=Hk,hv=Hv,dv=Dv,batch=B,interval=limit,shared=shared,step=step,
                         y_bitwise_equal=equal_y,state_bitwise_equal=equal_state,passed=equal_y and equal_state))
        if not rows[-1]['passed']:save();raise SystemExit(2)
    receipts.append(dict(hk=Hk,hv=Hv,dv=Dv,batch=B,interval=limit,shared=shared,seed_unchanged=before==digest(base)))
    save()
if not a.quick:
 for limit in (2,4,8,16):
  for shared in (False,True):
    B=4;Hk=16;Hv=32;Dv=128
    base=random((1 if shared else B,Hv,Dv,128))*.01;before=digest(base)
    candidate=State(base,B,Hk,limit,fused=a.fused);reference=mx.broadcast_to(base,(B,Hv,Dv,128))
    for step in range(35):
        x=inputs(B,Hk,Hv,Dv,True);y,reference=gated_delta_kernel(*x,reference)
        cy=candidate.step(*x);dense=candidate.materialize();mx.eval(y,reference,cy,dense)
        equal=bits(y,cy) and bits(reference,dense)
        rows.append(dict(filter_case=True,interval=limit,shared=shared,batch=B,step=step,passed=equal))
        if not equal:save();raise SystemExit(2)
        indices={0:[3,1,0],2:[2,0],8:[1]}.get(step)
        if indices is not None:
            idx=mx.array(indices,mx.int32);candidate.filter(idx);reference=mx.take(reference,idx,axis=0);B=len(indices)
    receipts.append(dict(filter_case=True,interval=limit,shared=shared,seed_unchanged=before==digest(base)))
    save()
assert all(r['seed_unchanged'] for r in receipts)
print(len(rows),'fixed-history output/matrix cases bitwise equal; poisoned unused slots and source hashes passed')
