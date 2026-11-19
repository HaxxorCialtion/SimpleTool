"""Replay-state shrinking/reordering plus edge gates, checked at every step."""
import json,hashlib
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel,normalize_qk
from simpletool.deferred_gdn import DeferredState
ROOT=Path(__file__).resolve().parent;mx.random.seed(20261009);cases=[];receipts=[]
def equal(a,b):return mx.array_equal(a.view(mx.uint32),b.view(mx.uint32)).item()
for limit in [2,4,8,16]:
    for shared in [False,True]:
        batch=4;hk=16;hv=32;dv=128;seed=mx.random.normal((1 if shared else batch,hv,dv,128))*.01
        before=hashlib.sha256(np.asarray(seed).tobytes()).hexdigest()
        reference=mx.broadcast_to(seed,(batch,hv,dv,128));d=DeferredState(seed,batch,hk,limit)
        for step in range(35):
            q,k=normalize_qk(mx.random.normal((batch,1,hk,128)),mx.random.normal((batch,1,hk,128)),inv_scale=128**-.5,eps=1e-6)
            v=mx.random.normal((batch,1,hv,dv))*.1
            g=mx.broadcast_to(mx.array([0.,1.,2.**-20,1e-30]*8)[None,None,:],(batch,1,hv))
            beta=mx.broadcast_to(mx.array([0.,1.,.5,1.]*8)[None,None,:],(batch,1,hv))
            y,reference=gated_delta_kernel(q,k,v,g,beta,reference);output=d.step(q,k,v,g,beta);dense=d.materialize()
            mx.eval(y,reference,output,dense,d.arrays);passed=equal(y,output) and equal(reference,dense)
            cases.append(dict(limit=limit,shared=shared,step=step,batch=batch,pending=d.pending,passed=passed))
            if not passed:
                (ROOT/'reports/deferred_gdn_filter_20261008.json').write_text(json.dumps(dict(cases=cases,all_passed=False),indent=2)+'\n')
                raise SystemExit(2)
            indices={0:[3,1,0],2:[2,0],8:[1]}.get(step)
            if indices is not None:
                reference=mx.take(reference,mx.array(indices),axis=0);d.filter(mx.array(indices));batch=len(indices)
        unchanged=before==hashlib.sha256(np.asarray(seed).tobytes()).hexdigest();assert unchanged
        receipts.append(dict(limit=limit,shared=shared,seed_unchanged=unchanged))
        (ROOT/'reports/deferred_gdn_filter_20261008.json').write_text(json.dumps(dict(cases=cases,receipts=receipts,all_passed=True,
            scope='B4 to B3 to B2 to B1 reorder/shrink; zero/one/tiny decay and beta; bitwise y and full matrix'),indent=2)+'\n')
print(len(cases),'edge/filter cases passed')
