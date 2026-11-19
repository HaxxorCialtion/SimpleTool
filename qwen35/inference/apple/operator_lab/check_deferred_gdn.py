"""Compare every readout and reconstructed FP32 matrix with installed packed GDN."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx_lm.models.gated_delta import gated_delta_kernel,normalize_qk
from simpletool.deferred_gdn import DeferredState
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'reports/deferred_gdn_regression_20261008.json')
a=p.parse_args();mx.random.seed(20261008);cases=[];receipts=[]
def save():
    a.output.write_text(json.dumps(dict(cases=cases,seed_receipts=receipts,all_passed=all(c['passed'] for c in cases),
        scope='FP32 scalar-gate packed GDN; every step including replay and rebase; no full-model claim'),indent=2)+'\n')
def equal(a,b):return mx.array_equal(a.view(mx.uint32),b.view(mx.uint32)).item()
def digest(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()
shapes=[(8,16,128)] if a.quick else [(16,32,128),(16,16,128),(8,16,64),(8,16,256),(1,1,8)]
for hk,hv,dv in shapes:
    for batch in ([4] if a.quick else [1,2,3,4]):
        for limit in ([4] if a.quick else [1,2,4,8,16]):
            for shared in ([True] if a.quick else [False,True]):
                seed=mx.random.normal((1 if shared else batch,hv,dv,128))*.01
                before=digest(seed);reference=mx.broadcast_to(seed,(batch,hv,dv,128));candidate=DeferredState(seed,batch,hk,limit)
                steps=9 if a.quick else 2*limit+3
                for step in range(steps):
                    q,k=normalize_qk(mx.random.normal((batch,1,hk,128)),mx.random.normal((batch,1,hk,128)),inv_scale=128**-.5,eps=1e-6)
                    v=mx.random.normal((batch,1,hv,dv))*.1;g=mx.exp(-mx.random.uniform(.01,1,(batch,1,hv)));beta=mx.sigmoid(mx.random.normal((batch,1,hv)))
                    y,reference=gated_delta_kernel(q,k,v,g,beta,reference)
                    output=candidate.step(q,k,v,g,beta);dense=candidate.materialize();mx.eval(y,reference,output,dense,candidate.arrays)
                    passed=equal(y,output) and equal(reference,dense)
                    cases.append(dict(hk=hk,hv=hv,dv=dv,batch=batch,limit=limit,shared_seed=shared,step=step,
                        pending=candidate.pending,y_bitwise_equal=equal(y,output),state_bitwise_equal=equal(reference,dense),
                        max_abs_y=float(mx.max(mx.abs(y-output)).item()),max_abs_state=float(mx.max(mx.abs(reference-dense)).item()),passed=passed))
                    if not passed:
                        save();print(cases[-1],flush=True);raise SystemExit(2)
                unchanged=before==digest(seed);receipts.append(dict(hk=hk,hv=hv,dv=dv,batch=batch,limit=limit,shared=shared,unchanged=unchanged,sha256=before))
                save();assert unchanged,'immutable seed changed'
print(len(cases),'bitwise output and full-state cases passed',flush=True)
