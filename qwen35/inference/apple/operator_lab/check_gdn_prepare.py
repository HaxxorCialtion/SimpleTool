"""Bitwise fused GDN preparation vs actual MLX conv/SiLU/RMS formulas."""
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import normalize_qk
from simpletool.gdn_prepare import prepare

ROOT=Path(__file__).resolve().parent
rng=np.random.default_rng(920073)
cases=[]


def digest(arrays):
    return [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in arrays]


def reference(history, qkv, weight, hk, hv):
    B,_,D=qkv.shape
    full=mx.concatenate([history,qkv],axis=1)
    conv=nn.silu(mx.conv1d(full,weight,groups=D))
    q,k,v=[t.reshape(B,1,h,128) for t,h in zip(mx.split(conv,[hk*128,hk*256],axis=-1),(hk,hk,hv))]
    q,k=normalize_qk(q,k,inv_scale=128**-.5,eps=1e-6)
    return q,k,v,mx.contiguous(full[:,-3:,:])


for hk,hv in [(1,1),(1,2),(8,16),(16,16),(16,32)]:
    D=(2*hk+hv)*128
    for B in (1,2,3,4):
        for scale in (0.,.001,1.,20.):
            for strided in (False,True):
                raw=rng.standard_normal((B,4,D*2 if strided else D)).astype(np.float32)*scale
                raw[:,:,::17]=-0.0
                full=mx.array(raw)[:,:,::2] if strided else mx.array(raw)
                history,qkv=full[:,:3],full[:,3:]
                weight=mx.array(rng.standard_normal((D,4,1)).astype(np.float32)*.2)
                mx.eval(history,qkv,weight)
                before=digest([history,qkv,weight])
                expected=reference(history,qkv,weight,hk,hv)
                actual=prepare(history,qkv,weight,hk,hv)
                mx.eval(expected,actual)
                equal=[mx.array_equal(x.view(mx.uint32),y.view(mx.uint32)).item() for x,y in zip(actual,expected)]
                errors=[mx.max(mx.abs(x-y)).item() for x,y in zip(actual,expected)]
                immutable=before==digest([history,qkv,weight])
                cases.append(dict(hk=hk,hv=hv,batch=B,scale=scale,strided=strided,bitwise_equal=equal,max_errors=errors,inputs_unchanged=immutable))
                assert immutable
    print(hk,hv,'mismatch configurations',sum(not all(r['bitwise_equal']) for r in cases),flush=True)
result=dict(cases=cases,all_passed=all(all(r['bitwise_equal']) for r in cases),compute='float32',
            scope='Conv/SiLU/QK RMS/history bitwise; finite seeded varied ranges, signed zero, strided inputs; not full-model or quality acceptance')
(ROOT/'reports/gdn_prepare_operator_20261008.json').write_text(json.dumps(result,indent=2)+'\n')
print(len(cases),'configurations; bitwise all',result['all_passed'],flush=True)
if not result['all_passed']:raise SystemExit(2)
