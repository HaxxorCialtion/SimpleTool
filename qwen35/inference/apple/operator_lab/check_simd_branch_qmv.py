"""Bitwise/immutability validation, including ragged rows and noncontiguous x."""
import json,hashlib
from pathlib import Path
import numpy as np
import mlx.core as mx
from simpletool.simd_branch_qmv import simd_branch_qmv
from simpletool.stream_grouped_qmv import stream_grouped_qmv
ROOT=Path(__file__).resolve().parent
mx.random.seed(20261008);cases=[]
def digest(array):return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()
for n,k in [(79,512),(256,2048),(512,2560),(8,512)]:
    w,s,b=mx.quantize(mx.random.normal((n,k)),bits=4,group_size=64)
    for dtype in [mx.float16,mx.float32]:
        scales=s.astype(dtype);biases=b.astype(dtype)
        for batch in [1,2,3,4]:
            # Some exact-zero inputs exercise signed-zero paths.
            backing=mx.where(mx.arange(k*2)%6==0,mx.array(-0.0),mx.random.normal((batch,k*2)))
            x=backing[:,::2]
            inputs=[x,w,scales,biases];mx.eval(inputs);before=[digest(a) for a in inputs]
            gold=mx.concatenate([stream_grouped_qmv(x[i:i+1],w,scales,biases) for i in range(batch)])
            for rows in [1,2,4,8]:
                for shared in [False,True]:
                    y=simd_branch_qmv(x,w,scales,biases,rows,shared)
                    equal=mx.array_equal(y.view(mx.uint32),gold.view(mx.uint32)).item()
                    cases.append(dict(N=n,K=k,batch=batch,rows=rows,shared=shared,metadata=str(dtype),bitwise_equal=equal))
            after=[digest(a) for a in inputs]
            assert before==after,'inputs changed'
out=dict(cases=cases,all_passed=all(c['bitwise_equal'] for c in cases),input_digests_unchanged=True,
         weight_bits=4,compute='float32',scope='synthetic affine int4 fixtures; exact independent single-vector reference')
(ROOT/'reports/simd_branch_qmv_regression_20261008.json').write_text(json.dumps(out,indent=2)+'\n')
print(len(cases),out['all_passed'])
if not out['all_passed']:raise SystemExit(2)
