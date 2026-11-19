"""Control bitwise regression; finite unroll error compared with FP64 dot reference."""
import hashlib,json
from pathlib import Path
import numpy as np
import mlx.core as mx
from simpletool.unroll_int4 import unroll_int4 as fma_int4,MODES
from simpletool.row_reuse_int4 import row_reuse_int4
ROOT=Path(__file__).resolve().parent;rng=np.random.default_rng(83170);rows=[]
def digest(xs):return [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in xs]
for k,n in [(64,79),(512,257),(2048,81),(2560,256),(9216,17)]:
 for dtype in (mx.float16,mx.float32):
  weight=mx.array(rng.standard_normal((n,k)).astype(np.float32)*.02)
  w,s,b=mx.quantize(weight,group_size=64,bits=4);s=s.astype(dtype);b=b.astype(dtype);mx.eval(w,s,b)
  # FP64 dot over the exact separately-rounded FP32 reconstructed weights.
  unpack=mx.dequantize(w,s.astype(mx.float32),b.astype(mx.float32),group_size=64,bits=4)
  exact_weights=np.asarray(unpack).astype(np.float64)
  for batch in (1,2,3,4):
   for strided in (False,True):
    array=rng.standard_normal((batch,k*2 if strided else k)).astype(np.float32)*.3
    x=mx.array(array)[:,::2] if strided else mx.array(array);mx.eval(x)
    before=digest([x,w,s,b]);target=np.asarray(x).astype(np.float64)@exact_weights.T
    scale=np.abs(np.asarray(x).astype(np.float64))@np.abs(exact_weights.T)
    for tile,threads in [(1,64),(2,128),(4,128),(8,256)]:
     ref=row_reuse_int4(x,w,s,b,tile,threads);mx.eval(ref)
     for mode in MODES:
      y=fma_int4(x,w,s,b,mode,tile,threads);mx.eval(y);value=np.asarray(y)
      control=np.array_equal(value.view(np.uint32),np.asarray(ref).view(np.uint32))
      err=np.abs(value.astype(np.float64)-target);normalized=float(np.max(err/np.maximum(scale,1e-30)))
      passed=control and np.isfinite(value).all() and normalized<2e-5
      rows.append(dict(k=k,n=n,batch=batch,metadata=str(dtype),strided=strided,rows=tile,threads=threads,mode=mode,
       control_bitwise_equal=control,max_abs_error_fp64=float(err.max()),max_normalized_error=normalized,
       bitwise_equal_to_row_reuse=np.array_equal(value.view(np.uint32),np.asarray(ref).view(np.uint32)),passed=bool(passed)))
      if not passed:raise RuntimeError(rows[-1])
    assert digest([x,w,s,b])==before
out=dict(cases=rows,all_passed=True,inputs_unchanged=True,arithmetic_order_unchanged=True,all_tested_operator_outputs_bitwise_equal_to_baseline=all(r['bitwise_equal_to_row_reuse'] for r in rows),scope='Synthetic FP32 projected output; control bitwise equals row reuse; FP64 reference over exact reconstructed FP32 weights, finite bounded error, not greedy/quality acceptance')
(ROOT/'reports/unroll_int4_operator_20261008.json').write_text(json.dumps(out,indent=2)+'\n')
print(len(rows),'operator configurations passed; bitwise differences:',sum(not r['bitwise_equal_to_row_reuse'] for r in rows),flush=True)
