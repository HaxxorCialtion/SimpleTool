"""Bitwise same-row arithmetic gate, signed zero, strided input, immutable buffers."""
import hashlib,json
from pathlib import Path
import numpy as np
import mlx.core as mx
from simpletool.lut_int4 import lut_int4
from simpletool.row_reuse_int4 import row_reuse_int4
ROOT=Path(__file__).resolve().parent;rng=np.random.default_rng(47366);rows=[]
def sha(xs):return [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in xs]
for k,n in [(256,79),(512,257),(2048,81),(2560,256),(9216,17)]:
 for dtype in (mx.float16,mx.float32):
  w,s,b=mx.quantize(mx.array(rng.standard_normal((n,k)).astype(np.float32)*.02),group_size=64,bits=4);s=s.astype(dtype);b=b.astype(dtype);mx.eval(w,s,b)
  for batch in (1,2,3,4):
   for stride in (False,True):
    v=rng.standard_normal((batch,k*2 if stride else k)).astype(np.float32)*.3;v.ravel()[::13]=np.float32(-0.0)
    x=mx.array(v)[:,::2] if stride else mx.array(v);mx.eval(x);before=sha([x,w,s,b])
    for tile in (1,2,4,8):
     for threads in (64,128,256):
      ref=row_reuse_int4(x,w,s,b,tile,threads);y=lut_int4(x,w,s,b,tile,threads);mx.eval(ref,y)
      equal=np.array_equal(np.asarray(ref).view(np.uint32),np.asarray(y).view(np.uint32))
      rows.append(dict(k=k,n=n,batch=batch,metadata=str(dtype),strided=stride,rows=tile,threads=threads,bitwise_equal=equal))
      if not equal:
       (ROOT/'reports/lut_int4_operator_20261008.json').write_text(json.dumps(dict(cases=rows,all_passed=False),indent=2)+'\n');raise SystemExit(2)
    assert before==sha([x,w,s,b])
(ROOT/'reports/lut_int4_operator_20261008.json').write_text(json.dumps(dict(cases=rows,all_passed=True,inputs_unchanged=True),indent=2)+'\n')
print(len(rows),'LUT operator cases bitwise equal',flush=True)
