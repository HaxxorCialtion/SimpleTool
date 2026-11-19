from pathlib import Path
import json
import numpy as np
import mlx.core as mx
from simpletool.kernels import int4_matmul

root=Path('artifacts/native_int4_fixture');root.mkdir(exist_ok=True)
mx.random.seed(67)
w,s,b=mx.quantize(mx.random.normal((79,256)),group_size=64,bits=4)
for name,a in [('packed',w),('scales',s),('biases',b)]:np.asarray(a).tofile(root/f'{name}.bin')
for m in (1,2,3,4):
    x=mx.random.normal((m,256));np.asarray(x).tofile(root/f'x{m}.bin')
    np.asarray(int4_matmul(x,w,s,b)).tofile(root/f'expected{m}.bin')
(root/'format.json').write_text(json.dumps({'M':[1,2,3,4],'N':79,'K':256,'bits':4,'group_size':64,
    'weights':'uint32 low to high nibbles','compute':'float32'},indent=2)+'\n')
