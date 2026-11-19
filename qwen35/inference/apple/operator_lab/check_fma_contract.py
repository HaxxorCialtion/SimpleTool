"""Current installed MLX compiler experiment; independent FP32 operation oracle."""
import json
from pathlib import Path
import numpy as np
import mlx.core as mx
rng=np.random.default_rng(89110);n=8192
xs=[mx.array(rng.standard_normal(n).astype(np.float32)) for _ in range(3)]
source=r'''
 const uint i=thread_position_in_grid.x;
 automatic[i]=a[i]*b[i]+c[i];
 explicit_fma[i]=metal::fma(a[i],b[i],c[i]);
 {
  #pragma clang fp contract(off)
  separate[i]=a[i]*b[i]+c[i];
 }
'''
k=mx.fast.metal_kernel(name='st_fma_contract_probe',input_names=['a','b','c'],output_names=['automatic','explicit_fma','separate'],source=source)
y=k(inputs=xs,grid=(n,1,1),threadgroup=(256,1,1),output_shapes=[(n,)]*3,output_dtypes=[mx.float32]*3);mx.eval(xs,y)
a,b,c=[np.asarray(x) for x in xs];automatic,fma,separate=[np.asarray(x) for x in y]
expected_fma=(a.astype(np.float64)*b.astype(np.float64)+c.astype(np.float64)).astype(np.float32)
expected_separate=(a*b+c).astype(np.float32)
bit=lambda a,b:np.array_equal(a.view(np.uint32),b.view(np.uint32))
out=dict(count=n,automatic_equals_explicit_fma=bit(automatic,fma),automatic_equals_separate=bit(automatic,separate),
 separate_equals_np_two_rounds=bit(separate,expected_separate),explicit_fma_equals_fp64_oracle=bit(fma,expected_fma),
 differing_elements=int(np.sum(fma.view(np.uint32)!=separate.view(np.uint32))),
 inference='Default local Metal compiler contracts same-expression multiply/add; disabling fast math is distinct from disabling FP contraction',
 scope='Current installed Python MLX custom kernel, finite random FP32 values; CPU FP64 oracle for these inputs; not all-input mathematical proof')
Path('reports/fma_contract_runtime_20261008.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
assert out['automatic_equals_explicit_fma'] and out['separate_equals_np_two_rounds'] and out['explicit_fma_equals_fp64_oracle'] and out['differing_elements']>0
