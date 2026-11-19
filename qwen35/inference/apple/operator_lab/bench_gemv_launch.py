"""Tune launch width and loop unrolling without changing the GEMV layout."""
import ast
import json
import math
import statistics
from pathlib import Path
import mlx.core as mx
from bench_tuned_kernel import timed
from simpletool.kernels import int4_matmul

def make_kernel(unroll=False,fast=False):
    tree=ast.parse(Path('simpletool/kernels.py').read_text())
    assign=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_INT4' for t in n.targets))
    source=ast.literal_eval(next(k.value for k in assign.value.keywords if k.arg=='source'))
    if unroll:
        source=source.replace('for (uint j=0; j<8; ++j)', '#pragma unroll\n            for (uint j=0; j<8; ++j)')
        source=source.replace('for (uint m=0; m<M; ++m)', '#pragma unroll\n                for (uint m=0; m<M; ++m)')
    return mx.fast.metal_kernel(name=f'simpletool_launch_u{int(unroll)}_f{int(fast)}',
        input_names=['x','w','scales','biases'],output_names=['y'],source=source,
        compile_options={'math_mode':'fast' if fast else 'safe'})

def main():
    mx.random.seed(59);rows=[]
    kernels={'safe':make_kernel(),'unroll':make_kernel(True),'fast_unroll':make_kernel(True,True)}
    for n,k in [(6144,2048),(2048,6144),(9216,2560),(2560,9216),(65536,2560)]:
        w=mx.random.normal((n,k)).astype(mx.float16)*.02
        p,s,b=mx.quantize(w,bits=4,group_size=64);s=s.astype(mx.float32);b=b.astype(mx.float32)
        x=mx.random.normal((4,k));mx.eval(x,p,s,b);del w
        funcs={'old':lambda:int4_matmul(x,p,s,b)}
        for name,kernel in kernels.items():
            for t in (32,64,128,256,512):
                funcs[f'{name}_{t}']=lambda kernel=kernel,t=t:kernel(inputs=[x,p,s,b],template=[('T',mx.float32),('M',4),('N',n),('K',k)],
                    grid=(math.ceil(n*32/t)*t,1,1),threadgroup=(t,1,1),output_shapes=[(4,n)],output_dtypes=[mx.float32])[0]
        reference=funcs['old']();mx.eval(reference)
        errors={name:float(mx.max(mx.abs(fn()-reference)).item()) for name,fn in funcs.items()}
        samples={name:[] for name in funcs};keys=list(funcs)
        for i in range(3):
            for key in keys[i*5:]+keys[:i*5]:samples[key].append(timed(funcs[key],20))
        med={name:statistics.median(v)*1e3 for name,v in samples.items()}
        row={'N':n,'K':k,'median_ms':med,'max_abs_difference':errors,'samples':samples}
        rows.append(row);print(n,k,med,errors,flush=True)
        Path('reports/gemv_launch_20261008.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
