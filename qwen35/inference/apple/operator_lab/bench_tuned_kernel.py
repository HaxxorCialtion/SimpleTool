import json
import statistics
import time
from pathlib import Path
import mlx.core as mx
from simpletool.kernels import int4_matmul
from simpletool.tuned_int4 import tuned_int4

def timed(fn,n=30):
    for _ in range(3):mx.eval(fn())
    t=time.perf_counter()
    for _ in range(n):mx.eval(fn())
    return (time.perf_counter()-t)/n

def main():
    mx.random.seed(20261008)
    rows=[]
    for n,k in [(6144,2048),(2048,6144),(9216,2560),(2560,9216),(65536,2560),(64,2560)]:
        w=mx.random.normal((n,k)).astype(mx.float16)*.02
        packed,s,b=mx.quantize(w,bits=4,group_size=64)
        s=s.astype(mx.float32);b=b.astype(mx.float32)
        x=mx.random.normal((4,k));mx.eval(x,packed,s,b);del w
        reference=mx.quantized_matmul(x,packed,s,b,transpose=True,bits=4,group_size=64)
        mx.eval(reference)
        variants={'old':lambda:int4_matmul(x,packed,s,b),
                  'mlx':lambda:mx.quantized_matmul(x,packed,s,b,transpose=True,bits=4,group_size=64)}
        for lanes in (4,8,16,32):
            for vector in (False,True):
                variants[f'l{lanes}_{"vec" if vector else "scalar"}']=lambda l=lanes,v=vector:tuned_int4(x,packed,s,b,l,v)
        samples={key:[] for key in variants}
        errors={}
        for key,fn in variants.items():
            result=fn();mx.eval(result)
            errors[key]=float(mx.max(mx.abs(reference-result)).item())
            assert mx.allclose(result,reference,atol=2e-5,rtol=2e-5).item(),(key,n,k,errors[key])
        keys=list(variants)
        for i in range(3):
            for key in keys[i*3:]+keys[:i*3]:samples[key].append(timed(variants[key]))
        med={key:statistics.median(v)*1000 for key,v in samples.items()}
        row={'N':n,'K':k,'M':4,'median_ms':med,'max_abs_difference_vs_mlx':errors,'samples_s':samples}
        rows.append(row);print(n,k,med,flush=True)
        Path('reports/tuned_kernel_20261008.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
