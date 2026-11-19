"""Rotating within-process microbench, unchanged int4 weights, FP32 only."""
import argparse
import json
import statistics
import time
from pathlib import Path
import mlx.core as mx
from simpletool.kernels import int4_matmul
from simpletool.row_reuse_int4 import row_reuse_int4


def timed(fn,repeats):
    for _ in range(3):mx.eval(fn())
    begin=time.perf_counter()
    for _ in range(repeats):mx.eval(fn())
    return (time.perf_counter()-begin)/repeats


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=30);p.add_argument('--rounds',type=int,default=5)
    args=p.parse_args();mx.random.seed(20261009)
    results=[]
    for n,k in [(79,256),(9216,2560),(2560,9216),(65536,2560),(64,2560),(6144,2048),(2048,6144)]:
        packed,s,b=mx.quantize(mx.random.normal((n,k))*.02,bits=4,group_size=64)
        x=mx.random.normal((4,k));mx.eval(x,packed,s,b);mx.clear_cache()
        reference=int4_matmul(x,packed,s,b);mx.eval(reference)
        variants={'baseline':lambda:int4_matmul(x,packed,s,b)}
        for r in (1,2,4,8):
            for t in (64,128,256):
                variants[f'r{r}_t{t}']=lambda r=r,t=t:row_reuse_int4(x,packed,s,b,r,t)
        equal={};errors={}
        for name,fn in variants.items():
            result=fn();mx.eval(result)
            equal[name]=bool(mx.array_equal(reference,result).item())
            errors[name]=float(mx.max(mx.abs(reference-result)).item())
            assert mx.allclose(result,reference,atol=2e-5,rtol=2e-5).item(),(n,k,name,errors[name])
        samples={name:[] for name in variants};keys=list(variants)
        for i in range(args.rounds):
            offset=(i*3)%len(keys)
            for name in keys[offset:]+keys[:offset]:samples[name].append(timed(variants[name],args.repeats))
        medians={name:statistics.median(v) for name,v in samples.items()}
        results.append({'N':n,'K':k,'M':4,'median_ms':{k:v*1000 for k,v in medians.items()},
            'bitwise_equal':equal,'max_abs_error':errors,'samples_s':samples,
            'competing_workloads_uncontrolled':True,'compute':'float32','weights':'affine int4 group64'})
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        best=min(medians,key=medians.get)
        print(n,k,'best',best,'speedup',medians['baseline']/medians[best],'bitwise',equal[best],flush=True)


if __name__=='__main__':main()
