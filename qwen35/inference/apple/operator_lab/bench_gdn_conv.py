import json
import statistics
import time
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from simpletool.gdn_conv import conv_state


def main():
    mx.random.seed(89);rows=[]
    for B,D in ((4,6144),(4,8192),(3,8192),(2,8192),(1,8192)):
        history=mx.random.normal((B,3,D));qkv=mx.random.normal((B,1,D));weight=mx.random.normal((D,4,1))*.2
        mx.eval(history,qkv,weight)
        def stock():
            full=mx.concatenate([history,qkv],axis=1)
            return nn.silu(mx.conv1d(full,weight,groups=D)),mx.contiguous(full[:,-3:,:])
        def fused():return conv_state(history,qkv,weight)
        expected=stock();actual=fused();mx.eval(expected,actual)
        equal=[mx.array_equal(x,y).item() for x,y in zip(actual,expected)]
        errors=[mx.max(mx.abs(x-y)).item() for x,y in zip(actual,expected)]
        assert all(mx.allclose(x,y,atol=2e-6,rtol=2e-6).item() for x,y in zip(actual,expected)),errors
        samples={'stock':[],'fused':[]}
        for round in range(5):
            for name,fn in [('stock',stock),('fused',fused)][::1 if round%2 else -1]:
                for _ in range(3):mx.eval(fn())
                start=time.perf_counter()
                for _ in range(100):mx.eval(fn())
                samples[name].append((time.perf_counter()-start)/100)
        rows.append({'B':B,'D':D,'bitwise_equal_out_history':equal,'max_errors':errors,
            'median_us':{k:statistics.median(v)*1e6 for k,v in samples.items()},'samples_s':samples})
        print(rows[-1],flush=True)
    Path('reports/gdn_conv_state_micro_20261008.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
