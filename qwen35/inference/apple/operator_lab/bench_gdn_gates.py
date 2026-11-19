import json
import statistics
import time
from pathlib import Path
import mlx.core as mx
from mlx_lm.models.gated_delta import compute_g
from simpletool.gdn_gates import gate_pair


def main():
    mx.random.seed(79);rows=[]
    for B,H in ((4,32),(4,16),(3,32),(2,32),(1,32)):
        a=mx.random.normal((B,1,H))*4;b=mx.random.normal((B,1,H))*4
        A_log=mx.random.normal((H,));dt=mx.random.normal((H,))
        mx.eval(a,b,A_log,dt)
        def stock():return compute_g(A_log,a,dt),mx.sigmoid(b)
        def fused():return gate_pair(a,b,A_log,dt)
        expected=stock();actual=fused();mx.eval(expected,actual)
        equal=[mx.array_equal(x,y).item() for x,y in zip(actual,expected)]
        errors=[mx.max(mx.abs(x-y)).item() for x,y in zip(actual,expected)]
        assert all(mx.allclose(x,y,atol=1e-7,rtol=1e-6).item() for x,y in zip(actual,expected)),errors
        samples={'stock':[],'fused':[]}
        for round in range(5):
            for name,fn in [('stock',stock),('fused',fused)][::1 if round%2 else -1]:
                for _ in range(3):mx.eval(fn())
                start=time.perf_counter()
                for _ in range(200):mx.eval(fn())
                samples[name].append((time.perf_counter()-start)/200)
        rows.append({'B':B,'H':H,'bitwise_equal_g_beta':equal,'max_errors':errors,
            'median_us':{k:statistics.median(v)*1e6 for k,v in samples.items()},'samples_s':samples})
        print(rows[-1],flush=True)
    Path('reports/gdn_gate_pair_micro_20261008.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
