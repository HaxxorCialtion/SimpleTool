"""Actual int4 matrices; rotate layers, preserving failures and raw samples."""
import json,time,statistics,hashlib,argparse
from pathlib import Path
import mlx.core as mx
from simpletool.simd_branch_qmv import simd_branch_qmv
from simpletool.stream_grouped_qmv import stream_grouped_qmv
from simpletool.row_reuse_int4 import row_reuse_int4
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--size',choices=('2b','4b'),default='4b');p.add_argument('--rounds',type=int,default=5)
p.add_argument('--output',type=Path);a=p.parse_args()
out=a.output or ROOT/f'reports/simd_branch_qmv_{a.size}_20261008.json'
mx.set_cache_limit(256*1024**2);mx.random.seed(20261008)
weights={}
for f in (ROOT/f'artifacts/{a.size}_int4_compact').glob('*.safetensors'):weights.update(mx.load(str(f)))
mx.eval(list(weights.values()))
names=sorted(k[:-7] for k in weights if '.mlp.down_proj.weight' in k)
families={'down':names,'gate':[k.replace('down_proj','gate_proj') for k in names],
          'vocabulary':['language_model.model.embed_tokens']}
records=[]
def save():
    out.write_text(json.dumps(dict(size=a.size,weight_bits=4,compute='float32',rounds=a.rounds,records=records,
        source_sha256=hashlib.sha256((ROOT/'simpletool/simd_branch_qmv.py').read_bytes()).hexdigest(),
        scope='same actual layers rotated, synchronous operation wall time; no DRAM/GPU-only claim'),indent=2)+'\n')
for family,paths in families.items():
    matrices=[tuple(weights[path+suffix] for suffix in ('.weight','.scales','.biases')) for path in paths]
    n=matrices[0][0].shape[0];k=matrices[0][0].shape[1]*8
    for batch in [2,3,4]:
        x=mx.random.normal((batch,k));mx.eval(x)
        configs=[('row_reuse',None,None),('stream',None,None)]+[(f'{"shared" if shared else "split"}_r{r}',r,shared) for shared in [False,True] for r in [1,2,4,8]]
        def call(cfg,matrix):
            name,rows,shared=cfg;w,s,b=matrix
            if name=='row_reuse':return row_reuse_int4(x,w,s,b,4 if n>=65536 else 8,256 if k<=2560 or n>=65536 else 128)
            if name=='stream':return stream_grouped_qmv(x,w,s,b)
            return simd_branch_qmv(x,w,s,b,rows,shared)
        gold=[mx.concatenate([stream_grouped_qmv(x[i:i+1],*matrix) for i in range(batch)]) for matrix in matrices]
        mx.eval(gold)
        parity={}
        for cfg in configs:
            results=[call(cfg,matrix) for matrix in matrices];mx.eval(results)
            parity[cfg[0]]=all(mx.array_equal(y.view(mx.uint32),g.view(mx.uint32)).item() for y,g in zip(results,gold))
            if cfg[0] not in ('row_reuse','stream') and not parity[cfg[0]]:
                records.append(dict(family=family,batch=batch,failed_kernel=cfg[0],parity=parity));save();raise SystemExit(2)
        samples={cfg[0]:[] for cfg in configs}
        # Every matrix is evaluated separately: no giant retained output graph.
        # Vocabulary repeats are one matrix, so that family is explicitly hot.
        repetitions=3 if family=='vocabulary' else 1
        for round_index in range(a.rounds):
            shift=round_index%len(configs)
            for cfg in configs[shift:]+configs[:shift]:
                start=time.perf_counter()
                for _ in range(repetitions):
                    for matrix in matrices:mx.eval(call(cfg,matrix))
                samples[cfg[0]].append((time.perf_counter()-start)/(len(matrices)*repetitions))
        med={key:statistics.median(vals) for key,vals in samples.items()}
        record=dict(family=family,batch=batch,N=n,K=k,matrix_count=len(matrices),samples_s=samples,median_s=med,
                    bitwise_equal_to_independent_vectors=parity,layer_paths=paths,
                    scope='distinct layers' if len(matrices)>1 else 'repeated single vocabulary matrix')
        records.append(record);save();print(family,batch,{key:round(value*1000,4) for key,value in med.items()},flush=True)
