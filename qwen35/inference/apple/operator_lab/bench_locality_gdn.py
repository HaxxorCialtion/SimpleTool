"""Exact output parity and rotating isolated dispatch timing; no bandwidth claim."""
import argparse,json,statistics,time,hashlib
from pathlib import Path
import numpy as np
import mlx.core as mx
from simpletool.locality_gdn import STATES
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--rounds',type=int,default=5);p.add_argument('--repeats',type=int,default=40);a=p.parse_args()
mx.set_cache_limit(256*1024**2);rng=np.random.default_rng(172903);rows=[]
def rand(shape):return mx.array(rng.standard_normal(shape).astype(np.float32))
def bits(x):return np.asarray(x).view(np.uint32)
for hk,hv,dv in [(16,32,128),(16,16,128)]:
 for b in (1,2,3,4):
  for sb in ((1,) if b==1 else (1,b)):
   for interval,phase,write in [(4,0,False),(4,2,False),(4,3,True),(16,0,False),(16,14,False),(16,15,True)]:
    base=rand((sb,hv,dv,128))*.01;keys=rand((b,interval-1,hk,128))*.03
    gates=mx.exp(-mx.abs(rand((b,interval-1,hv))));deltas=rand((b,interval-1,hv,dv))*.1
    q=rand((b,1,hk,128))*.03;k=rand(q.shape)*.03;v=rand((b,1,hv,dv))*.1
    g=mx.exp(-mx.abs(rand((b,1,hv))));beta=mx.sigmoid(rand(g.shape));pending=mx.array(phase,mx.int32)
    inputs=[q,k,v,g,beta,base,keys,gates,deltas,pending];mx.eval(inputs)
    before=hashlib.sha256(np.asarray(base).tobytes()).hexdigest();functions={};parity={};samples={o:[] for o in STATES}
    for order,State in STATES.items():
     def run(*xs,State=State):
        d=State(xs[5],b,hk,interval,xs[9],write,True);d.keys,d.gates,d.deltas=xs[6:9]
        y=d.step(*xs[:5]);return [y,*d.arrays]
     functions[order]=mx.compile(run)
     out=functions[order](*inputs);mx.eval(out)
     if order=='baseline':reference=[bits(x).copy() for x in out]
     parity[order]=all(np.array_equal(x,bits(y)) for x,y in zip(reference,out))
     if not parity[order]:raise RuntimeError(f'{order} output mismatch')
    orders=list(STATES)
    for r in range(a.rounds):
     shift=r%3
     for order in orders[shift:]+orders[:shift]:
        start=time.perf_counter()
        for _ in range(a.repeats):out=functions[order](*inputs);mx.eval(out)
        samples[order].append((time.perf_counter()-start)/a.repeats)
    assert before==hashlib.sha256(np.asarray(base).tobytes()).hexdigest()
    row=dict(hk=hk,hv=hv,dv=dv,batch=b,stored_batch=sb,interval=interval,phase=phase,write=write,
             median_s={o:statistics.median(v) for o,v in samples.items()},samples_s=samples,bitwise_equal=parity,seed_unchanged=True)
    rows.append(row)
    (ROOT/'reports/locality_gdn_ab_20261008.json').write_text(json.dumps(dict(cases=rows,rounds=a.rounds,repeats=a.repeats,
      scope='isolated compiled FP32 GDN step plus immutable history outputs; GPU synchronised each call; no model projections or decode; not DRAM counters or iPhone'),indent=2)+'\n')
    print(hv,b,sb,interval,phase,write,row['median_s'],flush=True)
print(len(rows),'isolated configurations complete',flush=True)
