"""All actual layer weight sweep; bitwise gate and rotated SIMD LUT tile timings."""
import argparse,json,time,statistics
from pathlib import Path
import mlx.core as mx
from simpletool.row_reuse_int4 import row_reuse_int4
from simpletool.lut_int4 import lut_int4
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--size',choices=('4b','2b'),default='4b');a=p.parse_args()
mx.set_cache_limit(256*1024**2);mx.random.seed(99181);weights={};records=[]
for f in (ROOT/f'artifacts/{a.size}_int4_compact').glob('*.safetensors'):weights.update(mx.load(str(f)))
mx.eval(list(weights.values()));names=sorted(k[:-7] for k in weights if '.mlp.down_proj.weight' in k)
families={'down':names,'gate':[k.replace('down_proj','gate_proj') for k in names],'vocabulary':['language_model.model.embed_tokens']}
for family,paths in families.items():
 mats=[tuple(weights[n+s] for s in ('.weight','.scales','.biases')) for n in paths];n=mats[0][0].shape[0];k=mats[0][0].shape[1]*8
 for batch in (2,3,4):
  x=mx.random.normal((batch,k));mx.eval(x);tile=4 if n>=65536 else 8;threads=256 if k<=2560 or n>=65536 else 128
  configs={'baseline':(tile,threads),**{f'r{r}t{t}':(r,t) for r in (1,2,4,8) for t in (128,256)}}
  keys=list(configs);samples={q:[] for q in keys};parity={q:True for q in keys}
  def call(key,mat):return row_reuse_int4(x,*mat,*configs[key]) if key=='baseline' else lut_int4(x,*mat,*configs[key])
  for mat in mats:
   reference=call('baseline',mat);mx.eval(reference)
   for key in keys:
    y=call(key,mat);mx.eval(y);parity[key]&=mx.array_equal(y.view(mx.uint32),reference.view(mx.uint32)).item()
    if not parity[key]:raise RuntimeError((family,batch,key,'LUT bitwise mismatch'))
  repeats=3 if family=='vocabulary' else 1
  for turn in range(5):
   shift=(3*turn)%len(keys)
   for key in keys[shift:]+keys[:shift]:
    start=time.perf_counter()
    for _ in range(repeats):
     for mat in mats:mx.eval(call(key,mat))
    samples[key].append((time.perf_counter()-start)/(len(mats)*repeats))
  med={q:statistics.median(v) for q,v in samples.items()};records.append(dict(family=family,batch=batch,n=n,k=k,layers=len(mats),tiles=configs,median_s=med,samples_s=samples,bitwise_equal=parity,layer_paths=paths))
  (ROOT/f'reports/lut_int4_micro_{a.size}_20261008.json').write_text(json.dumps(dict(records=records,weight_bits=4,compute='float32',scope='Distinct actual layers, synchronised wall time; vocabulary repeats one matrix; not GPU-only/DRAM/phone'),indent=2)+'\n')
  print(a.size,family,batch,med,flush=True)
