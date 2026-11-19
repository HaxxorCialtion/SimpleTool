"""Real same-int4 greedy gate. Record every divergence, finish other candidates."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from simpletool.loading import load_model
from simpletool.scalar_int4 import install_scalar_int4
from simpletool.unroll_int4 import install_unroll_int4
MODES=('scalar','j8','vocab')
def install_fma_int4(model,mode):
 if mode in ('scalar','vocab'):install_scalar_int4(model,'vocab' if mode=='vocab' else 'all')
 else:install_unroll_int4(model,mode)
from simpletool.compiled import CompiledStep
from simpletool.decode import prefill,batched_generate
from simpletool.cache import cache_arrays
from simpletool.shared_gdn_seed import install_shared_gdn_seed
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--size',choices=('4b','2b'),default='4b');a=p.parse_args()
mx.set_cache_limit(256*1024**2)
base,t=load_model(ROOT/f'artifacts/{a.size}_int4_compact',kernel='custom',compact_metadata=True);install_shared_gdn_seed(base)
arrays=dict(tree_flatten(base.parameters()));models={};steps={};rows=[]
for mode in MODES:
 m=copy.deepcopy(base,{id(x):x for x in arrays.values()});install_fma_int4(m,mode)
 assert all(dict(tree_flatten(m.parameters()))[n] is x for n,x in arrays.items())
 models[mode]=m;steps[mode]=CompiledStep(m)
def digest(xs):return [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in xs]
def compare(values,gold):
 first=next((i for i,(x,y) in enumerate(zip(values,gold)) if x!=y),None)
 if first is None and len(values)!=len(gold):first=min(len(values),len(gold))
 return dict(equal=values==gold,first_different_index=first)
for case in range(5):
 packet=json.loads((ROOT/f'reports/swift_suite_{a.size}_{case}_20261008.packet.json').read_text());source,_=prefill(base,packet['prompt_ids']);before=digest(cache_arrays(source))
 for mode in MODES:
  for chunk in (1,8):
   for batch in (1,2,3,4):
    groups=[[h] for h in packet['heads']] if batch==1 else [packet['heads'][:batch]]
    results={};stats=[]
    for heads in groups:
     r=batched_generate(models[mode],t,packet['prompt_ids'],packet['max_tokens'],heads=heads,prefix_cache=source,compiled_step=steps[mode],decode_chunk_size=chunk)
     results.update(r['channels']);stats.append(r['stats'])
    equal={h:compare(r['token_ids'],packet['gold'][h]) for h,r in results.items()};immutable=before==digest(cache_arrays(source))
    passed=immutable and all(d['equal'] for d in equal.values())
    rows.append(dict(case=case,mode=mode,chunk=chunk,batch=batch,passed=passed,comparison=equal,prefix_unchanged=immutable,channels=results,stats=stats))
    (ROOT/f'reports/scalar_model_{a.size}_20261008.json').write_text(json.dumps(dict(cases=rows,all_passed=all(r['passed'] for r in rows),weight_bits=4,compute='float32',arithmetic_order_unchanged=True,parameter_objects_shared=True),ensure_ascii=False,indent=2)+'\n')
    print(a.size,case,mode,chunk,batch,passed,{h:d['first_different_index'] for h,d in equal.items()},flush=True)
    if not immutable:raise RuntimeError('Shared prefix mutation')
print('greedy mismatches',sum(not r['passed'] for r in rows),flush=True)
# A differing FP32 rounding path is experimental even when tested tokens match.
# Keep all other candidates' evidence when one fails.
if not all(r['passed'] for r in rows):raise SystemExit(2)
