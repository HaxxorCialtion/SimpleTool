"""Rotated real natural-stop B4 requests; same int4 weights and prefill source."""
import argparse,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
import mlx.core as mx
from simpletool.loading import load_model
from simpletool.deferred_model import prepare_deferred_model
from simpletool.fixed_history_model import CompiledFixedHistoryStep
from simpletool.locality_gdn import STATES
from simpletool.compiled import CompiledStep
from simpletool.decode import prefill,batched_generate
from simpletool.cache import cache_arrays
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--size',choices=('4b','2b'),default='4b');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
mx.set_cache_limit(256*1024**2)
m,t=load_model(ROOT/f'artifacts/{a.size}_int4_compact',kernel='custom',compact_metadata=True);prepare_deferred_model(m)
keys=['dense1','baseline16','head16','value16'];rows=[]
for case in range(5):
 packet=json.loads((ROOT/f'reports/swift_suite_{a.size}_{case}_20261008.packet.json').read_text());source,pp=prefill(m,packet['prompt_ids'])
 def digest():return [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in cache_arrays(source)]
 before=digest();runners={'dense1':CompiledStep(m)}
 runners.update({o+'16':CompiledFixedHistoryStep(m,16,fused=True,state_type=STATES[o]) for o in STATES})
 samples={k:[] for k in keys};first={};last={};parity={k:True for k in keys}
 def run(key):
  start=time.perf_counter();result=batched_generate(m,t,packet['prompt_ids'],packet['max_tokens'],prefix_cache=source,compiled_step=runners[key])
  elapsed=time.perf_counter()-start
  equal=all(result['channels'][h]['token_ids']==packet['gold'][h] for h in packet['heads']);parity[key]&=equal
  if not equal:raise RuntimeError('natural request gold mismatch')
  last[key]=dict(active_batch_by_step=result['stats']['active_batch_by_step'],tokens_by_head={h:len(result['channels'][h]['token_ids']) for h in packet['heads']})
  return elapsed
 for key in keys:first[key]=run(key)
 for round in range(5):
  shift=(case+round)%len(keys)
  for key in keys[shift:]+keys[:shift]:samples[key].append(run(key))
 unchanged=before==digest();assert unchanged
 rows.append(dict(case=case,prefill_s=pp,median_wall_s={k:statistics.median(v) for k,v in samples.items()},samples_s=samples,
  first_run_s=first,token_equal_to_gold=parity,prefix_unchanged=unchanged,actual_stops=last))
 a.output.write_text(json.dumps(dict(size=a.size,cases=rows,weight_bits=4,compute='float32',rounds=5,
 scope='Same weight set and one immutable prefill per input; natural B4 markers, independent stops, batch shrink; five rotated warm rounds per real input',
 limits='Python complete request windows; first-run paths sequentially prewarmed, not fair cold GPU tests; not Swift/image/full multi-turn/quantization-quality/iPhone; external load uncontrolled'),indent=2)+'\n')
 print(a.size,case,rows[-1]['median_wall_s'],flush=True)
