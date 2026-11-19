"""Whole-model prediction and ALL functional cache arrays after each decode step."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from simpletool.loading import load_model
from simpletool.shared_gdn_seed import install_shared_gdn_seed
from simpletool.gdn_prepare import install_prepare_gdn
from simpletool.compiled import CompiledStep
from simpletool.decode import prefill
from simpletool.cache import fork_batch,cache_arrays

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--size',choices=('4b','2b'),required=True);a=p.parse_args()
mx.set_cache_limit(256*1024**2)
base,t=load_model(ROOT/f'artifacts/{a.size}_int4_compact',kernel='custom',compact_metadata=True)
install_shared_gdn_seed(base)
params=dict(tree_flatten(base.parameters()));models={'baseline':base}
for mode in ('conv','prepare','gates'):
    model=copy.deepcopy(base,{id(x):x for x in params.values()});install_prepare_gdn(model,mode)
    assert all(dict(tree_flatten(model.parameters()))[n] is x for n,x in params.items())
    models[mode]=model
packet=json.loads((ROOT/f'reports/swift_{a.size}_packet_20261008.json').read_text())
source,_=prefill(base,packet['prompt_ids'])
def digest():return [hashlib.sha256(np.asarray(x).tobytes()).hexdigest() for x in cache_arrays(source)]
before=digest();rows=[]
for B in (1,2,3,4):
    markers=[packet['marker_ids'][h] for h in packet['heads'][:B]]
    steps={key:CompiledStep(model) for key,model in models.items()}
    states={key:step.initialize(fork_batch(source,B),B,24) for key,step in steps.items()}
    prediction=None
    for used in range(len(markers[0])+8):
        current=mx.array([[m[used]] for m in markers]) if used<len(markers[0]) else prediction[:,None]
        outputs={}
        for key,step in steps.items():
            outputs[key],states[key]=step.forward(current,states[key],mx.array(used,mx.int32))
            mx.eval(outputs[key],states[key])
        prediction=outputs['baseline']
        for mode in ('conv','prepare','gates'):
            failures=[]
            for i,(actual,reference) in enumerate(zip(states[mode],states['baseline'])):
                for j,(x,y) in enumerate(zip(actual,reference)):
                    if not mx.array_equal(x.view(mx.uint32),y.view(mx.uint32)).item():
                        failures.append(dict(layer=i,slot=j,max_diff=mx.max(mx.abs(x-y)).item()))
            rows.append(dict(batch=B,step=used,mode=mode,prediction_equal=mx.array_equal(outputs[mode],prediction).item(),
                             whole_cache_bitwise_equal=not failures,failures=failures))
        print(a.size,'B',B,'step',used,'mismatches',sum(not r['whole_cache_bitwise_equal'] for r in rows),flush=True)
    assert before==digest()
result=dict(cases=rows,all_passed=all(r['whole_cache_bitwise_equal'] and r['prediction_equal'] for r in rows),
            source_unchanged=before==digest(),parameter_objects_shared=True,weight_bits=4,compute='float32',
            scope='One actual text prefix, full real marker sequences plus eight baseline-greedy steps; every functional cache array bitwise, B1..4, three prepare modes. Does not prove arbitrary input quality.')
(ROOT/f'reports/prepare_state_{a.size}_20261008.json').write_text(json.dumps(result,indent=2)+'\n')
if not result['all_passed']:raise SystemExit(2)
