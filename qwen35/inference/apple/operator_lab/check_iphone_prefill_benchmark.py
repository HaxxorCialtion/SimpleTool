"""Full-recompute reference for exact-length synthetic benchmark fixtures."""
import argparse,json
from pathlib import Path
import mlx.core as mx
from simpletool.loading import load_model
from simpletool.decode import HEADS,prefill,batched_generate
from simpletool.stream_grouped_qmv import install_stream_grouped_qmv
from simpletool.fused_swiglu_int4 import install_fused_swiglu_int4
from simpletool.compiled import CompiledStep
p=argparse.ArgumentParser();p.add_argument('--model',choices=['2b','4b'],required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
native=json.loads((root/f'reports/iphone_prefill_benchmark_check_{a.model}_20261008.json').read_text())
mx.set_cache_limit(128*1024**2)
model,tok=load_model(root/f'artifacts/{a.model}_int4_compact',kernel='custom',compact_metadata=True)
install_stream_grouped_qmv(model);install_fused_swiglu_int4(model);step=CompiledStep(model)
checks=[]
for actual in native['samples']:
 ids=actual['diagnostics']['promptTokenIDs']
 prompt_equal=tok.encode(actual['diagnostics']['promptText'])==ids
 cache,_=prefill(model,ids,chunk_size=64)
 result=batched_generate(model,tok,ids,24,prefix_cache=cache,compiled_step=step)
 parity={h:result['channels'][h]['token_ids']==actual['tokenIDs'][h] for h in HEADS}
 checks.append({'new_prefill_tokens':actual['newPrefillTokens'],'hf_prompt_equal':prompt_equal,'token_parity':parity})
 print(a.model,actual['newPrefillTokens'],prompt_equal,parity,flush=True)
receipt={'model':a.model,'checks':checks,'all_passed':all(c['hf_prompt_equal'] and all(c['token_parity'].values()) for c in checks),'native_state_restoration':{k:native[k] for k in ['history_restored','cancel_restored','chat_cache_restored']}}
(root/f'reports/iphone_prefill_benchmark_parity_{a.model}_20261008.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
assert receipt['all_passed']
