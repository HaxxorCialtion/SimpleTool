"""One private int4 model per process: allocator cap vs real hot decode."""
import argparse
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path
import mlx.core as mx
import mlx_lm.models.gated_delta as gd
from simpletool.loading import load_model
from simpletool.compiled import CompiledStep
from simpletool.decode import HEADS,prompt_ids,prefill,batched_generate
from probe_int4_memory import snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--cache-mib',type=int,default=-1,help='-1 keeps framework default')
    p.add_argument('--fused-head',action='store_true');p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--keep-seed-cache',action='store_true')
    p.add_argument('--shared-gdn-seed',action='store_true')
    p.add_argument('--stream-grouped-qmv',action='store_true')
    p.add_argument('--fused-swiglu',action='store_true')
    p.add_argument('--tokens',type=int,default=96);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.cache_mib<-1:p.error('cache cap must be -1 or nonnegative')
    old_limit=mx.set_cache_limit(args.cache_mib*1024*1024) if args.cache_mib>=0 else None
    envpath=Path(inspect.getsourcefile(gd));envhash=hashlib.sha256(envpath.read_bytes()).hexdigest()
    model,tok=load_model(args.model,kernel='custom',compact_metadata=True)
    if args.shared_gdn_seed:
        from simpletool.shared_gdn_seed import install_shared_gdn_seed
        install_shared_gdn_seed(model)
    if args.stream_grouped_qmv:
        from simpletool.stream_grouped_qmv import install_stream_grouped_qmv
        install_stream_grouped_qmv(model)
    if args.fused_swiglu:
        from simpletool.fused_swiglu_int4 import install_fused_swiglu_int4
        install_fused_swiglu_int4(model)
    mx.clear_cache();loaded=snapshot()
    banned=tuple(tok.convert_tokens_to_ids(t) for t in ('<|endoftext|>','<|im_end|>'))
    step=CompiledStep(model,banned,fused_head=args.fused_head,release_seed_cache=not args.keep_seed_cache)
    ids=prompt_ids(tok,[{'role':'user','content':'列出手机设置里的所有选项'}]);source,_=prefill(model,ids)
    def run():
        started=time.perf_counter()
        result=batched_generate(model,tok,ids,args.tokens,prefix_cache=source,
            compiled_step=step,ignore_stops=True,suppress_tokens=banned)
        tokens={h:c['token_ids'] for h,c in result['channels'].items()}
        assert all(len(v)==args.tokens for v in tokens.values())
        return result,time.perf_counter()-started,tokens,snapshot()
    warm,_,tokens,warm_memory=run();samples=[]
    for i in range(args.rounds):
        result,wall,actual,memory=run();assert actual==tokens
        samples.append({'wall_s':wall,'stats':result['stats'],'memory':memory})
        print(args.cache_mib,i,wall,memory['mlx_cached_buffer_bytes'],flush=True)
    before_clear=snapshot();mx.clear_cache();after_clear=snapshot()
    report={'model':args.model,'cache_limit_mib':args.cache_mib,'previous_limit_bytes':old_limit,
        'fused_head':args.fused_head,'release_seed_cache':not args.keep_seed_cache,
        'shared_gdn_seed':args.shared_gdn_seed,
        'stream_grouped_qmv':args.stream_grouped_qmv,'fused_swiglu':args.fused_swiglu,
        'tokens_per_head':args.tokens,'rounds':args.rounds,
        'median_wall_s':statistics.median(x['wall_s'] for x in samples),'samples':samples,
        'loaded_memory':loaded,'warmup_memory':warm_memory,'before_clear_memory':before_clear,
        'after_clear_memory':after_clear,'tokens':tokens,
        'tokens_sha256':hashlib.sha256(json.dumps(tokens,sort_keys=True).encode()).hexdigest(),
        'all_runs_token_equal':True,'one_model_per_process':True,'exact_prefix_cache_warm':True,
        'weight_bits':4,'compute':'float32','competing_workloads_uncontrolled':True,
        'gated_delta_sha256':envhash,'gated_delta_sha256_at_end':hashlib.sha256(envpath.read_bytes()).hexdigest(),
        'iPhone_runtime_tested':False}
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
