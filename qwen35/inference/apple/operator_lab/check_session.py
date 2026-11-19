import argparse
import json
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from simpletool.decode import HEADS, batched_generate, prompt_ids
from simpletool.session import ImageSession, TextSession


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--image')
    p.add_argument('--vision-gguf')
    p.add_argument('--mmproj')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compute',choices=['float16','float32'],default='float32')
    p.add_argument('--kernel',choices=['mlx','custom'],default='custom')
    p.add_argument('--split-projections',action='store_true')
    p.add_argument('--compiled',action='store_true')
    p.add_argument('--row-reuse',action='store_true')
    p.add_argument('--compact-metadata',action='store_true')
    p.add_argument('--cache-limit-mib',type=int)
    p.add_argument('--shared-gdn-seed',action='store_true')
    p.add_argument('--stream-grouped-qmv',action='store_true')
    p.add_argument('--fused-swiglu',action='store_true')
    p.add_argument('--vision-exporter',type=Path,default=Path('artifacts/vision_export'))
    args=p.parse_args()
    from simpletool.memory_policy import configure_allocator_cache
    try:configure_allocator_cache(args.cache_limit_mib)
    except ValueError as error:p.error(str(error))
    if args.compact_metadata and not args.row_reuse:p.error('compact metadata requires --row-reuse')
    if (args.stream_grouped_qmv or args.fused_swiglu) and not args.compact_metadata:p.error('new projection kernels require --compact-metadata')
    if args.split_projections and args.kernel!='custom':p.error('split projections requires custom kernel')
    if args.row_reuse and (args.split_projections or args.kernel!='custom'):p.error('row reuse requires unfused custom projections')
    from simpletool.loading import load_model
    model,tok=load_model(args.model,compute=args.compute,kernel=args.kernel,compact_metadata=args.compact_metadata)
    if args.split_projections:
        from simpletool.projection_fusion import fuse_projections
        fuse_projections(model,split_output=True)
    if args.row_reuse:
        from simpletool.row_reuse_int4 import install_row_reuse
        if not args.compact_metadata:install_row_reuse(model)
    if args.shared_gdn_seed:
        if args.split_projections or args.compute!='float32':p.error('shared GDN seed requires unfused FP32 GDN')
        from simpletool.shared_gdn_seed import install_shared_gdn_seed
        install_shared_gdn_seed(model)
    if args.stream_grouped_qmv:
        from simpletool.stream_grouped_qmv import install_stream_grouped_qmv
        install_stream_grouped_qmv(model)
    if args.fused_swiglu:
        from simpletool.fused_swiglu_int4 import install_fused_swiglu_int4
        install_fused_swiglu_int4(model)
    from simpletool.compiled import CompiledStep
    compiled=CompiledStep(model) if args.compiled else None
    if args.image:
        session=ImageSession(model,tok,args.image,args.vision_gguf,args.mmproj,args.vision_exporter,compiled_step=compiled)
    else:session=TextSession(model,tok,compiled_step=compiled)
    results=[]
    for question in ['请把明天上午九点设置成提醒','改成上午十点，其他要求不变']:
        result=session.reply(question,max_tokens=48)
        messages=session.messages[:-1]
        ids,emb,pos,next_pos,_=session.prepare(messages)
        fresh=batched_generate(model,tok,ids,48,embeddings=emb,positions=pos,next_position=next_pos)
        equal={h:result['channels'][h]['token_ids']==fresh['channels'][h]['token_ids'] for h in HEADS}
        result['fresh_recompute_token_equal']=equal
        results.append(result)
        print(question,result['stats'],equal,flush=True)
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    if not all(all(r['fresh_recompute_token_equal'].values()) for r in results):raise SystemExit(2)


if __name__=='__main__':main()
