import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from simpletool.decode import HEADS,batched_generate,prompt_ids,serial_generate

CASES=[
    '帮我把WiFi关掉，然后调低屏幕亮度',
    '明天早上八点半叫我起床，用最大的闹钟音量',
    '现在的电量是多少？',
    '取消刚才的操作，不要执行任何命令',
    '请把明天上午九点设置成提醒',
]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compute',choices=['float16','float32'],default='float32')
    p.add_argument('--kernel',choices=['mlx','custom'],default='custom')
    p.add_argument('--fused-gdn',action='store_true')
    p.add_argument('--split-projections',action='store_true')
    p.add_argument('--compiled',action='store_true')
    p.add_argument('--row-reuse',action='store_true')
    p.add_argument('--decode-chunk',type=int,choices=[1,2,4,8],default=1)
    p.add_argument('--conv-fusion',action='store_true')
    p.add_argument('--compact-metadata',action='store_true')
    p.add_argument('--fused-head',action='store_true')
    p.add_argument('--shared-gdn-seed',action='store_true')
    p.add_argument('--grouped-qmv',action='store_true')
    p.add_argument('--stream-grouped-qmv',action='store_true')
    p.add_argument('--fused-swiglu',action='store_true')
    args=p.parse_args()
    if args.split_projections and args.kernel!='custom':p.error('split projections requires custom kernel')
    if args.row_reuse and (args.kernel!='custom' or args.split_projections or args.fused_gdn):
        p.error('row reuse requires custom kernel and unfused projections')
    if args.decode_chunk>1 and not args.compiled:p.error('decode chunks require compiled inference')
    if args.conv_fusion and (args.split_projections or args.fused_gdn):p.error('conv fusion requires unfused projections')
    if args.compact_metadata and (not args.row_reuse or args.compute!='float32'):
        p.error('compact metadata requires row reuse and FP32 compute')
    if args.fused_head and (not args.row_reuse or not args.compiled):p.error('fused head requires row reuse and compiled inference')
    if args.shared_gdn_seed and (args.fused_gdn or args.split_projections or args.conv_fusion):p.error('shared GDN seed requires unfused GDN')
    if args.grouped_qmv and not args.compact_metadata:p.error('grouped QMV requires compact int4 metadata')
    if args.stream_grouped_qmv and (args.grouped_qmv or not args.compact_metadata):p.error('stream grouped QMV requires compact int4 metadata and excludes old grouped QMV')
    if args.fused_swiglu and (not args.compact_metadata or args.split_projections):p.error('fused SwiGLU requires compact int4 metadata and original MLP')
    from simpletool.loading import load_model
    model,tok=load_model(args.model,compute=args.compute)
    golds=[]
    for text in CASES:
        ids=prompt_ids(tok,[{'role':'user','content':text}])
        serial=serial_generate(model,tok,ids,96)
        golds.append((text,ids,serial))
    if args.fused_gdn and args.split_projections:p.error('choose one projection fusion strategy')
    if args.split_projections:
        from simpletool.projection_fusion import fuse_projections
        fuse_projections(model,split_output=True)
    elif args.fused_gdn:
        from simpletool.fused_gdn import fuse_gdn_projections
        fuse_gdn_projections(model)
    if args.kernel=='custom':
        from simpletool.kernels import install_int4_kernel
        install_int4_kernel(model)
    if args.row_reuse:
        from simpletool.row_reuse_int4 import install_row_reuse
        install_row_reuse(model)
    if args.compact_metadata:
        from simpletool.compact_metadata import install_compact_metadata
        print('metadata',install_compact_metadata(model),flush=True)
    if args.conv_fusion:
        from simpletool.gdn_conv import install_conv_fusion
        install_conv_fusion(model)
    if args.shared_gdn_seed:
        from simpletool.shared_gdn_seed import install_shared_gdn_seed
        install_shared_gdn_seed(model)
    if args.grouped_qmv:
        from simpletool.grouped_qmv import install_grouped_qmv
        install_grouped_qmv(model)
    if args.stream_grouped_qmv:
        from simpletool.stream_grouped_qmv import install_stream_grouped_qmv
        install_stream_grouped_qmv(model)
    if args.fused_swiglu:
        from simpletool.fused_swiglu_int4 import install_fused_swiglu_int4
        install_fused_swiglu_int4(model)
    from simpletool.compiled import CompiledStep
    compiled=CompiledStep(model,fused_head=args.fused_head) if args.compiled else None
    results=[]
    for text,ids,serial in golds:
        batch=batched_generate(model,tok,ids,96,compiled_step=compiled,decode_chunk_size=args.decode_chunk)
        equal={h:serial['channels'][h]['token_ids']==batch['channels'][h]['token_ids'] for h in HEADS}
        results.append({'prompt':text,'token_equal':equal,'serial':serial,'batch':batch})
        print(text,equal,flush=True)
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
    if not all(all(r['token_equal'].values()) for r in results):raise SystemExit(2)


if __name__=='__main__':main()
