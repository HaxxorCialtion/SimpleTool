"""Within-process A/B: old custom, compiled, row fusion, compiled + fusion."""
import argparse
import copy
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path
import mlx.core as mx
import mlx_lm.models.gated_delta as gd
from mlx.utils import tree_flatten
from simpletool.loading import load_model
from simpletool.decode import HEADS,prompt_ids,prefill,batched_generate
from simpletool.compiled import CompiledStep
from simpletool.projection_fusion import fuse_projections

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--tokens',type=int,default=96)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--strategy',choices=['projections','blocks','split_outputs','row_reuse','chunks','conv','compact_metadata','fused_head','seed_release','shared_gdn_seed','grouped_qmv','stream_grouped_qmv','fused_swiglu','stream_swiglu'],default='projections')
    args=p.parse_args();envpath=Path(inspect.getsourcefile(gd));start_hash=hashlib.sha256(envpath.read_bytes()).hexdigest()
    base,tok=load_model(args.model,kernel='custom')
    if args.strategy in ('conv','compact_metadata','fused_head','seed_release','shared_gdn_seed','grouped_qmv','stream_grouped_qmv','fused_swiglu','stream_swiglu'):
        from simpletool.row_reuse_int4 import install_row_reuse
        install_row_reuse(base)
    if args.strategy in ('fused_head','seed_release','shared_gdn_seed','grouped_qmv','stream_grouped_qmv','fused_swiglu','stream_swiglu'):
        from simpletool.compact_metadata import install_compact_metadata
        install_compact_metadata(base)
    optimized=copy.deepcopy(base,{id(a):a for _,a in tree_flatten(base.parameters())})
    compact_receipt=None
    if args.strategy in ('fused_swiglu','stream_swiglu'):
        if args.strategy=='stream_swiglu':
            from simpletool.stream_grouped_qmv import install_stream_grouped_qmv
            install_stream_grouped_qmv(optimized)
        from simpletool.fused_swiglu_int4 import install_fused_swiglu_int4
        install_fused_swiglu_int4(optimized)
    elif args.strategy=='stream_grouped_qmv':
        from simpletool.stream_grouped_qmv import install_stream_grouped_qmv
        install_stream_grouped_qmv(optimized)
    elif args.strategy=='grouped_qmv':
        from simpletool.grouped_qmv import install_grouped_qmv
        install_grouped_qmv(optimized)
    elif args.strategy=='shared_gdn_seed':
        from simpletool.shared_gdn_seed import install_shared_gdn_seed
        install_shared_gdn_seed(optimized)
    elif args.strategy=='compact_metadata':
        from simpletool.compact_metadata import install_compact_metadata
        compact_receipt=install_compact_metadata(optimized)
    elif args.strategy in ('projections','split_outputs'):
        fuse_projections(optimized,alias_base=base,split_output=args.strategy=='split_outputs')
    elif args.strategy=='conv':
        from simpletool.gdn_conv import install_conv_fusion
        install_conv_fusion(optimized)
    elif args.strategy in ('row_reuse','chunks'):
        from simpletool.row_reuse_int4 import install_row_reuse
        install_row_reuse(optimized)
    elif args.strategy not in ('fused_head','seed_release'):
        from simpletool.block_compile import install_block_compilation
        install_block_compilation(optimized)
    mx.eval(base.parameters(),optimized.parameters());mx.clear_cache()
    active_bytes=mx.get_active_memory()
    banned=tuple(tok.convert_tokens_to_ids(t) for t in ('<|endoftext|>','<|im_end|>'))
    ids=prompt_ids(tok,[{'role':'user','content':'列出手机设置里的所有选项'}])
    source,_=prefill(base,ids);fused_source,_=prefill(optimized,ids)
    modes={'baseline':(base,None,source),'compiled':(base,CompiledStep(base,banned),source),
           'fusion':(optimized,None,fused_source),'compiled_fusion':(optimized,CompiledStep(optimized,banned),fused_source)}
    if args.strategy=='blocks':modes={'baseline':(base,None,source),'blocks':(optimized,None,fused_source)}
    elif args.strategy=='split_outputs':modes={'baseline':modes['baseline'],'compiled':modes['compiled'],
                                             'split':modes['fusion'],'compiled_split':modes['compiled_fusion']}
    elif args.strategy=='row_reuse':modes={'baseline':modes['baseline'],'compiled':modes['compiled'],
                                        'row_reuse':modes['fusion'],'compiled_row_reuse':modes['compiled_fusion']}
    elif args.strategy=='chunks':
        step=CompiledStep(optimized,banned)
        modes={name:(optimized,step,fused_source) for name in ('baseline','chunk2','chunk4','chunk8')}
    elif args.strategy=='conv':modes={'baseline':modes['baseline'],'compiled':modes['compiled'],
                                   'conv':modes['fusion'],'compiled_conv':modes['compiled_fusion']}
    elif args.strategy=='compact_metadata':modes={'baseline':modes['baseline'],'compiled':modes['compiled'],
                                   'compact':modes['fusion'],'compiled_compact':modes['compiled_fusion']}
    elif args.strategy=='fused_head':modes={'baseline':modes['compiled'],
                                          'fused_head':(optimized,CompiledStep(optimized,banned,fused_head=True),fused_source)}
    elif args.strategy=='seed_release':modes={'baseline':(base,CompiledStep(base,banned,release_seed_cache=False),source),
                                            'release':(optimized,CompiledStep(optimized,banned),fused_source)}
    elif args.strategy=='shared_gdn_seed':modes={'baseline':modes['compiled'],
                                              'shared_gdn_seed':modes['compiled_fusion']}
    elif args.strategy=='grouped_qmv':modes={'baseline':modes['compiled'],
                                           'grouped_qmv':modes['compiled_fusion']}
    elif args.strategy=='stream_grouped_qmv':modes={'baseline':modes['compiled'],
                                                  'stream_grouped_qmv':modes['compiled_fusion']}
    elif args.strategy in ('fused_swiglu','stream_swiglu'):modes={'baseline':modes['compiled'],args.strategy:modes['compiled_fusion']}
    chunks={'baseline':1,'chunk2':2,'chunk4':4,'chunk8':8}
    def run(mode,single=False):
        model,compiled,cache=modes[mode];start=time.perf_counter()
        r=batched_generate(model,tok,ids,args.tokens,heads=HEADS[:1] if single else HEADS,
                           prefix_cache=cache,ignore_stops=True,compiled_step=compiled,suppress_tokens=banned,
                           decode_chunk_size=chunks[mode] if args.strategy=='chunks' else 1)
        r['stats']['request_wall_s']=time.perf_counter()-start
        r['stats']['active_memory_bytes']=mx.get_active_memory()
        r['stats']['cached_memory_bytes']=mx.get_cache_memory()
        r['stats']['process_peak_memory_bytes']=mx.get_peak_memory()
        assert all(len(c['token_ids'])==args.tokens and not set(c['token_ids']).intersection(banned) for c in r['channels'].values())
        return r
    warmup={};single_warmup={}
    for mode in modes:
        warmup[mode]=run(mode);single_warmup[mode]=run(mode,True)
        print('warm',mode,warmup[mode]['stats']['request_wall_s'],flush=True)
    equal={mode:{h:r['channels'][h]['token_ids']==warmup['baseline']['channels'][h]['token_ids'] for h in HEADS}
           for mode,r in warmup.items()}
    samples={mode:{'single':[],'four':[]} for mode in modes}
    keys=list(modes)
    for i in range(args.rounds):
        for mode in keys[i:]+keys[:i]:
            order=(False,True) if i%2 else (True,False)
            for single in order:
                r=run(mode,single);name='single' if single else 'four'
                samples[mode][name].append(r['stats'])
                print(mode,name,i,r['stats']['request_wall_s'],flush=True)
    summaries={}
    for mode,groups in samples.items():
        times={key:statistics.median(r['request_wall_s'] for r in rows) for key,rows in groups.items()}
        summaries[mode]={'single_wall_s':times['single'],'four_wall_s':times['four'],
                         'eta_wall':times['single']/times['four'],
                         'four_decode_tps':4*args.tokens/statistics.median(r['decode_s'] for r in groups['four'])}
    for value in summaries.values():value['speedup_vs_baseline']=summaries['baseline']['four_wall_s']/value['four_wall_s']
    result={'model':args.model,'strategy':args.strategy,'tokens_per_head':args.tokens,'compute':'float32','quantization':'affine int4 group64 unchanged',
            'exact_prefix_cache_warm':True,'same_process_rotating':True,'summaries':summaries,'token_equal_fixed_length_vs_baseline':equal,
            'samples':samples,'warmup':{key:r['stats'] for key,r in warmup.items()},
            'gated_delta_file':str(envpath),'gated_delta_sha256':start_hash,
            'gated_delta_sha256_at_end':hashlib.sha256(envpath.read_bytes()).hexdigest(),
            'shared_weight_storage_between_AB_models':True,'compact_metadata_receipt':compact_receipt,
            'metadata_storage_shared_between_AB_models':args.strategy!='compact_metadata',
            'active_memory_after_models_bytes':active_bytes,
            'gdn_packed_enabled':getattr(gd,'_ENABLE_GDN_PACKED',None),'competing_workloads_uncontrolled':True}
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'summaries':summaries,'equal':equal},indent=2))

if __name__=='__main__':main()
