"""Warm only unique int4 Metal specializations; never allocate branch caches."""
import time
import mlx.core as mx
from .kernels import SmallBatchLinear,SmallBatchEmbedding,int4_matmul
from .split_projection import SplitProjection
from .row_reuse_int4 import RowReuseLinear,RowReuseEmbedding,row_reuse_int4
from .compact_metadata import CompactInt4Linear,CompactInt4Embedding
from .compact_qmv import compact_qmv
from .stream_grouped_qmv import StreamGroupedLinear,StreamGroupedEmbedding,use_stream,stream_grouped_qmv
from .fused_swiglu_int4 import FusedSwiGLUMLP,fused_swiglu_int4


def warm_int4_kernels(model,batches=(2,3,4)):
    started=time.perf_counter();seen=set();records=[]
    # The split module owns a merged Linear; do not also warm its unused
    # concatenated-output kernel.
    nested={id(m.linear) for _,m in model.named_modules() if isinstance(m,SplitProjection)}
    fused_mlps=[m for _,m in model.named_modules() if isinstance(m,FusedSwiGLUMLP)]
    fused_members={id(p) for m in fused_mlps for p in (m.gate_proj,m.up_proj)}
    for _,module in model.named_modules():
        if isinstance(module,SplitProjection):
            linear=module.linear;kind='split';sizes=module.sizes
        elif isinstance(module,(SmallBatchLinear,SmallBatchEmbedding)) and id(module) not in nested:
            linear=module;kind='shared';sizes=(module.weight.shape[0],)
            if isinstance(module,(RowReuseLinear,RowReuseEmbedding)) and module.weight.shape[0]>=256:
                kind='row_reuse'
        else:continue
        if (linear.bits,linear.group_size,linear.mode)!=(4,64,'affine'):continue
        k=linear.weight.shape[1]*8
        if linear.scales.dtype not in (mx.float32,mx.float16):continue
        local_batches=tuple(batches)
        if isinstance(module,(CompactInt4Linear,CompactInt4Embedding)) and k%512==0 and linear.weight.shape[0]%8==0:
            local_batches=(1,)+local_batches
        for batch in local_batches:
            if batch>1 and id(module) in fused_members:continue
            config=module._row_reuse_config if kind=='row_reuse' else ()
            batch_kind='compact_qmv' if batch==1 and isinstance(module,(CompactInt4Linear,CompactInt4Embedding)) else kind
            if batch_kind=='compact_qmv':config=()
            x=mx.zeros((batch,1,k),mx.float32)
            if isinstance(module,(StreamGroupedLinear,StreamGroupedEmbedding)) and use_stream(module,x):
                batch_kind='stream_grouped_qmv';config=module._stream_grouped_config
            key=(batch_kind,batch,k,sizes,config,str(linear.scales.dtype))
            if key in seen:continue
            seen.add(key)
            start=time.perf_counter()
            if batch_kind=='stream_grouped_qmv':
                result=stream_grouped_qmv(x,linear.weight,linear.scales,linear.biases,*config)
            elif batch==1 and isinstance(module,(CompactInt4Linear,CompactInt4Embedding)):
                result=compact_qmv(x,linear.weight,linear.scales,linear.biases)
            elif kind=='split':result=module(x)
            elif kind=='row_reuse':result=row_reuse_int4(x,linear.weight,linear.scales,linear.biases,*config)
            else:result=int4_matmul(x,linear.weight,linear.scales,linear.biases)
            mx.eval(result)
            records.append({'kind':batch_kind,'batch':batch,'input_width':k,'output_widths':list(sizes),
                            'row_config':list(config),'metadata_dtype':str(linear.scales.dtype),
                            'wall_s':time.perf_counter()-start})
    for mlp in fused_mlps:
        gate,up=mlp.gate_proj,mlp.up_proj;n,k=gate.weight.shape[0],gate.weight.shape[1]*8
        for batch in batches:
            if batch not in (2,3,4):continue
            config=mlp._swiglu_config[batch]
            key=('fused_swiglu',batch,k,n,config,str(gate.scales.dtype))
            if key in seen:continue
            seen.add(key);fused_start=time.perf_counter()
            mx.eval(fused_swiglu_int4(mx.zeros((batch,1,k),mx.float32),gate,up,*config))
            records.append({'kind':'fused_swiglu','batch':batch,'input_width':k,'output_widths':[n],
                'row_config':list(config),'metadata_dtype':str(gate.scales.dtype),'wall_s':time.perf_counter()-fused_start})
    # This warms custom projection pipelines, not MLX/GDN/attention kernels or
    # shape-dependent whole-model compiled graphs. Device/driver owns PSO cache.
    return {'wall_s':time.perf_counter()-started,'specializations':records,
            'scope':'custom int4 projections only; compiled model graphs excluded'}
