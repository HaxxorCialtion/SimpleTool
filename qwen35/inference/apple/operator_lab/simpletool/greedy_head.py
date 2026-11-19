"""Fuse the int4 tied output projection with tiled greedy reduction.

The projection keeps the row-reuse kernel's exact FP32 accumulation order.
Only per-threadgroup candidate values and token IDs leave the first kernel;
the full M*vocab logits array is never allocated for small-batch decode.
"""
import math
import mlx.core as mx
from .row_reuse_int4 import RowReuseLinear,RowReuseEmbedding


_PROJECT=mx.fast.metal_kernel(name='simpletool_int4_greedy_tiles',
    input_names=['x','w','scales','biases','suppressed'],output_names=['values','indices'],source=r'''
        const uint lane=thread_index_in_simdgroup;
        const uint simd=thread_index_in_threadgroup/32;
        const uint row0=(thread_position_in_grid.x/32)*ROWS;
        threadgroup float group_values[NSG*M];
        threadgroup uint group_indices[NSG*M];
        float acc[ROWS][M];
        for (uint r=0;r<ROWS;++r)
            for (uint m=0;m<M;++m) acc[r][m]=0.0f;
        if (row0<N) {
            for (uint word=lane;word<K/8;word+=32) {
                uint packed[ROWS];float s[ROWS],b[ROWS];
                for (uint r=0;r<ROWS;++r) {
                    const uint row=row0+r;
                    packed[r]=row<N?w[row*(K/8)+word]:0u;
                    s[r]=row<N?float(scales[row*(K/64)+word/8]):0.0f;
                    b[r]=row<N?float(biases[row*(K/64)+word/8]):0.0f;
                }
                for (uint j=0;j<8;++j) {
                    float input[M];
                    for (uint m=0;m<M;++m) input[m]=x[m*K+word*8+j];
                    for (uint r=0;r<ROWS;++r) {
                        const float value=float((packed[r]>>(4*j))&15)*s[r]+b[r];
                        for (uint m=0;m<M;++m) acc[r][m]+=value*input[m];
                    }
                }
            }
        }
        for (uint m=0;m<M;++m) {
            float best=-INFINITY;uint best_id=0xffffffffu;
            for (uint r=0;r<ROWS;++r) {
                float value=simd_sum(acc[r][m]);
                const uint token=row0+r;
                for (uint i=0;i<NSUP;++i) if (token==uint(suppressed[i])) value=-INFINITY;
                const bool better=(isnan(value)&&(!isnan(best)||token<best_id))
                    ||(!isnan(best)&&(value>best||(value==best&&token<best_id)));
                if (token<N && better) { best=value;best_id=token; }
            }
            if (lane==0) { group_values[simd*M+m]=best;group_indices[simd*M+m]=best_id; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Every SIMD group, including those covering only invalid tail rows,
        // reaches the barrier. One group reduces the candidates for all M.
        if (simd==0) {
            const uint tile=thread_position_in_grid.x/(NSG*32);
            for (uint m=0;m<M;++m) {
                const float value=lane<NSG?group_values[lane*M+m]:-INFINITY;
                const uint index=lane<NSG?group_indices[lane*M+m]:0xffffffffu;
                const bool any_nan=simd_any(isnan(value));
                const float maximum=simd_max(isnan(value)?-INFINITY:value);
                const uint chosen=simd_min((any_nan?isnan(value):value==maximum)?index:0xffffffffu);
                if (lane==0) { values[m*TILES+tile]=any_nan?NAN:maximum;indices[m*TILES+tile]=chosen; }
            }
        }
    ''')


_REDUCE=mx.fast.metal_kernel(name='simpletool_greedy_tile_reduce',
    input_names=['values','indices'],output_names=['tokens'],source=r'''
        const uint m=thread_position_in_grid.x/32;
        const uint lane=thread_index_in_simdgroup;
        if (m>=M) return;
        float best=-INFINITY;uint best_id=0xffffffffu;
        for (uint tile=lane;tile<TILES;tile+=32) {
            const float value=values[m*TILES+tile];
            const uint index=indices[m*TILES+tile];
            const bool better=(isnan(value)&&(!isnan(best)||index<best_id))
                ||(!isnan(best)&&(value>best||(value==best&&index<best_id)));
            if (better) { best=value;best_id=index; }
        }
        const bool any_nan=simd_any(isnan(best));
        const float maximum=simd_max(isnan(best)?-INFINITY:best);
        const uint chosen=simd_min((any_nan?isnan(best):best==maximum)?best_id:0xffffffffu);
        if (lane==0) tokens[m]=chosen;
    ''')


def greedy_int4(head,x,suppress_tokens=(),rows=None,threads=None):
    if not isinstance(head,(RowReuseLinear,RowReuseEmbedding)):
        raise ValueError('Fused greedy head requires installed row-reuse int4 projection')
    if 'bias' in head:raise ValueError('Biased output heads are unsupported')
    k=x.shape[-1];m=x.size//k;n=head.weight.shape[0]
    if (head.bits,head.group_size,head.mode)!=(4,64,'affine') or x.dtype!=mx.float32 or not 1<=m<=4:
        raise ValueError('Fused greedy head requires B1..4, affine int4 group64 and FP32 activations')
    if any(not isinstance(t,int) or not 0<=t<n for t in suppress_tokens):raise ValueError('Invalid suppressed token')
    if m==1:
        # Preserve the fast stock/compact QMV order for the single remaining
        # branch. This first candidate fuses M=2/3/4 only.
        logits=head.as_linear(x.reshape(m,k)) if isinstance(head,RowReuseEmbedding) else head(x.reshape(m,k))
        if suppress_tokens:logits[:,list(suppress_tokens)]=-mx.inf
        return mx.argmax(logits,axis=-1)
    rows0,threads0=head._row_reuse_config
    rows=rows or rows0;threads=threads or threads0
    if rows not in (1,2,4,8) or threads not in (64,128,256):raise ValueError('Unsupported greedy tile')
    nsg=threads//32;tiles=math.ceil(n/(rows*nsg))
    suppressed=mx.array(suppress_tokens or (0,),mx.int32)
    values,indices=_PROJECT(inputs=[x,head.weight,head.scales,head.biases,suppressed],
        template=[('M',m),('N',n),('K',k),('ROWS',rows),('NSG',nsg),('TILES',tiles),('NSUP',len(suppress_tokens))],
        grid=(tiles*threads,1,1),threadgroup=(threads,1,1),
        output_shapes=[(m,tiles),(m,tiles)],output_dtypes=[mx.float32,mx.uint32])
    return _REDUCE(inputs=[values,indices],template=[('M',m),('TILES',tiles)],
        grid=(128,1,1),threadgroup=(128,1,1),output_shapes=[(m,)],output_dtypes=[mx.uint32])[0]
