"""Split-K online softmax, reusing each prefix load across branch queries."""
import mlx.core as mx


_PARTIAL = mx.fast.metal_kernel(
    name='simpletool_prefix_online_softmax',
    input_names=['Q','KP','VP','KT','VT','length'],output_names=['partial','maxima','sums'],
    ensure_row_contiguous=False,
    source=r'''
        const uint lane = thread_index_in_simdgroup;
        const uint group = thread_position_in_grid.x / 32;
        const uint h = group / S;
        const uint split = group % S;
        const uint kh = h / R;
        const uint total = P + uint(length);
        const uint begin = total * split / S;
        const uint end = total * (split+1) / S;
        float qr[B][D/32], acc[B][D/32], maximum[B], denom[B];
        for (uint b=0;b<B;++b) {
            maximum[b]=-INFINITY; denom[b]=0.0f;
            for (uint j=0;j<D/32;++j) {
                const uint d=lane+j*32;
                qr[b][j]=float(Q[b*Q_strides[0]+h*Q_strides[1]+d*Q_strides[3]]);
                acc[b][j]=0.0f;
            }
        }
        for (uint t=begin;t<end;++t) {
            float score[B], old_scale[B], new_scale[B];
            for (uint b=0;b<B;++b) score[b]=0.0f;
            for (uint j=0;j<D/32;++j) {
                const uint d=lane+j*32;
                const float pk=t<P ? float(KP[kh*KP_strides[1]+t*KP_strides[2]+d*KP_strides[3]]) : 0.0f;
                for (uint b=0;b<B;++b) {
                    const float k=t<P ? pk : float(KT[b*KT_strides[0]+kh*KT_strides[1]+(t-P)*KT_strides[2]+d*KT_strides[3]]);
                    score[b]+=qr[b][j]*k;
                }
            }
            for (uint b=0;b<B;++b) {
                score[b]=simd_sum(score[b])*rsqrt(float(D));
                const float m=max(maximum[b],score[b]);
                old_scale[b]=exp(maximum[b]-m);
                new_scale[b]=exp(score[b]-m);
                maximum[b]=m;
                denom[b]=denom[b]*old_scale[b]+new_scale[b];
            }
            for (uint j=0;j<D/32;++j) {
                const uint d=lane+j*32;
                const float pv=t<P ? float(VP[kh*VP_strides[1]+t*VP_strides[2]+d*VP_strides[3]]) : 0.0f;
                for (uint b=0;b<B;++b) {
                    const float v=t<P ? pv : float(VT[b*VT_strides[0]+kh*VT_strides[1]+(t-P)*VT_strides[2]+d*VT_strides[3]]);
                    acc[b][j]=acc[b][j]*old_scale[b]+v*new_scale[b];
                }
            }
        }
        for (uint b=0;b<B;++b) {
            const uint row=(b*H+h)*S+split;
            if (lane==0) {maxima[row]=maximum[b]; sums[row]=denom[b];}
            for (uint j=0;j<D/32;++j) partial[row*D+lane+j*32]=acc[b][j];
        }
    ''')

_REDUCE = mx.fast.metal_kernel(
    name='simpletool_prefix_softmax_reduce',
    input_names=['partial','maxima','sums'],output_names=['output'],
    source=r'''
        const uint lane=thread_index_in_simdgroup;
        const uint row=thread_position_in_grid.x/32;
        const float m=lane<S ? maxima[row*S+lane] : -INFINITY;
        const float maximum=simd_max(m);
        const float weight=lane<S ? exp(m-maximum) : 0.0f;
        const float local=lane<S ? sums[row*S+lane]*weight : 0.0f;
        const float denominator=simd_sum(local);
        for (uint j=0;j<D/32;++j) {
            const uint d=lane+j*32;
            float value=0.0f;
            for (uint s=0;s<S;++s)
                value+=simd_broadcast(weight,s)*partial[(row*S+s)*D+d];
            output[row*D+d]=T(value/denominator);
        }
    ''')


def prefix_sdpa(q,cache,tail_k,tail_v):
    b,h,_,d=q.shape
    p=cache.prefix_length
    splits=32
    length=cache.used if hasattr(cache,'used') else mx.array(tail_k.shape[2],dtype=mx.int32)
    if not isinstance(length,mx.array):length=mx.array(length,dtype=mx.int32)
    partial,maximum,sums=_PARTIAL(
        inputs=[q,cache.prefix_k,cache.prefix_v,tail_k,tail_v,length],
        template=[('B',b),('H',h),('D',d),('R',h//cache.prefix_k.shape[1]),('P',p),('S',splits)],
        grid=(h*splits*32,1,1),threadgroup=(128,1,1),
        output_shapes=[(b,h,splits,d),(b,h,splits),(b,h,splits)],
        output_dtypes=[mx.float32,mx.float32,mx.float32])
    return _REDUCE(inputs=[partial,maximum,sums],template=[('D',d),('S',splits),('T',q.dtype)],
        grid=(b*h*32,1,1),threadgroup=(128,1,1),
        output_shapes=[q.shape],output_dtypes=[q.dtype])[0]
