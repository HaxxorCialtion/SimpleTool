"""Single-vector affine-int4 qmv, widening compact metadata in registers.

Load/qdot arithmetic adapted from Apple's MLX qmv_fast_impl (Copyright
2023-2024 Apple Inc.; MIT license in third_party/MLX_LICENSE). Scope is M=1,
group64, K divisible by512 and N divisible by8. Unaligned shapes retain stock
MLX dispatch; its generic QMV uses a different reduction than the fast path.
"""
import math
import mlx.core as mx


_QMV=mx.fast.metal_kernel(name='simpletool_int4_compact_qmv',
    input_names=['x','w','scales','biases'],output_names=['y'],source=r'''
        const uint lane=thread_index_in_simdgroup;
        const uint row0=(thread_position_in_grid.x/32)*4;
        if (row0>=N) return;
        float result[4]={0.0f,0.0f,0.0f,0.0f};
        float values[16];
        for (uint k=0;k<K;k+=512) {
            const uint base=k+lane*16;
            float sum=0.0f;
            for (uint i=0;i<16;i+=4) {
                sum+=x[base+i]+x[base+i+1]+x[base+i+2]+x[base+i+3];
                values[i]=x[base+i];
                values[i+1]=x[base+i+1]/16.0f;
                values[i+2]=x[base+i+2]/256.0f;
                values[i+3]=x[base+i+3]/4096.0f;
            }
            for (uint r=0;r<4;++r) {
                const uint row=min(row0+r,uint(N-1));
                const device ushort* words=(const device ushort*)(w+row*(K/8)+base/8);
                float acc=0.0f;
                for (uint i=0;i<4;++i) {
                    acc+=(values[4*i]*(words[i]&0x000f)
                          +values[4*i+1]*(words[i]&0x00f0)
                          +values[4*i+2]*(words[i]&0x0f00)
                          +values[4*i+3]*(words[i]&0xf000));
                }
                const float s=float(scales[row*(K/64)+base/64]);
                const float b=float(biases[row*(K/64)+base/64]);
                result[r]+=s*acc+sum*b;
            }
        }
        for (uint r=0;r<4;++r) {
            const float value=simd_sum(result[r]);
            if (lane==0 && row0+r<N) y[row0+r]=value;
        }
    ''')


def compact_qmv(x,w,scales,biases):
    k=x.shape[-1];n=w.shape[0]
    if x.dtype!=mx.float32 or x.size!=k or k%512 or n%8:
        raise ValueError('Compact QMV requires one FP32 vector, K divisible by512 and N divisible by8')
    if scales.dtype!=mx.float16 or biases.dtype!=mx.float16 or w.shape[1]*8!=k:
        raise ValueError('Compact QMV requires int4 weights with compact scale/bias')
    return _QMV(inputs=[x,w,scales,biases],template=[('K',k),('N',n)],
                grid=(math.ceil(n/8)*64,1,1),threadgroup=(64,1,1),
                output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.float32])[0]
