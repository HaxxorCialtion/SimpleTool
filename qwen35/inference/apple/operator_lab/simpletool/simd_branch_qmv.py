"""Experimental int4 QMV: one branch per SIMD group, optional shared packed tile.

Keeps the verified compact single-vector FP32 arithmetic/reduction order.
The split layout reduces per-thread branch accumulators. Shared mode stages
each packed tile/metadata once for the cooperating branch SIMD groups.
Apple MLX QMV arithmetic attribution: third_party/MLX_LICENSE.
"""
import math
import mlx.core as mx

SOURCE=r'''
    const uint lane=thread_index_in_simdgroup;
    const uint branch=simdgroup_index_in_threadgroup;
    const uint row0=threadgroup_position_in_grid.x*ROWS;
    threadgroup ushort4 packed_tile[ROWS*32];
    threadgroup float scale_tile[ROWS*32],bias_tile[ROWS*32];
    float result[ROWS];
    for(uint r=0;r<ROWS;++r) result[r]=0.0f;
    for(uint block=0;block<K;block+=512) {
        const uint base=block+lane*16;
        if constexpr(SHARED) {
            for(uint r=branch;r<ROWS;r+=M) {
                const uint row=min(row0+r,uint(N-1));
                const device ushort4* words=(const device ushort4*)(w+row*(K/8)+base/8);
                packed_tile[r*32+lane]=words[0];
                scale_tile[r*32+lane]=float(scales[row*(K/64)+base/64]);
                bias_tile[r*32+lane]=float(biases[row*(K/64)+base/64]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        float partial[ROWS],sum=0.0f;
        for(uint r=0;r<ROWS;++r) partial[r]=0.0f;
        for(uint i=0;i<4;++i) {
            const uint offset=branch*K+base+i*4;
            sum+=x[offset]+x[offset+1]+x[offset+2]+x[offset+3];
            const float a=x[offset],b=x[offset+1]/16.0f,c=x[offset+2]/256.0f,d=x[offset+3]/4096.0f;
            for(uint r=0;r<ROWS;++r) {
                const uint row=min(row0+r,uint(N-1));
                ushort packed;
                if constexpr(SHARED) packed=packed_tile[r*32+lane][i];
                else packed=((const device ushort*)(w+row*(K/8)+base/8))[i];
                partial[r]+=(a*(packed&0x000f)+b*(packed&0x00f0)+c*(packed&0x0f00)+d*(packed&0xf000));
            }
        }
        for(uint r=0;r<ROWS;++r) {
            const uint row=min(row0+r,uint(N-1));
            const float s=SHARED ? scale_tile[r*32+lane] : float(scales[row*(K/64)+base/64]);
            const float b=SHARED ? bias_tile[r*32+lane] : float(biases[row*(K/64)+base/64]);
            result[r]+=s*partial[r]+sum*b;
        }
        if constexpr(SHARED) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for(uint r=0;r<ROWS;++r) {
        const float value=simd_sum(result[r]);
        if(lane==0 && row0+r<N) y[branch*N+row0+r]=value;
    }
'''
_KERNEL=mx.fast.metal_kernel(name='simpletool_simd_branch_qmv',input_names=['x','w','scales','biases'],output_names=['y'],source=SOURCE)

def simd_branch_qmv(x,w,scales,biases,rows=4,shared=True):
    k=x.shape[-1];m=x.size//k;n=w.shape[0]
    if (x.dtype!=mx.float32 or w.dtype!=mx.uint32 or not 1<=m<=4 or k%512
        or w.shape!=(n,k//8) or rows not in (1,2,4,8)
        or scales.shape!=(n,k//64) or biases.shape!=scales.shape
        or scales.dtype not in (mx.float16,mx.float32) or biases.dtype!=scales.dtype):
        raise ValueError('SIMD-branch QMV requires FP32 B1..4 and packed affine int4/group64, K%512=0')
    threads=32*m
    return _KERNEL(inputs=[x,w,scales,biases],template=[('M',m),('N',n),('K',k),('ROWS',rows),('SHARED',shared)],
        grid=(math.ceil(n/rows)*threads,1,1),threadgroup=(threads,1,1),
        output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.float32])[0]
