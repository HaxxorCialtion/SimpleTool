"""Batched affine-int4 QMV: factor scale/bias out of each 16-value dot.

FP32 arithmetic adapted from Apple's MLX fast single-vector QMV (MIT,
third_party/MLX_LICENSE). All branches reuse packed row loads; each branch
retains the single-vector reduction order. This changes rounding relative to
the older per-weight dequantization kernel and requires real token validation.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from .compact_metadata import CompactInt4Linear,CompactInt4Embedding


_GROUPED=mx.fast.metal_kernel(name='simpletool_int4_grouped_qmv',
    input_names=['x','w','scales','biases'],output_names=['y'],source=r'''
        const uint lane=thread_index_in_simdgroup;
        const uint row0=(thread_position_in_grid.x/32)*ROWS;
        if (row0>=N) return;
        float result[ROWS][M];
        for (uint r=0;r<ROWS;++r) for (uint m=0;m<M;++m) result[r][m]=0.0f;
        for (uint block=0;block<K;block+=512) {
            const uint base=block+lane*16;
            float values[M][16],sum[M];
            for (uint m=0;m<M;++m) {
                sum[m]=0.0f;
                for (uint i=0;i<16;i+=4) {
                    sum[m]+=x[m*K+base+i]+x[m*K+base+i+1]+x[m*K+base+i+2]+x[m*K+base+i+3];
                    values[m][i]=x[m*K+base+i];
                    values[m][i+1]=x[m*K+base+i+1]/16.0f;
                    values[m][i+2]=x[m*K+base+i+2]/256.0f;
                    values[m][i+3]=x[m*K+base+i+3]/4096.0f;
                }
            }
            for (uint r=0;r<ROWS;++r) {
                const uint row=min(row0+r,uint(N-1));
                const device ushort* words=(const device ushort*)(w+row*(K/8)+base/8);
                const float s=float(scales[row*(K/64)+base/64]);
                const float b=float(biases[row*(K/64)+base/64]);
                ushort packed[4];for (uint i=0;i<4;++i) packed[i]=words[i];
                for (uint m=0;m<M;++m) {
                    float acc=0.0f;
                    for (uint i=0;i<4;++i) {
                        acc+=(values[m][4*i]*(packed[i]&0x000f)
                              +values[m][4*i+1]*(packed[i]&0x00f0)
                              +values[m][4*i+2]*(packed[i]&0x0f00)
                              +values[m][4*i+3]*(packed[i]&0xf000));
                    }
                    result[r][m]+=s*acc+sum[m]*b;
                }
            }
        }
        for (uint r=0;r<ROWS;++r) for (uint m=0;m<M;++m) {
            const float value=simd_sum(result[r][m]);
            if (lane==0 && row0+r<N) y[m*N+row0+r]=value;
        }
    ''')


def grouped_qmv(x,w,scales,biases,rows=4,threads=128):
    k=x.shape[-1];m=x.size//k;n=w.shape[0]
    if (x.dtype!=mx.float32 or not 1<=m<=4 or k%512 or w.shape[1]*8!=k
        or rows not in (1,2,4,8) or threads not in (64,128,256)
        or scales.dtype not in (mx.float16,mx.float32) or biases.dtype!=scales.dtype):
        raise ValueError('Grouped QMV needs FP32 B1..4, int4 group64, K divisible by512')
    return _GROUPED(inputs=[x,w,scales,biases],
        template=[('K',k),('N',n),('M',m),('ROWS',rows)],
        grid=(math.ceil(n/(rows*(threads//32)))*threads,1,1),threadgroup=(threads,1,1),
        output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.float32])[0]


class GroupedLinear(CompactInt4Linear):
    def __call__(self,x):
        if x.dtype==mx.float32 and 2<=x.size//x.shape[-1]<=4 and x.shape[-1]%512==0:
            out=grouped_qmv(x,self.weight,self.scales,self.biases,*self._grouped_config)
            return out+self.bias if 'bias' in self else out
        return super().__call__(x)


class GroupedEmbedding(CompactInt4Embedding):
    def as_linear(self,x):
        if x.dtype==mx.float32 and 2<=x.size//x.shape[-1]<=4 and x.shape[-1]%512==0:
            return grouped_qmv(x,self.weight,self.scales,self.biases,*self._grouped_config)
        return super().as_linear(x)


def install_grouped_qmv(model,rows=4,threads=128):
    for _,module in model.named_modules():
        if isinstance(module,CompactInt4Embedding):module.__class__=GroupedEmbedding
        elif isinstance(module,CompactInt4Linear):module.__class__=GroupedLinear
        elif isinstance(module,(nn.QuantizedLinear,nn.QuantizedEmbedding)):
            raise ValueError('Install compact metadata before grouped QMV')
        else:continue
        module._grouped_config=(rows,threads)
