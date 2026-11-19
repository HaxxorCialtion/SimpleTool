"""Stream four inputs at a time to reduce the batched QMV register footprint.

Same single-vector FP32 arithmetic as grouped_qmv, with packed rows reused
across branches. Apple MLX QMV arithmetic attribution: third_party/MLX_LICENSE.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from .compact_metadata import CompactInt4Linear,CompactInt4Embedding

_STREAM=mx.fast.metal_kernel(name='simpletool_int4_stream_grouped_qmv',
    input_names=['x','w','scales','biases'],output_names=['y'],source=r'''
        const uint lane=thread_index_in_simdgroup;
        const uint row0=(thread_position_in_grid.x/32)*ROWS;
        if (row0>=N) return;
        float result[ROWS][M];
        for (uint r=0;r<ROWS;++r) for (uint m=0;m<M;++m) result[r][m]=0.0f;
        for (uint block=0;block<K;block+=512) {
            const uint base=block+lane*16;
            float partial[ROWS][M],sum[M];
            for (uint r=0;r<ROWS;++r) for (uint m=0;m<M;++m) partial[r][m]=0.0f;
            for (uint m=0;m<M;++m) sum[m]=0.0f;
            for (uint i=0;i<4;++i) {
                float values[M][4];
                for (uint m=0;m<M;++m) {
                    const uint offset=m*K+base+i*4;
                    sum[m]+=x[offset]+x[offset+1]+x[offset+2]+x[offset+3];
                    values[m][0]=x[offset];values[m][1]=x[offset+1]/16.0f;
                    values[m][2]=x[offset+2]/256.0f;values[m][3]=x[offset+3]/4096.0f;
                }
                for (uint r=0;r<ROWS;++r) {
                    const uint row=min(row0+r,uint(N-1));
                    const device ushort* words=(const device ushort*)(w+row*(K/8)+base/8);
                    const ushort packed=words[i];
                    for (uint m=0;m<M;++m) {
                        partial[r][m]+=(values[m][0]*(packed&0x000f)
                            +values[m][1]*(packed&0x00f0)
                            +values[m][2]*(packed&0x0f00)
                            +values[m][3]*(packed&0xf000));
                    }
                }
            }
            for (uint r=0;r<ROWS;++r) {
                const uint row=min(row0+r,uint(N-1));
                const float s=float(scales[row*(K/64)+base/64]);
                const float b=float(biases[row*(K/64)+base/64]);
                for (uint m=0;m<M;++m) result[r][m]+=s*partial[r][m]+sum[m]*b;
            }
        }
        for (uint r=0;r<ROWS;++r) for (uint m=0;m<M;++m) {
            const float value=simd_sum(result[r][m]);
            if (lane==0 && row0+r<N) y[m*N+row0+r]=value;
        }
    ''')

def stream_grouped_qmv(x,w,scales,biases,rows=4,threads=128):
    k=x.shape[-1];m=x.size//k;n=w.shape[0]
    if (x.dtype!=mx.float32 or not 1<=m<=4 or k%512 or w.shape[1]*8!=k
        or rows not in (1,2,4,8) or threads not in (64,128,256)
        or scales.dtype not in (mx.float16,mx.float32) or biases.dtype!=scales.dtype):
        raise ValueError('Stream grouped QMV needs FP32 B1..4, int4 group64, K divisible by512')
    return _STREAM(inputs=[x,w,scales,biases],template=[('K',k),('N',n),('M',m),('ROWS',rows)],
        grid=(math.ceil(n/(rows*(threads//32)))*threads,1,1),threadgroup=(threads,1,1),
        output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.float32])[0]

def use_stream(module,x):
    m=x.size//x.shape[-1];n=module.weight.shape[0];k=x.shape[-1]
    # Actual 4B microbench: at M4 the vocabulary and long down-projection
    # retain row-reuse; smaller live batches can use grouped arithmetic.
    return (x.dtype==mx.float32 and 2<=m<=4 and k%512==0 and n>=256
        and not (m==4 and (n>=65536 or k>4096)))

class StreamGroupedLinear(CompactInt4Linear):
    def __call__(self,x):
        if use_stream(self,x):
            out=stream_grouped_qmv(x,self.weight,self.scales,self.biases,*self._stream_grouped_config)
            return out+self.bias if 'bias' in self else out
        return super().__call__(x)

class StreamGroupedEmbedding(CompactInt4Embedding):
    def as_linear(self,x):
        if use_stream(self,x):
            return stream_grouped_qmv(x,self.weight,self.scales,self.biases,*self._stream_grouped_config)
        return super().as_linear(x)

def install_stream_grouped_qmv(model,rows=4,threads=128):
    modules=[module for _,module in model.named_modules()
        if isinstance(module,(nn.QuantizedLinear,nn.QuantizedEmbedding))]
    if any(not isinstance(m,(CompactInt4Linear,CompactInt4Embedding)) for m in modules):
        raise ValueError('Install compact int4 metadata before stream grouped QMV')
    for module in modules:
        module.__class__=StreamGroupedEmbedding if isinstance(module,CompactInt4Embedding) else StreamGroupedLinear
        module._stream_grouped_config=(rows,threads)
