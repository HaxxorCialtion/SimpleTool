"""Explicit FP32 FMA experiment for packed int4/group64 B2..4 projections.

FMA can change rounding when contraction is disabled. The local compiler
already contracts row_reuse; measured outputs match. Same immutable int4 weights,
metadata and input; no global fast-math, FP16 activation or dequantized matrix.
Must pass actual greedy-token validation before considering adoption.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from .compact_metadata import CompactInt4Linear,CompactInt4Embedding

SOURCE=r'''
    const uint row0=(thread_position_in_grid.x/32)*ROWS;
    const uint lane=thread_index_in_simdgroup;
    if(row0>=N) return;
    float acc[ROWS][M];
    for(uint r=0;r<ROWS;++r) for(uint m=0;m<M;++m) acc[r][m]=0.0f;
    for(uint word=lane;word<K/8;word+=32) {
        uint packed[ROWS];float s[ROWS],b[ROWS];
        for(uint r=0;r<ROWS;++r) {
            const uint row=row0+r;
            packed[r]=row<N?w[row*(K/8)+word]:0u;
            s[r]=row<N?float(scales[row*(K/64)+word/8]):0.0f;
            b[r]=row<N?float(biases[row*(K/64)+word/8]):0.0f;
        }
        for(uint j=0;j<8;++j) {
            float input[M];for(uint m=0;m<M;++m) input[m]=x[m*K+word*8+j];
            for(uint r=0;r<ROWS;++r) {
                const float code=float((packed[r]>>(4*j))&15);
                const float value=DQ_FMA ? metal::fma(code,s[r],b[r]) : code*s[r]+b[r];
                for(uint m=0;m<M;++m) {
                    if constexpr(ACC_FMA) acc[r][m]=metal::fma(value,input[m],acc[r][m]);
                    else acc[r][m]+=value*input[m];
                }
            }
        }
    }
    for(uint r=0;r<ROWS;++r) for(uint m=0;m<M;++m) {
        const float sum=simd_sum(acc[r][m]);
        if(lane==0 && row0+r<N) y[m*N+row0+r]=sum;
    }
'''
_KERNEL=mx.fast.metal_kernel(name='simpletool_fp32_fma_int4',input_names=['x','w','scales','biases'],output_names=['y'],source=SOURCE)
MODES={'control':(False,False),'acc':(True,False),'dequant':(False,True),'both':(True,True)}
def fma_int4(x,w,scales,biases,mode='acc',rows=8,threads=256):
    k=x.shape[-1];m=x.size//k;n=w.shape[0]
    if (mode not in MODES or x.dtype!=mx.float32 or w.dtype!=mx.uint32 or not 1<=m<=4 or k%64
        or w.shape!=(n,k//8) or scales.shape!=(n,k//64) or biases.shape!=scales.shape
        or scales.dtype not in (mx.float16,mx.float32) or biases.dtype!=scales.dtype
        or rows not in (1,2,4,8) or threads not in (64,128,256)):
        raise ValueError('FMA candidate requires FP32 B1..4, packed affine int4/group64, valid tile and metadata')
    acc,dq=MODES[mode]
    return _KERNEL(inputs=[x,w,scales,biases],
        template=[('K',k),('N',n),('M',m),('ROWS',rows),('ACC_FMA',acc),('DQ_FMA',dq)],
        grid=(math.ceil(n/(rows*(threads//32)))*threads,1,1),threadgroup=(threads,1,1),
        output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.float32])[0]

def eligible(module,x):
    return (x.dtype==mx.float32 and (x.ndim==2 or (x.ndim==3 and x.shape[1]==1))
            and 2<=x.size//x.shape[-1]<=4 and module.weight.shape[0]>=256)
class FMALinear(CompactInt4Linear):
    def __call__(self,x):
        if not eligible(self,x):return super().__call__(x)
        y=fma_int4(x,self.weight,self.scales,self.biases,self._fma_mode,*self._row_reuse_config)
        return y+self.bias if 'bias' in self else y
class FMAEmbedding(CompactInt4Embedding):
    def as_linear(self,x):
        if not eligible(self,x):return super().as_linear(x)
        return fma_int4(x,self.weight,self.scales,self.biases,self._fma_mode,*self._row_reuse_config)

def install_fma_int4(model,mode='acc'):
    if mode not in MODES:raise ValueError('Invalid FMA mode')
    modules=[m for _,m in model.named_modules() if isinstance(m,(nn.QuantizedLinear,nn.QuantizedEmbedding))]
    if any(type(m) not in (CompactInt4Linear,CompactInt4Embedding) for m in modules):
        raise ValueError('FMA candidate requires original compact int4 modules')
    for m in modules:
        m.__class__=FMAEmbedding if isinstance(m,CompactInt4Embedding) else FMALinear
        m._fma_mode=mode
