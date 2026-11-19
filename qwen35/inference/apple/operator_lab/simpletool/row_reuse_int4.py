"""Reuse FP32 input registers across multiple int4 rows in one SIMD group."""
import math
import mlx.core as mx
import mlx.nn as nn
from .kernels import SmallBatchLinear,SmallBatchEmbedding,_supported


_ROW_REUSE=mx.fast.metal_kernel(
    name='simpletool_int4_row_reuse',
    input_names=['x','w','scales','biases'],output_names=['y'],
    source=r'''
        const uint row0=(thread_position_in_grid.x/32)*ROWS;
        const uint lane=thread_index_in_simdgroup;
        if (row0>=N) return;
        float acc[ROWS][M];
        for (uint r=0;r<ROWS;++r)
            for (uint m=0;m<M;++m) acc[r][m]=0.0f;
        for (uint word=lane;word<K/8;word+=32) {
            uint packed[ROWS];float s[ROWS],b[ROWS];
            for (uint r=0;r<ROWS;++r) {
                const uint row=row0+r;
                packed[r]=row<N?w[row*(K/8)+word]:0u;
                s[r]=row<N?scales[row*(K/64)+word/8]:0.0f;
                b[r]=row<N?biases[row*(K/64)+word/8]:0.0f;
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
        for (uint r=0;r<ROWS;++r)
            for (uint m=0;m<M;++m) {
                const float sum=simd_sum(acc[r][m]);
                if (lane==0 && row0+r<N) y[m*N+row0+r]=sum;
            }
    ''')


def row_reuse_int4(x,w,scales,biases,rows=2,threads=128):
    k=x.shape[-1];m=x.size//k;n=w.shape[0]
    if x.dtype!=mx.float32 or scales.dtype not in (mx.float32,mx.float16) or biases.dtype!=scales.dtype:
        raise ValueError('Row-reuse requires FP32 compute and matching FP32/compact metadata')
    if not 1<=m<=4 or rows not in (1,2,4,8) or k%64 or w.shape[1]*8!=k:
        raise ValueError('Unsupported row-reuse shape')
    return _ROW_REUSE(inputs=[x,w,scales,biases],
        template=[('M',m),('N',n),('K',k),('ROWS',rows)],
        grid=(math.ceil(n/(rows*(threads//32)))*threads,1,1),
        threadgroup=(threads,1,1),output_shapes=[(*x.shape[:-1],n)],
        output_dtypes=[mx.float32])[0]


class RowReuseLinear(SmallBatchLinear):
    def __call__(self,x):
        if not _supported(self,x) or x.dtype!=mx.float32 or self.weight.shape[0]<256:return super().__call__(x)
        out=row_reuse_int4(x,self.weight,self.scales,self.biases,*self._row_reuse_config)
        return out+self.bias if 'bias' in self else out


class RowReuseEmbedding(SmallBatchEmbedding):
    def as_linear(self,x):
        if not _supported(self,x) or x.dtype!=mx.float32 or self.weight.shape[0]<256:return super().as_linear(x)
        return row_reuse_int4(x,self.weight,self.scales,self.biases,*self._row_reuse_config)


def install_row_reuse(model,rows=None,threads=None):
    for _,module in model.named_modules():
        if isinstance(module,nn.QuantizedLinear):module.__class__=RowReuseLinear
        elif isinstance(module,nn.QuantizedEmbedding):module.__class__=RowReuseEmbedding
        else:continue
        n=module.weight.shape[0];k=module.weight.shape[1]*8
        # Initial M4 Pro policy from rotating microbench. Never dequantize.
        selected=(4,256) if n>=65536 else (8,256) if k<=2560 else (8,128)
        module._row_reuse_config=(rows or selected[0],threads or selected[1])
