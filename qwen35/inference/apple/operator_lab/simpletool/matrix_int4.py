"""Experimental FP32 SIMD-group matrix kernel, padding M<=8 to eight rows."""
import math
import mlx.core as mx
import mlx.nn as nn
from .kernels import SmallBatchLinear,SmallBatchEmbedding,_supported

_MATRIX = mx.fast.metal_kernel(
    name='simpletool_int4_simd_matrix',
    input_names=['x','w','scales','biases'],output_names=['y'],
    header='#include <metal_simdgroup_matrix>\n',
    source=r'''
        const uint tid=thread_position_in_threadgroup.x;
        const uint sg=tid/32;
        const uint row_local=tid/4;
        const uint row=threadgroup_position_in_grid.x*NT+row_local;
        threadgroup float aa[8*(KT+PAD)];
        threadgroup float bb[KT*(NT+PAD)];
        threadgroup float cc[8*(NT+PAD)];
        simdgroup_float8x8 accumulator(0.0f);
        for (uint start=0;start<K;start+=KT) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i=tid;i<8*KT;i+=NT*4)
                aa[(i/KT)*(KT+PAD)+i%KT]=(i/KT<M) ? x[(i/KT)*K+start+i%KT] : 0.0f;
            for (uint word=tid%4;word<KT/8;word+=4) {
                const uint group=(start+word*8)/64;
                const uint packed=row<N ? w[row*(K/8)+start/8+word] : 0;
                const float s=row<N ? scales[row*(K/64)+group] : 0.0f;
                const float b=row<N ? biases[row*(K/64)+group] : 0.0f;
                for (uint j=0;j<8;++j)
                    bb[(word*8+j)*(NT+PAD)+row_local]=float((packed>>(j*4))&15)*s+b;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint inner=0;inner<KT;inner+=8) {
                simdgroup_float8x8 a,b;
                simdgroup_load(a,aa+inner,KT+PAD);
                simdgroup_load(b,bb+inner*(NT+PAD)+sg*8,NT+PAD);
                simdgroup_multiply_accumulate(accumulator,a,b,accumulator);
            }
        }
        simdgroup_store(accumulator,cc+sg*8,NT+PAD);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i=tid;i<8*NT;i+=NT*4) {
            const uint m=i/NT;
            const uint n=threadgroup_position_in_grid.x*NT+i%NT;
            if (m<M && n<N) y[m*N+n]=cc[(i/NT)*(NT+PAD)+i%NT];
        }
    ''')


def matrix_int4(x,w,scales,biases,nt=32,kt=64,pad=8):
    k=x.shape[-1];m=x.size//k;n=w.shape[0]
    if x.dtype!=mx.float32 or scales.dtype!=mx.float32 or biases.dtype!=mx.float32 or not 1<=m<=8 or k%kt or kt%32 or nt not in (8,16,32,64):
        raise ValueError('unsupported SIMD matrix kernel shape / dtype')
    threads=nt*4
    return _MATRIX(inputs=[x,w,scales,biases],template=[('M',m),('N',n),('K',k),('NT',nt),('KT',kt),('PAD',pad)],
        grid=(math.ceil(n/nt)*threads,1,1),threadgroup=(threads,1,1),
        output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.float32])[0]


class MatrixLinear(SmallBatchLinear):
    def __call__(self,x):
        if not _supported(self,x) or x.dtype!=mx.float32:return super().__call__(x)
        out=matrix_int4(x,self.weight,self.scales,self.biases,*self._matrix_tile)
        return out+self.bias if 'bias' in self else out


class MatrixEmbedding(SmallBatchEmbedding):
    def as_linear(self,x):
        if not _supported(self,x) or x.dtype!=mx.float32:return super().as_linear(x)
        return matrix_int4(x,self.weight,self.scales,self.biases,*self._matrix_tile)


def install_matrix_kernel(model,nt=32,kt=64):
    for _,m in model.named_modules():
        if isinstance(m,nn.QuantizedLinear):m.__class__=MatrixLinear
        elif isinstance(m,nn.QuantizedEmbedding):m.__class__=MatrixEmbedding
        else:continue
        m._matrix_tile=(nt,kt)
