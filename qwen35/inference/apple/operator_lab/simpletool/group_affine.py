"""Experimental group-dot then affine scale. Changes FP32 summation order."""
import math
import mlx.core as mx
import mlx.nn as nn
from .kernels import SmallBatchLinear,SmallBatchEmbedding,_supported

_GROUP = mx.fast.metal_kernel(
    name='simpletool_int4_group_affine',
    input_names=['x','xsum','w','scales','biases'],output_names=['y'],
    source=r'''
        const uint row=thread_position_in_grid.x/32;
        const uint lane=thread_index_in_simdgroup;
        if (row>=N) return;
        float accum[M];
        for (uint m=0;m<M;++m) accum[m]=0.0f;
        for (uint group=lane/8;group<K/64;group+=4) {
            const uint word=group*8+(lane%8);
            const uint packed=w[row*(K/8)+word];
            float part[M];
            for (uint m=0;m<M;++m) part[m]=0.0f;
            for (uint j=0;j<8;++j) {
                const float quant=float((packed>>(4*j))&15);
                for (uint m=0;m<M;++m) part[m]+=quant*x[m*K+word*8+j];
            }
            for (uint m=0;m<M;++m) {
                part[m]+=simd_shuffle_xor(part[m],1);
                part[m]+=simd_shuffle_xor(part[m],2);
                part[m]+=simd_shuffle_xor(part[m],4);
                if (lane%8==0) accum[m]+=part[m]*scales[row*(K/64)+group]
                    +xsum[m*(K/64)+group]*biases[row*(K/64)+group];
            }
        }
        for (uint m=0;m<M;++m) {
            const float value=simd_sum(accum[m]);
            if (lane==0) y[m*N+row]=value;
        }
    ''')


def group_affine(x,w,scales,biases):
    k=x.shape[-1];m=x.size//k;n=w.shape[0]
    if k%256 or not 1<=m<=8 or x.dtype!=mx.float32:
        raise ValueError('group affine requires FP32, K multiple of 256, M=1..8')
    sums=mx.sum(x.reshape(m,k//64,64),axis=-1)
    return _GROUP(inputs=[x,sums,w,scales,biases],
        template=[('M',m),('K',k),('N',n)],grid=(math.ceil(n/4)*128,1,1),threadgroup=(128,1,1),
        output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.float32])[0]


class GroupLinear(SmallBatchLinear):
    def __call__(self,x):
        if not _supported(self,x) or x.dtype!=mx.float32 or x.shape[-1]%256:return super().__call__(x)
        out=group_affine(x,self.weight,self.scales,self.biases)
        return out+self.bias if 'bias' in self else out


class GroupEmbedding(SmallBatchEmbedding):
    def as_linear(self,x):
        if not _supported(self,x) or x.dtype!=mx.float32 or x.shape[-1]%256:return super().as_linear(x)
        return group_affine(x,self.weight,self.scales,self.biases)


def install_group_affine(model):
    for _,m in model.named_modules():
        if isinstance(m,nn.QuantizedLinear):m.__class__=GroupLinear
        elif isinstance(m,nn.QuantizedEmbedding):m.__class__=GroupEmbedding
