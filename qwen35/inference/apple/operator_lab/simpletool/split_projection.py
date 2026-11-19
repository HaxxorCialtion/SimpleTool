"""One GEMV dispatch, several contiguous outputs, no requantization."""
import math
from functools import lru_cache
import mlx.core as mx
import mlx.nn as nn

@lru_cache(None)
def kernel(count):
    stores=[];start='0'
    for i in range(count):
        end='+'.join(f'C{j}' for j in range(i+1))
        stores.append(f'{"if" if i==0 else "else if"} (row < {end}) y{i}[m*C{i}+row-({start})]=sum;')
        start=end
    source=r'''
        const uint row=thread_position_in_grid.x/32;
        const uint lane=thread_index_in_simdgroup;
        if (row>=N) return;
        float acc[M];
        for (uint m=0;m<M;++m) acc[m]=0.0f;
        for (uint word=lane;word<K/8;word+=32) {
            const uint packed=w[row*(K/8)+word];
            const uint group=word/8;
            const float s=float(scales[row*(K/64)+group]);
            const float b=float(biases[row*(K/64)+group]);
            for (uint j=0;j<8;++j) {
                const float value=float((packed>>(4*j))&15)*s+b;
                for (uint m=0;m<M;++m) acc[m]+=value*float(x[m*K+word*8+j]);
            }
        }
        for (uint m=0;m<M;++m) {
            const float sum=simd_sum(acc[m]);
            if (lane==0) { STORES }
        }
    '''.replace('STORES',' '.join(stores))
    return mx.fast.metal_kernel(name=f'simpletool_int4_split_{count}',
        input_names=['x','w','scales','biases'],output_names=[f'y{i}' for i in range(count)],source=source)


class SplitProjection(nn.Module):
    def __init__(self,linear,sizes):
        super().__init__();self.linear=linear;self.sizes=tuple(sizes)

    def __call__(self,x):
        k=x.shape[-1];m=x.size//k;n=sum(self.sizes);linear=self.linear
        if not (2<=m<=4 and x.dtype==mx.float32 and linear.bits==4 and linear.group_size==64 and linear.mode=='affine'):
            out=self.linear(x)
            return mx.split(out,[sum(self.sizes[:i]) for i in range(1,len(self.sizes))],axis=-1)
        return kernel(len(self.sizes))(
            inputs=[x,linear.weight,linear.scales,linear.biases],
            template=[('M',m),('N',n),('K',k)]+[(f'C{i}',v) for i,v in enumerate(self.sizes)],
            grid=(math.ceil(n/4)*128,1,1),threadgroup=(128,1,1),
            output_shapes=[(*x.shape[:-1],v) for v in self.sizes],output_dtypes=[mx.float32]*len(self.sizes))
