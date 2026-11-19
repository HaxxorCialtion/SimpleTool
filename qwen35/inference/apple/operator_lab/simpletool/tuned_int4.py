"""Experimental shared-weight GEMV layouts; no change to weight representation."""
import math
import mlx.core as mx
import mlx.nn as nn
from .kernels import SmallBatchLinear, SmallBatchEmbedding, _supported, int4_matmul

_KERNEL = mx.fast.metal_kernel(
    name='simpletool_int4_lane_tuning',
    input_names=['x','w','scales','biases'], output_names=['y'],
    source=r'''
        const uint row = thread_position_in_grid.x / LANES;
        const uint lane = thread_index_in_simdgroup % LANES;
        if (row >= N) return;
        float acc[M];
        for (uint m=0; m<M; ++m) acc[m]=0.0f;
        for (uint word=lane; word<K/8; word+=LANES) {
            const uint packed = w[row*(K/8)+word];
            const uint group = word/8;
            const float s = scales[row*(K/64)+group];
            const float b = biases[row*(K/64)+group];
            if constexpr (VECTOR) {
                float4 lo = float4((uint4(packed) >> uint4(0,4,8,12)) & 15u)*s+b;
                float4 hi = float4((uint4(packed) >> uint4(16,20,24,28)) & 15u)*s+b;
                auto xv = reinterpret_cast<const device float4*>(x);
                for (uint m=0; m<M; ++m) {
                    acc[m] += dot(lo,xv[m*(K/4)+word*2]);
                    acc[m] += dot(hi,xv[m*(K/4)+word*2+1]);
                }
            } else {
                for (uint j=0; j<8; ++j) {
                    const float value = float((packed >> (4*j)) & 15)*s+b;
                    for (uint m=0; m<M; ++m) acc[m] += value*x[m*K+word*8+j];
                }
            }
        }
        for (uint m=0; m<M; ++m) {
            float sum = acc[m];
            if constexpr (LANES == 32) sum = simd_sum(sum);
            else {
                for (uint offset=1; offset<LANES; offset*=2)
                    sum += simd_shuffle_xor(sum,offset);
            }
            if (lane==0) y[m*N+row]=sum;
        }
    ''')


def tuned_int4(x,w,scales,biases,lanes=8,vector=False,threads=128):
    k=x.shape[-1];m=x.size//k;n=w.shape[0]
    if x.dtype != mx.float32 or scales.dtype != mx.float32 or biases.dtype != mx.float32:
        raise ValueError('experimental kernel requires FP32 floating tensors')
    if not 1<=m<=8 or k%64 or w.shape[1]*8!=k or lanes not in (4,8,16,32):
        raise ValueError('unsupported kernel shape')
    return _KERNEL(inputs=[x,w,scales,biases],
        template=[('M',m),('N',n),('K',k),('LANES',lanes),('VECTOR',vector)],
        grid=(math.ceil(n*lanes/threads)*threads,1,1),threadgroup=(threads,1,1),
        output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.float32])[0]


class TunedLinear(SmallBatchLinear):
    def __call__(self,x):
        if not _supported(self,x) or x.dtype != mx.float32:
            return super().__call__(x)
        out=tuned_int4(x,self.weight,self.scales,self.biases,*self._int4_tuning)
        return out+self.bias if 'bias' in self else out


class TunedEmbedding(SmallBatchEmbedding):
    def as_linear(self,x):
        if not _supported(self,x) or x.dtype != mx.float32:
            return super().as_linear(x)
        return tuned_int4(x,self.weight,self.scales,self.biases,*self._int4_tuning)


def install_tuned_kernel(model,lanes=8,vector=False,threads=128):
    for _,module in model.named_modules():
        if isinstance(module,nn.QuantizedLinear):module.__class__=TunedLinear
        elif isinstance(module,nn.QuantizedEmbedding):module.__class__=TunedEmbedding
        else:continue
        module._int4_tuning=(lanes,vector,threads)
