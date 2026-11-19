import math
import mlx.core as mx
import mlx.nn as nn


_INT4 = mx.fast.metal_kernel(
    name="simpletool_int4_shared_rows",
    input_names=["x", "w", "scales", "biases"], output_names=["y"],
    source=r'''
        const uint row = thread_position_in_grid.x / 32;
        const uint lane = thread_index_in_simdgroup;
        if (row >= N) return;
        float acc[M];
        for (uint m=0; m<M; ++m) acc[m]=0.0f;
        for (uint word=lane; word<K/8; word+=32) {
            const uint packed = w[row*(K/8)+word];
            const uint group = (word*8)/64;
            const float s = float(scales[row*(K/64)+group]);
            const float b = float(biases[row*(K/64)+group]);
            for (uint j=0; j<8; ++j) {
                const float value = float((packed >> (4*j)) & 15)*s+b;
                for (uint m=0; m<M; ++m)
                    acc[m] += value * float(x[m*K+word*8+j]);
            }
        }
        for (uint m=0; m<M; ++m) {
            const float sum = simd_sum(acc[m]);
            if (lane==0) y[m*N+row] = T(sum);
        }
    ''')


def int4_matmul(x, w, scales, biases):
    k = x.shape[-1]
    m = x.size // k
    n = w.shape[0]
    if not 1 <= m <= 8 or k % 64 or w.shape[1] * 8 != k:
        raise ValueError("kernel supports M=1..8, affine int4 group=64")
    return _INT4(inputs=[x, w, scales, biases],
                 template=[("T", x.dtype), ("M", m), ("N", n), ("K", k)],
                 grid=(math.ceil(n / 4) * 128, 1, 1), threadgroup=(128, 1, 1),
                 output_shapes=[(*x.shape[:-1], n)], output_dtypes=[x.dtype])[0]


def _supported(module, x):
    return (module.bits == 4 and module.group_size == 64 and module.mode == "affine"
            and 2 <= x.size // x.shape[-1] <= 4 and x.dtype in (mx.float16, mx.float32))


class SmallBatchLinear(nn.QuantizedLinear):
    def __call__(self, x):
        if not _supported(self, x):
            return super().__call__(x)
        out = int4_matmul(x, self.weight, self.scales, self.biases)
        if "bias" in self:
            out = out + self.bias
        return out


class SmallBatchEmbedding(nn.QuantizedEmbedding):
    def as_linear(self, x):
        if not _supported(self, x):
            return super().as_linear(x)
        return int4_matmul(x, self.weight, self.scales, self.biases)


def install_int4_kernel(model):
    count = 0
    for _, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            module.__class__ = SmallBatchLinear
            count += 1
        elif isinstance(module, nn.QuantizedEmbedding):
            module.__class__ = SmallBatchEmbedding
            count += 1
    return count


def restore_mlx_kernel(model):
    for _,module in model.named_modules():
        if isinstance(module,SmallBatchLinear):module.__class__=nn.QuantizedLinear
        elif isinstance(module,SmallBatchEmbedding):module.__class__=nn.QuantizedEmbedding
