"""Experimental int4 projection policy selected by native Metal GPU timings.

Only measured gate/down and tied vocabulary families change. Single-vector,
prefill and other projections retain the original Compact paths. Per-lane FP32
arithmetic order remains unchanged; complete greedy validation is still required.
The policy is explicit input, never a production default or phone tuning claim.
"""
import mlx.nn as nn

from .compact_metadata import CompactInt4Linear, CompactInt4Embedding
from .fma_int4 import eligible
from .row_reuse_int4 import row_reuse_int4
from .scalar_int4 import scalar_int4
from .unroll_int4 import unroll_int4


def project(module, x):
    mode, rows, threads = module._native_policy[str(x.size // x.shape[-1])]
    args = (x, module.weight, module.scales, module.biases)
    if mode == 'baseline':
        return row_reuse_int4(*args, rows, threads)
    if mode == 'j8':
        return unroll_int4(*args, 'j8', rows, threads)
    return scalar_int4(*args, rows, threads)


class NativeTunedLinear(CompactInt4Linear):
    def __call__(self, x):
        if not eligible(self, x):
            return super().__call__(x)
        y = project(self, x)
        return y + self.bias if 'bias' in self else y


class NativeTunedEmbedding(CompactInt4Embedding):
    def as_linear(self, x):
        if not eligible(self, x):
            return super().as_linear(x)
        return project(self, x)


def install_native_tuned_int4(model, policy):
    prepared = []
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedEmbedding):
            family = 'vocabulary'
            expected = CompactInt4Embedding
        elif name.endswith('.mlp.gate_proj') or name.endswith('.mlp.down_proj'):
            family = 'gate' if name.endswith('gate_proj') else 'down'
            expected = CompactInt4Linear
        else:
            continue
        if type(module) is not expected or (module.bits, module.group_size, module.mode) != (4, 64, 'affine'):
            raise ValueError('Native policy requires original compact int4 modules')
        choices = policy[family]
        for batch in ('2', '3', '4'):
            mode, rows, threads = choices[batch]
            if mode not in ('baseline', 'j8', 'scalar') or rows not in (1, 2, 4, 8) or threads not in (128, 256):
                raise ValueError('Invalid explicit native projection policy')
        prepared.append((module, choices, expected is CompactInt4Embedding))
    if not prepared:
        raise ValueError('No measured compact projection families found')
    for module, choices, embedding in prepared:
        module.__class__ = NativeTunedEmbedding if embedding else NativeTunedLinear
        module._native_policy = choices
    return len(prepared)
