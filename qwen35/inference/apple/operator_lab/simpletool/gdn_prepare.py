"""Decode-only convolution/SiLU/QK RMS/history preparation in one dispatch.

Convolution order follows the existing owned gdn_conv kernel; RMS follows the
four-contiguous-elements/lane MLX tree observed in the pinned local source.
Apple MLX arithmetic provenance: third_party/MLX_LICENSE, MLX_LM_LICENSE.
No precision reduction, borrowed buffer writes or int4 parameter conversion.
Runtime bitwise and complete greedy gates are required before use.
"""
import math
import mlx.core as mx
from mlx_lm.models.qwen3_5 import GatedDeltaNet
from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel, normalize_qk

from .gdn_conv import conv_state
from .gdn_gates import gate_pair
from .shared_gdn_seed import SharedSeedGDN, fork_update

SOURCE = r'''
    const uint lane=thread_index_in_simdgroup;
    const uint group=thread_position_in_grid.x/32;
    const uint heads=2*Hk+Hv;
    if(group>=B*heads) return;
    const uint batch=group/heads,head=group%heads;
    float values[4];
    for(uint i=0;i<4;++i) {
        const uint c=head*128+lane*4+i;
        float acc=0.0f;
        for(uint j=0;j<3;++j) acc+=history[(batch*3+j)*D+c]*weight[c*4+j];
        acc+=qkv[batch*D+c]*weight[c*4+3];
        const float t=1.0f/(1.0f+metal::precise::exp(metal::abs(acc)));
        const float sig=acc<0.0f?t:1.0f-t;
        values[i]=acc*sig;
        next_history[(batch*3+0)*D+c]=history[(batch*3+1)*D+c];
        next_history[(batch*3+1)*D+c]=history[(batch*3+2)*D+c];
        next_history[(batch*3+2)*D+c]=qkv[batch*D+c];
    }
    if(head<2*Hk) {
        float sum=0.0f;
        for(uint i=0;i<4;++i) sum+=values[i]*values[i];
        sum=simd_sum(sum);
        const float inv=metal::precise::rsqrt(sum/128.0f+(1.0e-6f/128.0f));
        for(uint i=0;i<4;++i) {
            const uint index=(batch*Hk+head%Hk)*128+lane*4+i;
            const float normalized=values[i]*inv;
            if(head<Hk) q[index]=normalized*(1.0f/128.0f);
            else k[index]=normalized*0.08838834764831845f;
        }
    } else {
        for(uint i=0;i<4;++i)
            v[(batch*Hv+head-2*Hk)*128+lane*4+i]=values[i];
    }
'''
_PREPARE = mx.fast.metal_kernel(name='simpletool_gdn_conv_norm_history',
    input_names=['history', 'qkv', 'weight'], output_names=['q', 'k', 'v', 'next_history'], source=SOURCE)


def prepare(history, qkv, weight, hk, hv):
    batch, sequence, d = qkv.shape
    if (sequence != 1 or not 1 <= batch <= 4 or hk < 1 or hv % hk
            or d != (2*hk+hv)*128 or history.shape != (batch, 3, d) or weight.shape != (d, 4, 1)):
        raise ValueError('GDN prepare requires B1..4, S1, Dk/Dv128, kernel4')
    if any(x.dtype != mx.float32 for x in (history, qkv, weight)):
        raise ValueError('GDN prepare uses FP32 exclusively')
    return _PREPARE(inputs=[history, qkv, weight], template=[('B', batch), ('D', d), ('Hk', hk), ('Hv', hv)],
                    grid=(math.ceil(batch*(2*hk+hv)/4)*128, 1, 1), threadgroup=(128, 1, 1),
                    output_shapes=[(batch, 1, hk, 128), (batch, 1, hk, 128), (batch, 1, hv, 128), history.shape],
                    output_dtypes=[mx.float32]*4)


class PrepareGDN(SharedSeedGDN):
    def __call__(self, inputs, mask=None, cache=None):
        B, S, _ = inputs.shape
        if (S != 1 or inputs.dtype != mx.float32 or self.conv_kernel_size != 4
                or self.head_k_dim != 128 or self.head_v_dim != 128 or mask is not None
                or cache is None or cache[0] is None or cache[1] is None or cache.lengths is not None
                or self.training or self.sharding_group is not None):
            return super().__call__(inputs, mask, cache)
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b, a = self.in_proj_b(inputs), self.in_proj_a(inputs)
        if self._prepare_mode == 'conv':
            conv, history = conv_state(cache[0], qkv, self.conv1d.weight)
            q, k, v = [t.reshape(B, S, h, 128) for t, h in zip(
                mx.split(conv, [self.key_dim, 2*self.key_dim], axis=-1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads])]
            q, k = normalize_qk(q, k, inv_scale=128**-.5, eps=1e-6)
        else:
            q, k, v, history = prepare(cache[0], qkv, self.conv1d.weight, self.num_k_heads, self.num_v_heads)
        cache[0] = history
        if self._prepare_mode == 'gates':
            g, beta = gate_pair(a, b, self.A_log, self.dt_bias)
        else:
            g, beta = compute_g(self.A_log, a, self.dt_bias), mx.sigmoid(b)
        if B > 1 and cache[1].shape[0] == 1:
            out, state = fork_update(q, k, v, g, beta, cache[1])
        else:
            out, state = gated_delta_kernel(q, k, v, g, beta, cache[1])
        cache[1] = state
        cache.advance(S)
        cache.share_gdn_seed = True
        return self.out_proj(self.norm(out, z).reshape(B, S, -1))


def install_prepare_gdn(model, mode='prepare'):
    if mode not in ('conv', 'prepare', 'gates'):
        raise ValueError('Explicit prepare mode required')
    modules = [layer.linear_attn for layer in model.layers if layer.is_linear]
    if any(type(m) not in (GatedDeltaNet, SharedSeedGDN) for m in modules):
        raise ValueError('Requires original or shared-seed GDN')
    for module in modules:
        module.__class__ = PrepareGDN
        module._prepare_mode = mode
