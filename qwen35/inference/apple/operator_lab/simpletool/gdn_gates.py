"""One Metal dispatch for GDN decay and beta; recurrence remains unchanged.

FP32 scalar formulas follow Apple MLX Exp/Sigmoid/LogAddExp/log1p (MIT).
Copyright © 2023 Apple Inc. License at third_party/MLX_LICENSE.
"""
import math
import mlx.core as mx
from mlx_lm.models.qwen3_5 import GatedDeltaNet
from mlx_lm.models.gated_delta import normalize_qk,gated_delta_kernel
import mlx.nn as nn

_GATES=mx.fast.metal_kernel(
    name='simpletool_gdn_gate_pair',input_names=['a','b','A_log','dt_bias'],output_names=['g','beta'],
    header=r'''
        inline float simpletool_log1p(float x) {
            float xp1=1.0f+x;
            if (xp1==metal::numeric_limits<float>::max()) return xp1;
            if (xp1==1.0f) return x;
            return x*(metal::log(xp1)/(xp1-1.0f));
        }
    ''',
    source=r'''
        const uint i=thread_position_in_grid.x;
        if (i>=N) return;
        const uint h=i%Hv;
        const float v=a[i]+dt_bias[h];
        const float hi=metal::max(v,0.0f),lo=metal::min(v,0.0f);
        const float soft=(metal::isinf(v)?hi:
            hi+simpletool_log1p(metal::exp(lo-hi)));
        g[i]=metal::precise::exp(-metal::precise::exp(A_log[h])*soft);
        const float t=1.0f/(1.0f+metal::precise::exp(metal::abs(b[i])));
        beta[i]=b[i]<0.0f?t:1.0f-t;
    ''')


def gate_pair(a,b,A_log,dt_bias):
    if any(x.dtype!=mx.float32 for x in (a,b,A_log,dt_bias)):
        raise ValueError('Gate fusion requires FP32')
    return _GATES(inputs=[a,b,A_log,dt_bias],template=[('N',a.size),('Hv',a.shape[-1])],
        grid=(math.ceil(a.size/128)*128,1,1),threadgroup=(128,1,1),
        output_shapes=[a.shape,b.shape],output_dtypes=[mx.float32,mx.float32])


class GateFusedGDN(GatedDeltaNet):
    def __call__(self,inputs,mask=None,cache=None):
        if (inputs.shape[1]!=1 or inputs.dtype!=mx.float32 or mask is not None or cache is None
            or cache[0] is None or cache[1] is None or cache.lengths is not None or self.training
            or self.sharding_group is not None):
            return super().__call__(inputs,mask,cache)
        B,S,_=inputs.shape
        qkv=self.in_proj_qkv(inputs)
        z=self.in_proj_z(inputs).reshape(B,S,self.num_v_heads,self.head_v_dim)
        b=self.in_proj_b(inputs);a=self.in_proj_a(inputs)
        conv_input=mx.concatenate([cache[0],qkv],axis=1)
        cache[0]=mx.contiguous(conv_input[:,-(self.conv_kernel_size-1):,:])
        conv=nn.silu(self.conv1d(conv_input))
        q,k,v=[t.reshape(B,S,h,d) for t,h,d in zip(
            mx.split(conv,[self.key_dim,2*self.key_dim],axis=-1),
            [self.num_k_heads,self.num_k_heads,self.num_v_heads],
            [self.head_k_dim,self.head_k_dim,self.head_v_dim])]
        q,k=normalize_qk(q,k,inv_scale=self.head_k_dim**-.5,eps=1e-6)
        g,beta=gate_pair(a,b,self.A_log,self.dt_bias)
        out,state=gated_delta_kernel(q,k,v,g,beta,cache[1])
        cache[1]=state;cache.advance(S)
        return self.out_proj(self.norm(out,z).reshape(B,S,-1))


def install_gate_fusion(model):
    for layer in model.layers:
        if layer.is_linear:
            if type(layer.linear_attn) is not GatedDeltaNet:
                raise ValueError('Gate fusion requires original unfused GDN modules')
            layer.linear_attn.__class__=GateFusedGDN
