"""Decode-only depthwise conv + SiLU + convolution-history update in one kernel.

Depthwise accumulation follows Apple MLX's sequential FP32 conv1d (MIT).
Copyright © 2023 Apple Inc. License at third_party/MLX_LICENSE.
The recurrent GDN matrix, normalization, and projection precision are unchanged.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import GatedDeltaNet
from mlx_lm.models.gated_delta import normalize_qk,gated_delta_update


_CONV=mx.fast.metal_kernel(
    name='simpletool_gdn_conv_state',input_names=['history','qkv','weight'],
    output_names=['out','next_history'],
    source=r'''
        const uint i=thread_position_in_grid.x;
        if (i>=B*D) return;
        const uint b=i/D,c=i%D;
        float acc=0.0f;
        for (uint j=0;j<3;++j) acc+=history[(b*3+j)*D+c]*weight[c*4+j];
        acc+=qkv[i]*weight[c*4+3];
        const float t=1.0f/(1.0f+metal::precise::exp(metal::abs(acc)));
        const float sig=acc<0.0f?t:1.0f-t;
        out[i]=acc*sig;
        next_history[(b*3+0)*D+c]=history[(b*3+1)*D+c];
        next_history[(b*3+1)*D+c]=history[(b*3+2)*D+c];
        next_history[(b*3+2)*D+c]=qkv[i];
    ''')


def conv_state(history,qkv,weight):
    B,S,D=qkv.shape
    if S!=1 or history.shape!=(B,3,D) or weight.shape!=(D,4,1):
        raise ValueError('Conv fusion requires decode S1, kernel4, matching history')
    if any(x.dtype!=mx.float32 for x in (history,qkv,weight)):
        raise ValueError('Conv fusion requires FP32 floating tensors')
    return _CONV(inputs=[history,qkv,weight],template=[('B',B),('D',D)],
        grid=(math.ceil(B*D/256)*256,1,1),threadgroup=(256,1,1),
        output_shapes=[qkv.shape,history.shape],output_dtypes=[mx.float32,mx.float32])


class ConvFusedGDN(GatedDeltaNet):
    def __call__(self,inputs,mask=None,cache=None):
        if (inputs.shape[1]!=1 or inputs.dtype!=mx.float32 or self.conv_kernel_size!=4
            or mask is not None or cache is None or cache[0] is None
            or cache.lengths is not None or self.training or self.sharding_group is not None):
            return super().__call__(inputs,mask,cache)
        B,S,_=inputs.shape
        qkv=self.in_proj_qkv(inputs)
        z=self.in_proj_z(inputs).reshape(B,S,self.num_v_heads,self.head_v_dim)
        b=self.in_proj_b(inputs);a=self.in_proj_a(inputs)
        conv,history=conv_state(cache[0],qkv,self.conv1d.weight)
        cache[0]=history
        q,k,v=[t.reshape(B,S,h,d) for t,h,d in zip(
            mx.split(conv,[self.key_dim,2*self.key_dim],axis=-1),
            [self.num_k_heads,self.num_k_heads,self.num_v_heads],
            [self.head_k_dim,self.head_k_dim,self.head_v_dim])]
        q,k=normalize_qk(q,k,inv_scale=self.head_k_dim**-.5,eps=1e-6)
        out,state=gated_delta_update(q,k,v,a,b,self.A_log,self.dt_bias,cache[1])
        cache[1]=state;cache.advance(S)
        return self.out_proj(self.norm(out,z).reshape(B,S,-1))


def install_conv_fusion(model):
    for layer in model.layers:
        if layer.is_linear:
            if type(layer.linear_attn) is not GatedDeltaNet:
                raise ValueError('Conv fusion requires original unfused GDN modules')
            layer.linear_attn.__class__=ConvFusedGDN
