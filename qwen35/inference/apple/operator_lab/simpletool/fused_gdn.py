"""Projection fusion around MLX-LM's Qwen3.5 GDN.

Recurrence follows mlx-lm/models/qwen3_5.py, Copyright 2026 Apple Inc., MIT.
Only projection grouping changes; normalization, convolution and update are reused.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.gated_delta import gated_delta_update,normalize_qk
from mlx_lm.models.qwen3_5 import GatedDeltaNet


def concatenate_linears(a,b):
    if type(a) is not type(b):raise TypeError('mixed projection implementations')
    if 'bias' in a or 'bias' in b:raise ValueError('bias projections unsupported')
    if isinstance(a,nn.QuantizedLinear):
        if (a.bits,a.group_size,a.mode)!=(b.bits,b.group_size,b.mode):raise ValueError('quantization mismatch')
        out=nn.QuantizedLinear(a.weight.shape[1]*32//a.bits,a.weight.shape[0]+b.weight.shape[0],
                               bias=False,group_size=a.group_size,bits=a.bits,mode=a.mode)
        out.weight=mx.concatenate([a.weight,b.weight],axis=0)
        out.scales=mx.concatenate([a.scales,b.scales],axis=0)
        out.biases=mx.concatenate([a.biases,b.biases],axis=0)
    else:
        out=nn.Linear(a.weight.shape[1],a.weight.shape[0]+b.weight.shape[0],bias=False)
        out.weight=mx.concatenate([a.weight,b.weight],axis=0)
    mx.eval(out.parameters())
    return out


class FusedGDN(GatedDeltaNet):
    def __call__(self,inputs,mask=None,cache=None):
        if self.sharding_group is not None:raise ValueError('fused GDN is single-device only')
        bsz,length,_=inputs.shape
        if hasattr(self,'split_proj'):
            qkv,z,b,a=self.split_proj(inputs)
        elif hasattr(self,'all_proj'):
            qkv,z,b,a=mx.split(self.all_proj(inputs),
                [self.conv_dim,self.conv_dim+self.value_dim,self.conv_dim+self.value_dim+self.num_v_heads],axis=-1)
        else:
            qkv,z=mx.split(self.qkvz_proj(inputs),[self.conv_dim],axis=-1)
            b,a=mx.split(self.ba_proj(inputs),2,axis=-1)
        z=z.reshape(bsz,length,self.num_v_heads,self.head_v_dim)
        conv_state=cache[0] if cache is not None and cache[0] is not None else mx.zeros(
            (bsz,self.conv_kernel_size-1,self.conv_dim),dtype=inputs.dtype)
        if mask is not None:qkv=mx.where(mask[...,None],qkv,0)
        conv_input=mx.concatenate([conv_state,qkv],axis=1)
        if cache is not None:
            keep=self.conv_kernel_size-1
            if cache.lengths is not None:
                ends=mx.clip(cache.lengths,0,length)
                positions=(ends[:,None]+mx.arange(keep))[...,None]
                cache[0]=mx.take_along_axis(conv_input,positions,axis=1)
            else:cache[0]=mx.contiguous(conv_input[:,-keep:,:])
        conv_out=nn.silu(self.conv1d(conv_input))
        q,k,v=[t.reshape(bsz,length,h,d) for t,h,d in zip(
            mx.split(conv_out,[self.key_dim,2*self.key_dim],-1),
            [self.num_k_heads,self.num_k_heads,self.num_v_heads],
            [self.head_k_dim,self.head_k_dim,self.head_v_dim])]
        state=cache[1] if cache else None
        q,k=normalize_qk(q,k,inv_scale=self.head_k_dim**-0.5,eps=1e-6)
        out,state=gated_delta_update(q,k,v,a,b,self.A_log,self.dt_bias,state,mask,use_kernel=not self.training)
        if cache is not None:cache[1]=state;cache.advance(length)
        return self.out_proj(self.norm(out,z).reshape(bsz,length,-1))


def fuse_gdn_projections(model):
    count=0
    for layer in model.layers:
        if layer.is_linear and not isinstance(layer.linear_attn,FusedGDN):
            gdn=layer.linear_attn
            gdn.qkvz_proj=concatenate_linears(gdn.in_proj_qkv,gdn.in_proj_z)
            gdn.ba_proj=concatenate_linears(gdn.in_proj_b,gdn.in_proj_a)
            del gdn.in_proj_qkv,gdn.in_proj_z,gdn.in_proj_b,gdn.in_proj_a
            gdn.__class__=FusedGDN
            count+=1
    return count
