"""Merge independent row projections without requantization or precision changes."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_next import Qwen3NextMLP, swiglu
from .fused_gdn import FusedGDN
from .kernels import install_int4_kernel


def merge(*modules,aliases=None):
    a=modules[0]
    if not all(isinstance(m,nn.QuantizedLinear) and 'bias' not in m
               and (m.bits,m.group_size,m.mode)==(a.bits,a.group_size,a.mode)
               and m.weight.shape[1]==a.weight.shape[1] for m in modules):
        raise ValueError('fusion requires matching bias-free quantized projections')
    n=sum(m.weight.shape[0] for m in modules);k=a.weight.shape[1]*32//a.bits
    result=nn.QuantizedLinear(k,n,bias=False,group_size=a.group_size,bits=a.bits,mode=a.mode)
    for key in ('weight','scales','biases'):
        setattr(result,key,mx.concatenate([getattr(m,key) for m in modules],axis=0))
    mx.eval(result.parameters())
    if aliases is not None:
        offset=0
        for m in aliases:
            rows=m.weight.shape[0]
            for key in ('weight','scales','biases'):
                setattr(m,key,getattr(result,key)[offset:offset+rows])
            offset+=rows
    return result


class FusedMLP(Qwen3NextMLP):
    def __call__(self,x):
        if hasattr(self,'split_proj'):gate,up=self.split_proj(x)
        else:gate,up=mx.split(self.gate_up_proj(x),2,axis=-1)
        return self.down_proj(swiglu(gate,up))


def fuse_projections(model,alias_base=None,split_output=False):
    from .split_projection import SplitProjection
    for index,layer in enumerate(model.layers):
        base=alias_base.layers[index] if alias_base is not None else None
        if layer.is_linear:
            g=layer.linear_attn
            if not isinstance(g,FusedGDN):
                aliases=[getattr(base.linear_attn,k) for k in ('in_proj_qkv','in_proj_z','in_proj_b','in_proj_a')] if base else None
                merged=merge(g.in_proj_qkv,g.in_proj_z,g.in_proj_b,g.in_proj_a,aliases=aliases)
                if split_output:g.split_proj=SplitProjection(merged,[g.conv_dim,g.value_dim,g.num_v_heads,g.num_v_heads])
                else:g.all_proj=merged
                del g.in_proj_qkv,g.in_proj_z,g.in_proj_b,g.in_proj_a
                g.__class__=FusedGDN
        m=layer.mlp
        if not isinstance(m,FusedMLP):
            aliases=[base.mlp.gate_proj,base.mlp.up_proj] if base else None
            merged=merge(m.gate_proj,m.up_proj,aliases=aliases)
            if split_output:m.split_proj=SplitProjection(merged,[m.gate_proj.weight.shape[0],m.up_proj.weight.shape[0]])
            else:m.gate_up_proj=merged
            del m.gate_proj,m.up_proj
            m.__class__=FusedMLP
    install_int4_kernel(model)
