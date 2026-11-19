"""Compile small decode blocks, reusing one graph across matching GDN layers."""
from functools import lru_cache
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import GatedDeltaNet
from mlx_lm.models.gated_delta import gated_delta_update,normalize_qk
from mlx_lm.models.activations import precise_swiglu


@lru_cache(None)
def post_projection(hk,hv,dk,dv,conv_size,eps):
    key_dim=hk*dk;conv_dim=2*key_dim+hv*dv
    @mx.compile
    def forward(qkv,z,b,a,conv_state,state,conv_weight,A_log,dt_bias,norm_weight):
        B,S,_=qkv.shape
        conv_input=mx.concatenate([conv_state,qkv],axis=1)
        next_conv=mx.contiguous(conv_input[:,-(conv_size-1):,:])
        conv_out=nn.silu(mx.conv1d(conv_input,conv_weight,groups=conv_dim))
        q,k,v=[t.reshape(B,S,h,d) for t,h,d in zip(
            mx.split(conv_out,[key_dim,2*key_dim],axis=-1),[hk,hk,hv],[dk,dk,dv])]
        q,k=normalize_qk(q,k,inv_scale=dk**-.5,eps=1e-6)
        out,next_state=gated_delta_update(q,k,v,a,b,A_log,dt_bias,state,use_kernel=True)
        normed=mx.fast.rms_norm(out,norm_weight,eps)
        out=precise_swiglu(out,z,normed)
        return out.reshape(B,S,-1),next_conv,next_state
    return forward


class BlockCompiledGDN(GatedDeltaNet):
    def __call__(self,inputs,mask=None,cache=None):
        if inputs.shape[1]!=1 or mask is not None or cache is None or cache[0] is None or cache[1] is None or self.training:
            return super().__call__(inputs,mask,cache)
        if self.sharding_group is not None:raise ValueError('block compile is single-device only')
        B,S,_=inputs.shape
        qkv=self.in_proj_qkv(inputs)
        z=self.in_proj_z(inputs).reshape(B,S,self.num_v_heads,self.head_v_dim)
        b=self.in_proj_b(inputs);a=self.in_proj_a(inputs)
        out,conv,state=post_projection(self.num_k_heads,self.num_v_heads,self.head_k_dim,
            self.head_v_dim,self.conv_kernel_size,self.layer_norm_epsilon)(
                qkv,z,b,a,cache[0],cache[1],self.conv1d.weight,self.A_log,self.dt_bias,self.norm.weight)
        cache[0]=conv;cache[1]=state;cache.advance(S)
        return self.out_proj(out)


@lru_cache(None)
def attention_core(B,Hq,Hk,D,P,scale,dtype):
    R=Hq//Hk
    def forward(q,pk,pv,tk,tv):
        q=q.astype(mx.float32)
        grouped=q.reshape(B,Hk,R,D).transpose(1,0,2,3).reshape(Hk,B*R,D)
        ps=(grouped*scale)@pk.astype(mx.float32).swapaxes(-1,-2)
        ps=ps.reshape(Hk,B,R,-1).transpose(1,0,2,3)
        ts=(q.reshape(B,Hk,R,D)*scale)@tk.astype(mx.float32).swapaxes(-1,-2)
        maximum=mx.maximum(mx.max(ps,axis=-1,keepdims=True),mx.max(ts,axis=-1,keepdims=True))
        ep=mx.exp(ps-maximum);et=mx.exp(ts-maximum)
        denom=mx.sum(ep,axis=-1,keepdims=True)+mx.sum(et,axis=-1,keepdims=True)
        pp=ep.transpose(1,0,2,3).reshape(Hk,B*R,P)
        po=(pp@pv.astype(mx.float32)).reshape(Hk,B,R,D).transpose(1,0,2,3)
        to=et@tv.astype(mx.float32)
        return ((po+to)/denom).reshape(B,Hq,1,D).astype(dtype)
    return mx.compile(forward,shapeless=True)


def compiled_segments(q,cache,tk,tv,scale):
    B,Hq,_,D=q.shape;Hk=cache.prefix_k.shape[1];P=cache.prefix_length
    return attention_core(B,Hq,Hk,D,P,scale,q.dtype)(q,cache.prefix_k[0],cache.prefix_v[0],tk,tv)


def install_block_compilation(model):
    for layer in model.layers:
        if layer.is_linear:layer.linear_attn.__class__=BlockCompiledGDN
        else:layer.self_attn._compile_segments=True
