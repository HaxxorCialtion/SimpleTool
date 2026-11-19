"""Explicit full-model experimental decode; existing runtime defaults unchanged."""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import normalize_qk,compute_g
from .deferred_gdn import DeferredState
from .shared_gdn_seed import SharedSeedGDN,install_shared_gdn_seed
from .attention import install_fork_attention
from .cache import SharedPrefixKV
from .compiled import StaticPrefixKV,CompiledStep

class DeferredGDN(SharedSeedGDN):
    def __call__(self,inputs,mask=None,cache=None):
        if cache is None or not hasattr(cache,'deferred_state'):
            return super().__call__(inputs,mask,cache)
        B,S,_=inputs.shape
        if S!=1 or inputs.dtype!=mx.float32 or mask is not None or self.training or self.sharding_group is not None:
            raise ValueError('Deferred model is single-device FP32 one-token decode only')
        d=cache.deferred_state
        qkv=self.in_proj_qkv(inputs)
        z=self.in_proj_z(inputs).reshape(B,S,self.num_v_heads,self.head_v_dim)
        beta=self.in_proj_b(inputs);a=self.in_proj_a(inputs)
        full=mx.concatenate([cache[0],qkv],axis=1)
        cache[0]=mx.contiguous(full[:,-(self.conv_kernel_size-1):,:])
        cv=nn.silu(self.conv1d(full))
        q,k,v=[t.reshape(B,S,h,width) for t,h,width in zip(mx.split(cv,[self.key_dim,2*self.key_dim],axis=-1),
            [self.num_k_heads,self.num_k_heads,self.num_v_heads],[self.head_k_dim,self.head_k_dim,self.head_v_dim])]
        q,k=normalize_qk(q,k,inv_scale=self.head_k_dim**-.5,eps=1e-6)
        out=d.step(q,k,v,compute_g(self.A_log,a,self.dt_bias),mx.sigmoid(beta))
        cache[1]=d.base;cache.advance(S)
        return self.out_proj(self.norm(out,z).reshape(B,S,-1))

def prepare_deferred_model(model):
    install_shared_gdn_seed(model)
    for layer in model.layers:
        if layer.is_linear:layer.linear_attn.__class__=DeferredGDN

class CompiledDeferredStep:
    advance_pending=CompiledStep.advance_pending
    def __init__(self,model,interval=4):
        if interval not in (2,4,8,16):raise ValueError('Deferred interval must be 2/4/8/16')
        install_fork_attention(model)
        self.model=model;self.interval=interval;self.suppress_tokens=();self.fused_head=False;self.release_seed_cache=True;self._blocks={}
        def forward(tokens,states,used):
            caches=[]
            for layer,state in zip(model.layers,states):
                if layer.is_linear:
                    c=ArraysCache(2);c.cache=[state[0],state[1]]
                    d=DeferredState(state[1],tokens.shape[0],layer.linear_attn.num_k_heads,interval)
                    d.keys,d.gates,d.deltas=state[2:];c.deferred_state=d
                else:c=StaticPrefixKV(state,used)
                caches.append(c)
            logits=model(tokens,cache=caches)
            next_states=[]
            for c in caches:
                if isinstance(c,StaticPrefixKV):next_states.append((c.prefix_k,c.prefix_v,c.tail_k,c.tail_v))
                else:next_states.append((c[0],*c.deferred_state.arrays))
            return mx.argmax(logits[:,-1,:],axis=-1),next_states
        self._raw_forward=forward;self.forward=mx.compile(forward)
    def initialize(self,caches,batch_size,capacity):
        states=[]
        for layer,c in zip(self.model.layers,caches):
            if isinstance(c,SharedPrefixKV):
                pk,pv=c.prefix_k,c.prefix_v;shape=(batch_size,pk.shape[1],capacity,pk.shape[-1])
                states.append((pk,pv,mx.zeros(shape,pk.dtype),mx.zeros(shape,pv.dtype)))
            else:
                d=DeferredState(c[1],batch_size,layer.linear_attn.num_k_heads,self.interval)
                states.append((c[0],*d.arrays))
        mx.eval(states)
        for c in caches:
            if not isinstance(c,SharedPrefixKV):c.cache=[None]*len(c.cache)
        return states
    @staticmethod
    def filter(states,indices):
        idx=mx.array(indices,dtype=mx.int32);result=[]
        for state in states:
            if len(state)==4:result.append((state[0],state[1],mx.take(state[2],idx,axis=0),mx.take(state[3],idx,axis=0)))
            else:
                base=state[1] if state[1].shape[0]==1 else mx.take(state[1],idx,axis=0)
                result.append((mx.take(state[0],idx,axis=0),base,*[mx.take(t,idx,axis=0) for t in state[2:]]))
        return result
