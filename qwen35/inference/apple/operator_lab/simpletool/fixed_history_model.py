"""Two compiled write/defer paths with fixed history shapes. Explicit experiment."""
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache
from .fixed_history_gdn import FixedHistoryState
from .deferred_model import CompiledDeferredStep
from .attention import install_fork_attention
from .cache import SharedPrefixKV
from .compiled import StaticPrefixKV

class CompiledFixedHistoryStep:
    filter=staticmethod(CompiledDeferredStep.filter)
    def __init__(self,model,interval=16,fused=False,state_type=FixedHistoryState):
        if interval not in (2,4,8,16):raise ValueError('Fixed history interval must be 2/4/8/16')
        install_fork_attention(model)
        self.model=model;self.interval=interval;self.suppress_tokens=();self.fused_head=False;self.release_seed_cache=True
        self.fused=fused;self.state_type=state_type
        self._blocks={};self.input_signatures=set()
        def make(write):
            def forward(tokens,states,used):
                caches=[]
                for layer,state in zip(model.layers,states):
                    if layer.is_linear:
                        c=ArraysCache(2);c.cache=[state[0],state[1]]
                        d=state_type(state[1],tokens.shape[0],layer.linear_attn.num_k_heads,interval,used%interval,write,fused)
                        d.keys,d.gates,d.deltas=state[2:];c.deferred_state=d
                    else:c=StaticPrefixKV(state,used)
                    caches.append(c)
                logits=model(tokens,cache=caches)
                result=[]
                for c in caches:
                    if isinstance(c,StaticPrefixKV):result.append((c.prefix_k,c.prefix_v,c.tail_k,c.tail_v))
                    else:result.append((c[0],*c.deferred_state.arrays))
                return mx.argmax(logits[:,-1,:],axis=-1),result
            return forward
        self.raw_defer=make(False);self.raw_write=make(True)
        self.defer=mx.compile(self.raw_defer);self.write=mx.compile(self.raw_write)
    def initialize(self,caches,batch_size,capacity):
        states=[]
        for layer,c in zip(self.model.layers,caches):
            if isinstance(c,SharedPrefixKV):
                pk,pv=c.prefix_k,c.prefix_v;shape=(batch_size,pk.shape[1],capacity,pk.shape[-1])
                states.append((pk,pv,mx.zeros(shape,pk.dtype),mx.zeros(shape,pv.dtype)))
            else:
                d=self.state_type(c[1],batch_size,layer.linear_attn.num_k_heads,self.interval)
                states.append((c[0],*d.arrays))
        mx.eval(states)
        for c in caches:
            if not isinstance(c,SharedPrefixKV):c.cache=[None]*len(c.cache)
        return states
    def forward(self,tokens,states,used):
        # Caller owns this host-created scalar. No intermediate prediction is
        # read by the scheduler. Include this host work in complete benchmarks.
        step=int(used.item());write=step%self.interval==self.interval-1
        self.input_signatures.add((write,tuple(tokens.shape),tuple(tuple(tuple(a.shape) for a in s) for s in states)))
        return (self.write if write else self.defer)(tokens,states,used)
    def advance_pending(self,prediction,states,used,count):
        if not 1<=count<=8:raise ValueError('decode block must contain 1..8 tokens')
        remaining=count-1;step=int(used.item());pieces=[prediction[:,None]]
        while remaining:
            phase=step%self.interval
            if phase==self.interval-1:
                prediction,states=self.write(prediction[:,None],states,mx.array(step,mx.int32))
                n=1;part=prediction[:,None]
            else:
                n=min(remaining,self.interval-1-phase)
                if n not in self._blocks:
                    def block(prediction,states,used,n=n):
                        predictions=[]
                        for i in range(n):
                            prediction,states=self.raw_defer(prediction[:,None],states,used+i)
                            predictions.append(prediction)
                        return mx.stack(predictions,axis=1),states
                    self._blocks[n]=mx.compile(block)
                part,states=self._blocks[n](prediction,states,mx.array(step,mx.int32))
                prediction=part[:,-1]
            pieces.append(part);remaining-=n;step+=n
        return mx.concatenate(pieces,axis=1),states
