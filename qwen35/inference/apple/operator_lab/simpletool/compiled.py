"""Fixed-capacity functional cache state for MLX graph compilation."""
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from .attention import install_fork_attention
from .cache import SharedPrefixKV


class StaticPrefixKV(SharedPrefixKV):
    def __init__(self, state, used):
        self.prefix_k, self.prefix_v, self.tail_k, self.tail_v = state
        self.prefix_length = self.prefix_k.shape[2]
        self.used = used
        self.offset = used + self.prefix_length

    def append(self, k, v):
        self.tail_k = mx.slice_update(self.tail_k,k,self.used.reshape(1),axes=(2,))
        self.tail_v = mx.slice_update(self.tail_v,v,self.used.reshape(1),axes=(2,))
        self.used = self.used + 1
        self.offset = self.offset + 1
        return self.tail_k,self.tail_v


class CompiledStep:
    def __init__(self, model, suppress_tokens=(), fused_head=False, release_seed_cache=True):
        install_fork_attention(model)
        self.model=model
        self.suppress_tokens=tuple(suppress_tokens)
        self.fused_head=fused_head
        self.release_seed_cache=release_seed_cache
        if fused_head:
            from .greedy_head import greedy_int4
            from .row_reuse_int4 import RowReuseEmbedding
            if not model.language_model.args.tie_word_embeddings or not isinstance(model.model.embed_tokens,RowReuseEmbedding):
                raise ValueError('Fused head requires a tied, row-reuse int4 embedding')

        def forward(tokens, states, used):
            caches=[]
            for layer,state in zip(model.layers,states):
                if layer.is_linear:
                    c=ArraysCache(2)
                    c.cache=list(state)
                else:c=StaticPrefixKV(state,used)
                caches.append(c)
            if self.fused_head:
                hidden=model.model(tokens,cache=caches)
                prediction=greedy_int4(model.model.embed_tokens,hidden[:,-1,:],self.suppress_tokens)
            else:logits=model(tokens,cache=caches)
            next_states=[]
            for c in caches:
                if isinstance(c,StaticPrefixKV):
                    next_states.append((c.prefix_k,c.prefix_v,c.tail_k,c.tail_v))
                else:next_states.append(tuple(c.cache))
            if self.suppress_tokens and not self.fused_head:
                logits[...,list(self.suppress_tokens)]=-mx.inf
            return prediction if self.fused_head else mx.argmax(logits[:,-1,:],axis=-1),next_states

        self.forward=mx.compile(forward)
        self._raw_forward=forward
        self._blocks={}

    def advance_pending(self,prediction,states,used,count):
        """Generate count tokens with one host evaluation, keeping last pending.

        Intermediate greedy tokens stay on GPU. This is ordinary autoregressive
        decoding, with no draft model. Callers discard tokens after a channel
        stop and compact its state only at the block boundary.
        """
        if not 1<=count<=8:raise ValueError('decode block must contain 1..8 tokens')
        if count not in self._blocks:
            def block(prediction,states,used):
                predictions=[prediction]
                for i in range(count-1):
                    prediction,states=self._raw_forward(prediction[:,None],states,used+i)
                    predictions.append(prediction)
                return mx.stack(predictions,axis=1),states
            self._blocks[count]=mx.compile(block)
        return self._blocks[count](prediction,states,used)

    def initialize(self, caches, batch_size, capacity):
        states=[]
        for c in caches:
            if isinstance(c,SharedPrefixKV):
                pk,pv=c.prefix_k,c.prefix_v
                shape=(batch_size,pk.shape[1],capacity,pk.shape[-1])
                states.append((pk,pv,mx.zeros(shape,pk.dtype),mx.zeros(shape,pv.dtype)))
            else:states.append(tuple(c.cache))
        mx.eval(states)
        if self.release_seed_cache:
            # Functional states now own these arrays. Keeping the initial
            # fork in cache.cache retained B copies of the recurrent matrix
            # for the entire decode, although no subsequent step reads them.
            # Prefix KV stays shared and intact; parent caches are separate.
            # batched_generate commits the final states back at completion.
            for c in caches:
                if not isinstance(c,SharedPrefixKV):c.cache=[None]*len(c.cache)
        return states

    @staticmethod
    def filter(states,indices):
        idx=mx.array(indices,dtype=mx.int32)
        return [tuple(mx.take(t,idx,axis=0) for t in state) if len(state)==2
                else (state[0],state[1],mx.take(state[2],idx,axis=0),mx.take(state[3],idx,axis=0))
                for state in states]
