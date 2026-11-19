import mlx.core as mx
from mlx_lm.models.qwen3_next import Qwen3NextAttention

from .cache import SharedPrefixKV

# Experimental online-softmax kernel is numerically verified but slower here.
USE_FUSED_PREFIX = False

def segmented_attention(q, cache, tail_k, tail_v, scale):
    """All branch queries become rows against a single prefix K/V matrix."""
    b, hq, _, d = q.shape
    if USE_FUSED_PREFIX and cache.prefix_length >= 128 and b <= 4 and d % 32 == 0:
        from .prefix_sdpa import prefix_sdpa
        return prefix_sdpa(q,cache,tail_k,tail_v)
    output_dtype = q.dtype
    # Match fused SDPA's accumulation precision; do not round scores to fp16.
    q = q.astype(mx.float32)
    hk = cache.prefix_k.shape[1]
    r = hq // hk
    # [Hkv, B*GQA, D] is a matrix, not B copies of the prefix.
    q_grouped = q.reshape(b, hk, r, d).transpose(1, 0, 2, 3).reshape(hk, b * r, d)
    p_scores = (q_grouped * scale) @ cache.prefix_k[0].astype(mx.float32).swapaxes(-1, -2)
    p_scores = p_scores.reshape(hk, b, r, -1).transpose(1, 0, 2, 3)
    q_tail = q.reshape(b, hk, r, d)
    t_scores = (q_tail * scale) @ tail_k.astype(mx.float32).swapaxes(-1, -2)
    if hasattr(cache, "used"):
        t_scores = mx.where(mx.arange(tail_k.shape[2]) < cache.used, t_scores, -mx.inf)
    scores = mx.concatenate([p_scores, t_scores], axis=-1)
    probs = mx.softmax(scores, axis=-1, precise=True)
    pp = probs[..., :cache.prefix_length]
    pt = probs[..., cache.prefix_length:]
    pp = pp.transpose(1, 0, 2, 3).reshape(hk, b * r, cache.prefix_length)
    p_out = pp @ cache.prefix_v[0].astype(mx.float32)
    p_out = p_out.reshape(hk, b, r, d).transpose(1, 0, 2, 3)
    t_out = pt @ tail_v.astype(mx.float32)
    return (p_out + t_out).reshape(b, hq, 1, d).astype(output_dtype)


class ForkAttention(Qwen3NextAttention):
    def __call__(self, x, mask=None, cache=None):
        if not isinstance(cache, SharedPrefixKV):
            return super().__call__(x, mask=mask, cache=cache)
        b, l, _ = x.shape
        q, gate = mx.split(self.q_proj(x).reshape(b, l, self.num_attention_heads, -1),
                           2, axis=-1)
        gate = gate.reshape(b, l, -1)
        k = self.k_proj(x).reshape(b, l, self.num_key_value_heads, -1)
        v = self.v_proj(x).reshape(b, l, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        q = self.rope(self.q_norm(q).transpose(0, 2, 1, 3), offset=cache.offset)
        k = self.rope(self.k_norm(k).transpose(0, 2, 1, 3), offset=cache.offset)
        tk, tv = cache.append(k, v)
        if getattr(self,'_compile_segments',False) and not hasattr(cache,'used'):
            from .block_compile import compiled_segments
            out=compiled_segments(q,cache,tk,tv,self.scale)
        else:
            out = segmented_attention(q, cache, tk, tv, self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(b, l, -1)
        return self.o_proj(out * mx.sigmoid(gate))


def install_fork_attention(model):
    for layer in model.layers:
        if not layer.is_linear:
            layer.self_attn.__class__ = ForkAttention
