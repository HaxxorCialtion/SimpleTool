import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache


class SharedPrefixKV:
    """One immutable prefix, one batched mutable suffix. Decode only."""

    def __init__(self, source, batch_size):
        self.prefix_k, self.prefix_v = source.keys_and_values()
        if self.prefix_k.shape[0] != 1:
            raise ValueError("fork requires a single prefix sequence")
        self.offset = source.offset
        self.prefix_length = source.offset
        self.batch_size = batch_size
        self.tail = KVCache()

    def append(self, k, v):
        if k.shape[0] != self.batch_size or k.shape[2] != 1:
            raise ValueError("shared prefix attention accepts batched one-token decode")
        tail = self.tail.update_and_fetch(k, v)
        self.offset += 1
        return tail

    def make_mask(self, n, **kwargs):
        if n != 1:
            raise ValueError("prefill must use ordinary cache, before fork")
        return None

    def filter(self, indices):
        self.tail.keys = mx.take(self.tail.keys, indices, axis=0)
        self.tail.values = mx.take(self.tail.values, indices, axis=0)
        self.batch_size = len(indices)


def fork_batch(caches, batch_size):
    result = []
    for c in caches:
        if isinstance(c, KVCache):
            result.append(SharedPrefixKV(c, batch_size))
        elif isinstance(c, ArraysCache):
            new = ArraysCache(len(c.cache))
            # The recurrent matrix changes independently in each branch.
            shared_seed=getattr(c,'share_gdn_seed',False)
            new.cache = [(t if shared_seed and i==1 else mx.repeat(t,batch_size,axis=0)) if t is not None else None
                         for i,t in enumerate(c.cache)]
            new.share_gdn_seed=shared_seed
            if new.cache[1] is not None and new.cache[1].dtype != mx.float32:
                new.cache[1] = new.cache[1].astype(mx.float32)
            result.append(new)
        else:
            raise TypeError(type(c))
    return result


def clone_single(caches):
    result = []
    for c in caches:
        if isinstance(c, KVCache):
            n = KVCache()
            n.keys, n.values, n.offset = mx.array(c.keys), mx.array(c.values), c.offset
        elif isinstance(c, ArraysCache):
            n = ArraysCache(len(c.cache))
            n.cache = list(c.cache)
        else:
            raise TypeError(type(c))
        result.append(n)
    return result


def cache_arrays(caches):
    tensors = []
    for c in caches:
        if isinstance(c, SharedPrefixKV):
            tensors += [c.prefix_k, c.prefix_v, c.tail.keys, c.tail.values]
        elif isinstance(c, KVCache):
            tensors += [c.keys, c.values]
        else:
            tensors += c.cache
    return [t for t in tensors if t is not None]


def filter_batch(caches, indices):
    indices = mx.array(indices, dtype=mx.int32)
    for c in caches:
        c.filter(indices)


def memory_accounting(caches):
    prefix = suffix = recurrent = 0
    for c in caches:
        if isinstance(c, SharedPrefixKV):
            prefix += c.prefix_k.nbytes + c.prefix_v.nbytes
            suffix += c.tail.nbytes
        elif isinstance(c, ArraysCache):
            recurrent += sum(t.nbytes for t in c.cache if t is not None)
    return {"prefix_kv_bytes_once": prefix, "suffix_kv_allocated_bytes": suffix,
            "batched_gdn_bytes": recurrent}
