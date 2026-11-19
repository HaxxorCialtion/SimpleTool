import json
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import Model, ModelArgs

from simpletool.attention import segmented_attention
from simpletool.cache import SharedPrefixKV, cache_arrays, fork_batch
from simpletool.decode import HEADS, batched_generate, prefill, serial_generate
from simpletool.kernels import int4_matmul
from simpletool.compiled import CompiledStep


class Tokenizer:
    def encode(self, text, **kwargs):
        # Unequal semantic markers, equal sequence length, as in real tokenizer.
        h = next(h for h in HEADS if h in text)
        return [2, HEADS.index(h) + 3, 8 if "/" not in text else 9]

    def convert_tokens_to_ids(self, text):
        return 127

    def decode(self, ids, **kwargs):
        return str(ids)


class Core(unittest.TestCase):
    def test_segmented_attention_and_prefix_isolation(self):
        mx.random.seed(3)
        prefix = KVCache()
        k = mx.random.normal((1, 2, 137, 32))
        v = mx.random.normal((1, 2, 137, 32))
        prefix.update_and_fetch(k, v)
        mx.eval(prefix.state)
        before = mx.array(prefix.keys)
        fork = SharedPrefixKV(prefix, 4)
        q = mx.random.normal((4, 8, 1, 32))
        tk = mx.random.normal((4, 2, 1, 32))
        tv = mx.random.normal((4, 2, 1, 32))
        tail = fork.append(tk, tv)
        out = segmented_attention(q, fork, *tail, 32**-0.5)
        ref = mx.fast.scaled_dot_product_attention(q,
            mx.concatenate([mx.repeat(k, 4, 0), tk], axis=2),
            mx.concatenate([mx.repeat(v, 4, 0), tv], axis=2), scale=32**-0.5)
        self.assertTrue(mx.allclose(out, ref, atol=2e-5).item())
        self.assertTrue(mx.array_equal(prefix.keys, before).item())
        self.assertEqual(fork.prefix_k.shape[0], 1)
        self.assertEqual(prefix.offset, 137)

    def test_int4_all_small_batches(self):
        mx.random.seed(5)
        w = mx.random.normal((79, 256)).astype(mx.float16)
        packed, scales, biases = mx.quantize(w, group_size=64, bits=4)
        errors = {}
        for m in range(1, 9):
            x = mx.random.normal((m, 256)).astype(mx.float16)
            actual = int4_matmul(x, packed, scales, biases)
            # float32 dequant reference does not share the custom kernel.
            ref = x.astype(mx.float32) @ mx.dequantize(packed, scales, biases,
                    group_size=64, bits=4).astype(mx.float32).T
            errors[m] = float(mx.max(mx.abs(actual.astype(mx.float32) - ref)).item())
            self.assertTrue(mx.allclose(actual.astype(mx.float32), ref, rtol=0.015, atol=0.03).item())
        print("int4_absolute_errors", json.dumps(errors))

    def test_prefix_kernel_strides_compaction_and_static_tail(self):
        from simpletool.prefix_sdpa import prefix_sdpa
        from simpletool.compiled import StaticPrefixKV
        mx.random.seed(19)
        for b in (1, 2, 3, 4):
            k = mx.random.normal((1, 4, 263, 256))[:, :, ::2, :]
            v = mx.random.normal((1, 4, 263, 256))[:, :, ::2, :]
            tk = mx.random.normal((b, 4, 96, 256))
            tv = mx.random.normal((b, 4, 96, 256))
            q = mx.random.normal((b, 16, 1, 256))
            cache = StaticPrefixKV((k, v, tk, tv), mx.array(17, mx.int32))
            ref = mx.fast.scaled_dot_product_attention(q,
                mx.concatenate([mx.repeat(k, b, 0), tk[:, :, :17]], axis=2),
                mx.concatenate([mx.repeat(v, b, 0), tv[:, :, :17]], axis=2), scale=256**-0.5)
            forward = mx.compile(lambda x, used: prefix_sdpa(x,
                StaticPrefixKV((k, v, tk, tv), used), tk, tv))
            actual = forward(q, mx.array(17, mx.int32))
            self.assertTrue(mx.allclose(actual, ref, atol=2e-5).item())

    def test_full_hybrid_model_batch_vs_serial(self):
        mx.random.seed(7)
        config = {"model_type": "qwen3_5", "text_config": {
            "model_type": "qwen3_5_text", "hidden_size": 64,
            "intermediate_size": 128, "num_hidden_layers": 4,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 32, "vocab_size": 128, "tie_word_embeddings": True,
            "linear_num_key_heads": 2, "linear_num_value_heads": 2,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25}}}
        model = Model(ModelArgs.from_dict(config))
        model.eval()
        mx.eval(model.parameters())
        ids = [11, 21, 31, 41, 51]
        base, _ = prefill(model, ids)
        snapshots = [mx.array(t) for t in cache_arrays(base)]
        serial = serial_generate(model, Tokenizer(), ids, max_tokens=8, prefix_cache=base)
        actual = batched_generate(model, Tokenizer(), ids, max_tokens=8, prefix_cache=base)
        for h in HEADS:
            self.assertEqual(actual["channels"][h]["token_ids"], serial["channels"][h]["token_ids"])
        compiled = batched_generate(model, Tokenizer(), ids, max_tokens=8,
                                   prefix_cache=base, compiled_step=CompiledStep(model))
        for h in HEADS:
            self.assertEqual(compiled["channels"][h]["token_ids"], serial["channels"][h]["token_ids"])
        for before, after in zip(snapshots, cache_arrays(base)):
            self.assertTrue(mx.array_equal(before, after).item())
        for c in fork_batch(base, 4):
            if hasattr(c, "cache"):
                self.assertEqual(c.cache[1].dtype, mx.float32)
        print("hybrid_serial_batch_equal", True)


if __name__ == "__main__":
    unittest.main()
