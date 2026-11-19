import unittest
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import Model,ModelArgs
from simpletool.block_compile import compiled_segments,install_block_compilation
from simpletool.cache import SharedPrefixKV,cache_arrays
from simpletool.decode import batched_generate,prefill,serial_generate,HEADS
from test_core import Tokenizer


class CompiledBlocks(unittest.TestCase):
    def test_attention_growing_tail_strides_and_batch_shrink(self):
        mx.random.seed(41)
        for B in (4,3,2,1):
            for P in (18,137,693):
                prefix=KVCache()
                k=mx.random.normal((1,2,P*2,128))[:,:,::2,:]
                v=mx.random.normal((1,2,P*2,128))[:,:,::2,:]
                prefix.update_and_fetch(k,v)
                cache=SharedPrefixKV(prefix,B)
                q=mx.random.normal((B,8,1,128))
                for T in (1,3,17,257):
                    tk=mx.random.normal((B,2,T*2,128))[:,:,::2,:]
                    tv=mx.random.normal((B,2,T*2,128))[:,:,::2,:]
                    actual=compiled_segments(q,cache,tk,tv,128**-.5)
                    ref=mx.fast.scaled_dot_product_attention(q,
                        mx.concatenate([mx.repeat(k,B,0),tk],axis=2),
                        mx.concatenate([mx.repeat(v,B,0),tv],axis=2),scale=128**-.5)
                    self.assertTrue(mx.allclose(actual,ref,atol=2e-5,rtol=2e-5).item(),(B,P,T))

    def test_hybrid_model_cache_isolation_and_tokens(self):
        mx.random.seed(43)
        config={'model_type':'qwen3_5','text_config':{
            'model_type':'qwen3_5_text','hidden_size':64,'intermediate_size':128,
            'num_hidden_layers':4,'num_attention_heads':4,'num_key_value_heads':2,
            'head_dim':32,'vocab_size':128,'tie_word_embeddings':True,
            'linear_num_key_heads':2,'linear_num_value_heads':2,
            'linear_key_head_dim':128,'linear_value_head_dim':128,
            'rope_parameters':{'rope_theta':10000000,'partial_rotary_factor':.25}}}
        model=Model(ModelArgs.from_dict(config));model.eval();mx.eval(model.parameters())
        ids=[11,21,31,41,51];source,_=prefill(model,ids)
        before=[mx.array(t) for t in cache_arrays(source)]
        reference=serial_generate(model,Tokenizer(),ids,8,prefix_cache=source)
        install_block_compilation(model)
        actual=batched_generate(model,Tokenizer(),ids,8,prefix_cache=source)
        for h in HEADS:self.assertEqual(actual['channels'][h]['token_ids'],reference['channels'][h]['token_ids'])
        for a,b in zip(before,cache_arrays(source)):self.assertTrue(mx.array_equal(a,b).item())


if __name__=='__main__':unittest.main()
