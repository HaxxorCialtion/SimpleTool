import unittest
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import normalize_qk, gated_delta_kernel
from mlx_lm.models.qwen3_5 import Model, ModelArgs
from simpletool.cache import fork_batch, cache_arrays
from simpletool.decode import HEADS, prefill, serial_generate, batched_generate
from simpletool.compiled import CompiledStep
from simpletool.shared_gdn_seed import fork_update, install_shared_gdn_seed
from tests.test_core import Tokenizer


class SharedGDNSeed(unittest.TestCase):
    def test_exact_recurrence_and_independent_branches(self):
        mx.random.seed(149)
        for hk,hv in ((16,32),(16,16),(8,16)):
            prefix=mx.random.normal((1,hv,128,128))*.01
            before=mx.array(prefix)
            for batch in (1,2,3,4):
                q,k=normalize_qk(mx.random.normal((batch,1,hk,128)),
                    mx.random.normal((batch,1,hk,128)),inv_scale=128**-.5,eps=1e-6)
                v=mx.random.normal((batch,1,hv,128))*.1
                g=mx.exp(-mx.random.uniform(.01,1.,(batch,1,hv)))
                beta=mx.sigmoid(mx.random.normal((batch,1,hv)))
                y,state=fork_update(q,k,v,g,beta,prefix)
                ry,rs=gated_delta_kernel(q,k,v,g,beta,mx.repeat(prefix,batch,axis=0))
                self.assertTrue(mx.array_equal(y,ry).item(),(hk,hv,batch,'y'))
                self.assertTrue(mx.array_equal(state,rs).item(),(hk,hv,batch,'state'))
                self.assertTrue(mx.array_equal(prefix,before).item())
                if batch>1:self.assertFalse(mx.array_equal(state[0],state[1]).item())

    def test_model_fork_ownership_and_compiled_tokens(self):
        mx.random.seed(151)
        config={'model_type':'qwen3_5','text_config':{
            'model_type':'qwen3_5_text','hidden_size':64,'intermediate_size':128,
            'num_hidden_layers':4,'num_attention_heads':4,'num_key_value_heads':2,
            'head_dim':32,'vocab_size':128,'tie_word_embeddings':True,
            'linear_num_key_heads':2,'linear_num_value_heads':2,
            'linear_key_head_dim':128,'linear_value_head_dim':128,
            'rope_parameters':{'rope_theta':10000000,'partial_rotary_factor':.25}}}
        model=Model(ModelArgs.from_dict(config));model.eval();mx.eval(model.parameters())
        ids=[11,21,31,41,51]
        gold_cache,_=prefill(model,ids)
        gold=serial_generate(model,Tokenizer(),ids,8,prefix_cache=gold_cache)
        install_shared_gdn_seed(model)
        cache,_=prefill(model,ids)
        snapshots=[mx.array(t) for t in cache_arrays(cache)]
        for source,fork in zip(cache,fork_batch(cache,4)):
            if isinstance(source,ArraysCache):
                self.assertIs(fork[1],source[1])
                self.assertEqual(fork[0].shape[0],4)
        for compiled in (None,CompiledStep(model)):
            result=batched_generate(model,Tokenizer(),ids,8,prefix_cache=cache,compiled_step=compiled)
            for h in HEADS:self.assertEqual(result['channels'][h]['token_ids'],gold['channels'][h]['token_ids'])
        for before,after in zip(snapshots,cache_arrays(cache)):
            self.assertTrue(mx.array_equal(before,after).item())


if __name__=='__main__':unittest.main()
