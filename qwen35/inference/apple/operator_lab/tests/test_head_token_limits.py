import unittest
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import Model,ModelArgs
from simpletool.cache import cache_arrays
from simpletool.compiled import CompiledStep
from simpletool.decode import HEADS,batched_generate,prefill
from test_core import Tokenizer

class HeadTokenLimits(unittest.TestCase):
    def test_independent_stop_matches_unbounded_prefix_and_preserves_seed(self):
        mx.random.seed(109)
        config={'model_type':'qwen3_5','text_config':{
            'model_type':'qwen3_5_text','hidden_size':64,'intermediate_size':128,
            'num_hidden_layers':4,'num_attention_heads':4,'num_key_value_heads':2,
            'head_dim':32,'vocab_size':128,'tie_word_embeddings':True,
            'linear_num_key_heads':2,'linear_num_value_heads':2,
            'linear_key_head_dim':128,'linear_value_head_dim':128,
            'rope_parameters':{'rope_theta':10000000,'partial_rotary_factor':.25}}}
        model=Model(ModelArgs.from_dict(config));model.eval();nn.quantize(model,bits=4,group_size=64)
        mx.eval(model.parameters());ids=[11,21,31,41,51];source,_=prefill(model,ids)
        saved=[mx.array(t) for t in cache_arrays(source)]
        compiled=CompiledStep(model,(127,));kwargs=dict(prefix_cache=source,compiled_step=compiled,ignore_stops=True,suppress_tokens=(127,))
        reference=batched_generate(model,Tokenizer(),ids,11,**kwargs)
        limits=dict(zip(HEADS,(3,5,8,11)))
        for chunk in (1,4):
            actual=batched_generate(model,Tokenizer(),ids,11,head_token_limits=limits,decode_chunk_size=chunk,**kwargs)
            for h in HEADS:
                self.assertEqual(actual['channels'][h]['token_ids'],reference['channels'][h]['token_ids'][:limits[h]])
                self.assertEqual(actual['channels'][h]['stop_reason'],'external_token_limit')
            for before,after in zip(saved,cache_arrays(source)):self.assertTrue(mx.array_equal(before,after).item())
            if chunk==1:self.assertEqual(actual['stats']['active_batch_by_step'],[4]*3+[3]*2+[2]*3+[1]*3)
        for bad in ({},dict(zip(HEADS,(0,5,8,11))),dict(zip(HEADS,(3,5,8,12)))):
            with self.assertRaises(ValueError):batched_generate(model,Tokenizer(),ids,11,head_token_limits=bad,**kwargs)

if __name__=='__main__':unittest.main()
