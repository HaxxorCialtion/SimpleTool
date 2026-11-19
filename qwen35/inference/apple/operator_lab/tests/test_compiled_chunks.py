import unittest
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import Model,ModelArgs
from simpletool.cache import cache_arrays
from simpletool.compiled import CompiledStep
from simpletool.decode import HEADS,batched_generate,prefill,serial_generate
from simpletool.row_reuse_int4 import install_row_reuse
from test_core import Tokenizer


class CompiledDecodeBlocks(unittest.TestCase):
    def test_partial_last_block_and_immutable_fork_source(self):
        mx.random.seed(83)
        config={'model_type':'qwen3_5','text_config':{
            'model_type':'qwen3_5_text','hidden_size':64,'intermediate_size':128,
            'num_hidden_layers':4,'num_attention_heads':4,'num_key_value_heads':2,
            'head_dim':32,'vocab_size':128,'tie_word_embeddings':True,
            'linear_num_key_heads':2,'linear_num_value_heads':2,
            'linear_key_head_dim':128,'linear_value_head_dim':128,
            'rope_parameters':{'rope_theta':10000000,'partial_rotary_factor':.25}}}
        model=Model(ModelArgs.from_dict(config));model.eval();nn.quantize(model,bits=4,group_size=64)
        mx.eval(model.parameters())
        ids=[11,21,31,41,51];source,_=prefill(model,ids)
        snapshot=[mx.array(t) for t in cache_arrays(source)]
        gold=serial_generate(model,Tokenizer(),ids,11,prefix_cache=source)
        install_row_reuse(model);step=CompiledStep(model)
        for chunk in (1,2,4,8):
            actual=batched_generate(model,Tokenizer(),ids,11,prefix_cache=source,
                                    compiled_step=step,decode_chunk_size=chunk)
            for h in HEADS:self.assertEqual(actual['channels'][h]['token_ids'],gold['channels'][h]['token_ids'],(chunk,h))
            for before,after in zip(snapshot,cache_arrays(source)):
                self.assertTrue(mx.array_equal(before,after).item())


if __name__=='__main__':unittest.main()
