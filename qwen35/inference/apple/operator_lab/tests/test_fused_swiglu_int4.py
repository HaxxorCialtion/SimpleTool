import unittest
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_next import swiglu
from simpletool.row_reuse_int4 import row_reuse_int4
from simpletool.fused_swiglu_int4 import fused_swiglu_int4

class FusedSwiGLU(unittest.TestCase):
    def test_exact_projection_and_activation_with_tails(self):
        mx.random.seed(179)
        for n,k in ((79,256),(512,2560),(256,2048)):
            gate=nn.QuantizedLinear(k,n,bias=False,group_size=64,bits=4)
            up=nn.QuantizedLinear(k,n,bias=False,group_size=64,bits=4)
            for narrow in (False,True):
                if narrow:
                    for p in (gate,up):p.scales=p.scales.astype(mx.float16);p.biases=p.biases.astype(mx.float16)
                for batch in (2,3,4):
                    x=mx.random.normal((batch,k*2))[:,::2]
                    g=row_reuse_int4(x,gate.weight,gate.scales,gate.biases,4,128)
                    u=row_reuse_int4(x,up.weight,up.scales,up.biases,4,128)
                    reference=swiglu(g,u)
                    for rows in (1,2,4,8):
                        actual=fused_swiglu_int4(x,gate,up,rows,128)
                        self.assertTrue(mx.array_equal(actual,reference).item(),(n,k,batch,narrow,rows,
                            float(mx.max(mx.abs(actual-reference)).item())))

if __name__=='__main__':unittest.main()
