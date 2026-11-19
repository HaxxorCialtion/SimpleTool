import unittest
import mlx.core as mx
import mlx.nn as nn
from simpletool.compact_metadata import install_compact_metadata
from simpletool.row_reuse_int4 import install_row_reuse
from simpletool.compact_qmv import compact_qmv


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(320,257,bias=True)
        self.embed=nn.Embedding(257,320)
        # Only offline quantization input is half; all execution remains FP32.
        self.linear.weight=self.linear.weight.astype(mx.float16)
        self.embed.weight=self.embed.weight.astype(mx.float16)
        nn.quantize(self,group_size=64,bits=4)
        self.apply(lambda a:a.astype(mx.float32) if mx.issubdtype(a.dtype,mx.floating) else a)


class CompactMetadata(unittest.TestCase):
    def test_direct_qmv_stock_order_strides_and_unaligned_rejection(self):
        mx.random.seed(109)
        for n,k in ((8,512),(64,512),(128,1024),(9216,2560),(2560,9216)):
            q,s,b=mx.quantize((mx.random.normal((n,k))*.02).astype(mx.float16),bits=4,group_size=64)
            x=mx.random.normal((1,k*2))[:,::2]
            expected=mx.quantized_matmul(x,q,s.astype(mx.float32),b.astype(mx.float32),bits=4,group_size=64)
            actual=compact_qmv(x,q,s,b)
            self.assertTrue(mx.array_equal(expected,actual).item(),(n,k,float(mx.max(mx.abs(expected-actual)).item())))
        q,s,b=mx.quantize(mx.zeros((79,512),mx.float16),bits=4,group_size=64)
        with self.assertRaises(ValueError):compact_qmv(mx.ones((1,512)),q,s,b)

    def test_lossless_storage_strides_prefill_decode_and_embedding(self):
        mx.random.seed(103);model=Model();install_row_reuse(model)
        tokens=mx.array([[0,256,17],[18,253,7]])
        embedded=model.embed(tokens)
        inputs=[mx.random.normal((m,640))[:,::2] for m in (1,2,3,4,40)]
        linear=[model.linear(x) for x in inputs]
        projected=[model.embed.as_linear(x) for x in inputs]
        mx.eval(embedded,linear,projected)
        packed_before=mx.array(model.linear.weight)
        receipt=install_compact_metadata(model)
        self.assertEqual(receipt['metadata_before_bytes'],2*receipt['metadata_after_bytes'])
        self.assertTrue(mx.array_equal(packed_before,model.linear.weight).item())
        self.assertEqual(model.embed(tokens).dtype,mx.float32)
        self.assertTrue(mx.array_equal(embedded,model.embed(tokens)).item())
        for x,l,p in zip(inputs,linear,projected):
            self.assertEqual(model.linear(x).dtype,mx.float32)
            self.assertTrue(mx.array_equal(l,model.linear(x)).item(),x.shape)
            self.assertTrue(mx.array_equal(p,model.embed.as_linear(x)).item(),x.shape)
        stored=model.embed.scales
        repeated=install_compact_metadata(model)
        self.assertEqual(repeated['saved_bytes'],0)
        self.assertIs(model.embed.scales,stored)
        self.assertTrue(mx.array_equal(embedded,model.embed(tokens)).item())
        with self.assertRaises(ValueError):model.linear(inputs[0].astype(mx.float16))

    def test_rejects_inexact_metadata_without_partial_mutation(self):
        model=Model();original_class=model.linear.__class__
        original_scales=model.linear.scales
        model.embed.scales=mx.full(model.embed.scales.shape,.1234567,mx.float32)
        with self.assertRaises(ValueError):install_compact_metadata(model)
        self.assertIs(model.linear.scales,original_scales)
        self.assertIs(model.linear.__class__,original_class)


if __name__=='__main__':unittest.main()
