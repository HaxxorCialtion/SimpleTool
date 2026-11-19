import unittest
import mlx.core as mx
import mlx.nn as nn
from simpletool.kernels import int4_matmul
from simpletool.projection_fusion import merge
from simpletool.split_projection import SplitProjection


class SplitOutputs(unittest.TestCase):
    def test_packed_parameters_and_outputs(self):
        mx.random.seed(61)
        for sizes in ((37,42),(37,23,7,12)):
            modules=[]
            for rows in sizes:
                module=nn.QuantizedLinear(256,rows,bias=False,bits=4,group_size=64)
                modules.append(module)
            merged=merge(*modules,aliases=modules)
            split=SplitProjection(merged,sizes)
            for m in (1,2,3,4,5):
                # Strided inputs also exercise Metal's contiguous-input handling.
                x=mx.random.normal((m,512))[:,::2]
                outputs=split(x)
                reference=(int4_matmul(x,merged.weight,merged.scales,merged.biases)
                           if 2<=m<=4 else merged(x))
                expected=mx.split(reference,[sum(sizes[:i]) for i in range(1,len(sizes))],axis=-1)
                for a,b in zip(outputs,expected):
                    self.assertTrue(mx.array_equal(a,b).item(),(sizes,m))
            for key in ('weight','scales','biases'):
                self.assertTrue(mx.array_equal(mx.concatenate([getattr(a,key) for a in modules]),getattr(merged,key)).item())


if __name__=='__main__':unittest.main()
