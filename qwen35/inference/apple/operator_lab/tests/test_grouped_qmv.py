import unittest
import mlx.core as mx
from simpletool.grouped_qmv import grouped_qmv
from simpletool.stream_grouped_qmv import stream_grouped_qmv


class GroupedQMV(unittest.TestCase):
    def test_every_branch_matches_stock_single_vector(self):
        mx.random.seed(167)
        for n,k in ((79,512),(64,2560),(520,1536),(512,9216)):
            w,s,b=mx.quantize(mx.random.normal((n,k))*.02,group_size=64,bits=4)
            # Tail output widths use stock generic QMV, which has a different
            # reduction; aligned widths give the exact fast-QMV reference.
            x=mx.random.normal((4,k*2))[:,::2]
            for compact in (False,True):
                ss=s.astype(mx.float16) if compact else s
                bb=b.astype(mx.float16) if compact else b
                reference=mx.concatenate([mx.quantized_matmul(x[i:i+1],w,ss.astype(mx.float32),
                    bb.astype(mx.float32),transpose=True,group_size=64,bits=4) for i in range(4)])
                for m in (1,2,3,4):
                    for rows in (1,2,4,8):
                        for kernel in (grouped_qmv,stream_grouped_qmv):
                            actual=kernel(x[:m],w,ss,bb,rows,128)
                            if n%8==0:self.assertTrue(mx.array_equal(actual,reference[:m]).item(),(n,k,m,rows,compact,kernel.__name__))
                            else:self.assertTrue(mx.allclose(actual,reference[:m],atol=2e-5,rtol=2e-5).item())


if __name__=='__main__':unittest.main()
