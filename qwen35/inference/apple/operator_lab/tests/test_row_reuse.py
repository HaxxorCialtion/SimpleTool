import unittest
import mlx.core as mx
from simpletool.kernels import int4_matmul
from simpletool.row_reuse_int4 import row_reuse_int4


class RowRegisterReuse(unittest.TestCase):
    def test_partial_rows_short_groups_strides_and_batch_shrink(self):
        mx.random.seed(73)
        for n,k in ((79,256),(129,320),(64,2560)):
            packed,s,b=mx.quantize(mx.random.normal((n,k))*.03,bits=4,group_size=64)
            for m in (1,2,3,4):
                x=mx.random.normal((m,k*2))[:,::2]
                expected=int4_matmul(x,packed,s,b)
                for rows in (1,2,4,8):
                    for threads in (64,128,256):
                        actual=row_reuse_int4(x,packed,s,b,rows,threads)
                        self.assertTrue(mx.array_equal(actual,expected).item(),(n,k,m,rows,threads))


if __name__=='__main__':unittest.main()
