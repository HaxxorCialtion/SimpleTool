import unittest
import mlx.core as mx
import mlx.nn as nn
from simpletool.gdn_conv import conv_state


class SingleStepConvolution(unittest.TestCase):
    def test_history_shift_numerics_strides_and_tail_threads(self):
        mx.random.seed(97)
        for B in (1,2,3,4):
            for D in (65,6144,8192):
                history=mx.random.normal((B,3,D*2))[:,:,::2]
                qkv=mx.random.normal((B,1,D*2))[:,:,::2]
                weights=mx.random.normal((D,8,1))[:,::2,:]*.2
                before=mx.array(history)
                full=mx.concatenate([history,qkv],axis=1)
                expected=nn.silu(mx.conv1d(full,weights,groups=D))
                actual,next_history=conv_state(history,qkv,weights)
                self.assertTrue(mx.array_equal(actual,expected).item(),(B,D))
                self.assertTrue(mx.array_equal(next_history,full[:,-3:,:]).item())
                self.assertTrue(mx.array_equal(history,before).item())


if __name__=='__main__':unittest.main()
