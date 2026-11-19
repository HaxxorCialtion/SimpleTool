import unittest
import mlx.core as mx
import mlx.nn as nn
from simpletool.vision import PositionRope


class Rope(unittest.TestCase):
    def test_equal_positions_reduce_to_text_rope(self):
        base = nn.RoPE(64, traditional=False, base=10000000)
        rope = PositionRope(base,64,10000000,[11,11,10])
        x = mx.random.normal((2,4,7,256)).astype(mx.float16)
        rope.positions = mx.broadcast_to(mx.arange(19,26)[None],(3,7))
        self.assertTrue(mx.allclose(rope(x),base(x,offset=19),atol=0.002,rtol=0.002).item())

    def test_interleaved_spatial_axes(self):
        base = nn.RoPE(64,traditional=False,base=10000000)
        rope = PositionRope(base,64,10000000,[11,11,10])
        rope.positions = mx.array([[2],[7],[11]])
        x = mx.concatenate([mx.ones((1,1,1,32)),mx.zeros((1,1,1,224))],axis=-1)
        y = rope(x)
        selected = [2,7,11]*10+[2,7]
        angle = mx.array(selected)*10000000**(-mx.arange(0,64,2,dtype=mx.float32)/64)
        self.assertTrue(mx.allclose(y[0,0,0,:32],mx.cos(angle),atol=1e-6).item())
        self.assertTrue(mx.allclose(y[0,0,0,32:64],mx.sin(angle),atol=1e-6).item())


if __name__ == '__main__':unittest.main()
