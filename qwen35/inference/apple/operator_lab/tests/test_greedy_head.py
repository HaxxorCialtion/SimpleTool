import unittest
import mlx.core as mx
import mlx.nn as nn
from simpletool.greedy_head import greedy_int4
from simpletool.row_reuse_int4 import install_row_reuse


def embedding(n,k,compact=False):
    head=nn.Embedding(n,k);head.weight=head.weight.astype(mx.float16)
    root=nn.Module();root.head=head;nn.quantize(root,bits=4,group_size=64)
    root.apply(lambda a:a.astype(mx.float32) if mx.issubdtype(a.dtype,mx.floating) else a)
    install_row_reuse(root)
    if compact:
        from simpletool.compact_metadata import install_compact_metadata
        install_compact_metadata(root)
    return root.head


def reference(head,x,suppressed):
    logits=head.as_linear(x)
    if suppressed:logits[:,list(suppressed)]=-mx.inf
    return mx.argmax(logits,axis=-1)


class TiledGreedyHead(unittest.TestCase):
    def test_partial_threadgroups_strides_metadata_and_batch_shrink(self):
        mx.random.seed(127)
        for n,k in ((257,256),(1025,320)):
            for compact in (False,True):
                head=embedding(n,k,compact)
                for m in (1,2,3,4):
                    x=mx.random.normal((m,k*2))[:,::2];suppressed=(0,1)
                    expected=reference(head,x,suppressed)
                    for rows in (1,2,4,8):
                        for threads in (64,128,256):
                            actual=greedy_int4(head,x,suppressed,rows,threads)
                            self.assertTrue(mx.array_equal(actual,expected).item(),(n,k,m,compact,rows,threads))

    def test_ties_all_suppressed_and_nan_precedence(self):
        head=embedding(17,256)
        head.scales=mx.zeros_like(head.scales);head.biases=mx.zeros_like(head.biases)
        for x in (mx.zeros((4,256)),mx.full((4,256),mx.nan)):
            for suppressed in ((),(0,1,2),tuple(range(17))):
                actual=greedy_int4(head,x,suppressed,rows=8,threads=256)
                self.assertTrue(mx.array_equal(actual,reference(head,x,suppressed)).item(),suppressed)


if __name__=='__main__':unittest.main()
