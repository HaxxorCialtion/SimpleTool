"""Losslessly narrow affine-int4 scale/bias storage; every dot product is FP32.

This does not requantize weights or change activation/state precision. The
installer validates *all* metadata before changing any module. Stock MLX qmm
fallback promotes metadata back to FP32; the custom row kernel widens directly
in registers. Embedding lookup widens only the selected token rows.
"""
import mlx.core as mx
import mlx.nn as nn
from .row_reuse_int4 import RowReuseLinear,RowReuseEmbedding,install_row_reuse
from .compact_qmv import compact_qmv


class CompactInt4Linear(RowReuseLinear):
    def __call__(self,x):
        if x.dtype!=mx.float32:raise ValueError('Compact int4 requires FP32 activations')
        if x.size==x.shape[-1] and x.shape[-1]%512==0 and self.weight.shape[0]%8==0:
            out=compact_qmv(x,self.weight,self.scales,self.biases)
            return out+self.bias if 'bias' in self else out
        return super().__call__(x)


class CompactInt4Embedding(RowReuseEmbedding):
    def __call__(self,x):
        # nn.QuantizedEmbedding uses metadata dtype for its output. Explicitly
        # restore selected rows before dequantization to preserve old numerics.
        return mx.dequantize(self.weight[x],self.scales[x].astype(mx.float32),
                            self.biases[x].astype(mx.float32),
                            group_size=64,bits=4,mode='affine')

    def as_linear(self,x):
        if x.dtype!=mx.float32:raise ValueError('Compact int4 requires FP32 activations')
        if x.size==x.shape[-1] and x.shape[-1]%512==0 and self.weight.shape[0]%8==0:
            return compact_qmv(x,self.weight,self.scales,self.biases)
        return super().as_linear(x)


def install_compact_metadata(model):
    prepared=[];before=after=0
    for name,module in model.named_modules():
        if not isinstance(module,(nn.QuantizedLinear,nn.QuantizedEmbedding)):continue
        if (module.bits,module.group_size,module.mode)!=(4,64,'affine'):
            raise ValueError('Compact metadata supports affine int4 group64 only')
        arrays=[]
        for key in ('scales','biases'):
            array=module[key]
            if array.dtype not in (mx.float32,mx.float16):
                raise ValueError(f'{name}.{key}: unsupported metadata dtype')
            compact=array if array.dtype==mx.float16 else array.astype(mx.float16)
            # An exact roundtrip is mandatory, never silently introduce error.
            # Already compact arrays undergo no narrowing. Avoid expanding
            # them merely to compare the same stored values during startup.
            if array.dtype==mx.float32 and not mx.array_equal(array,compact.astype(mx.float32)).item():
                raise ValueError(f'{name}.{key}: not exactly representable in compact storage')
            arrays.append(compact);before+=array.nbytes;after+=compact.nbytes
        prepared.append((module,*arrays))
    install_row_reuse(model)
    for module,scales,biases in prepared:
        module.scales=scales;module.biases=biases
        module.__class__=CompactInt4Embedding if isinstance(module,nn.QuantizedEmbedding) else CompactInt4Linear
    return {'modules':len(prepared),'metadata_before_bytes':before,'metadata_after_bytes':after,
            'saved_bytes':before-after,'all_roundtrip_exact':True,'compute':'float32',
            'packed_weight_bits':4,'single_vector_direct_compact_qmv_when_K_multiple_of512_N_multiple_of8':True,
            'prefill_and_other_stock_fallback_expands_metadata_temporarily':True}
