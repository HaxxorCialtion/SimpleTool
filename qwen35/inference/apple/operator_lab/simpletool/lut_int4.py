"""Experimental SIMD-shuffled FP32 dequantization lookup, same row arithmetic.

Eight word lanes in each quantization group share sixteen FP32 dequant values.
Each lane computes two levels and two shuffles select the exact level. This
trades repeated per-nibble dequant arithmetic for shuffle instructions. No speed
benefit is assumed. Same int4 payload; no dense matrix or FP16 activation.
"""
import math
import mlx.core as mx
from .fma_int4 import SOURCE as ROW_SOURCE

SOURCE=ROW_SOURCE.replace('uint packed[ROWS];float s[ROWS],b[ROWS];',
    'uint packed[ROWS];float s[ROWS],b[ROWS],levels0[ROWS],levels1[ROWS];')
SOURCE=SOURCE.replace('        for(uint j=0;j<8;++j) {',r'''
        for(uint r=0;r<ROWS;++r) {
            const float code=float(lane%8);
            levels0[r]=code*s[r]+b[r];
            levels1[r]=(code+8.0f)*s[r]+b[r];
        }
        for(uint j=0;j<8;++j) {''')
SOURCE=SOURCE.replace('const float code=float((packed[r]>>(4*j))&15);\n                const float value=DQ_FMA ? metal::fma(code,s[r],b[r]) : code*s[r]+b[r];',r'''
const uint code=(packed[r]>>(4*j))&15;
                const uint from=(lane&~7u)+(code&7u);
                const float lower=simd_shuffle(levels0[r],from);
                const float upper=simd_shuffle(levels1[r],from);
                const float value=code<8?lower:upper;''')
# Preserve original accumulation expression, including default contraction.
SOURCE=SOURCE.replace('if constexpr(ACC_FMA) acc[r][m]=metal::fma(value,input[m],acc[r][m]);\n                    else acc[r][m]+=value*input[m];','acc[r][m]+=value*input[m];')
_KERNEL=mx.fast.metal_kernel(name='simpletool_simd_lut_int4',input_names=['x','w','scales','biases'],output_names=['y'],source=SOURCE)
def lut_int4(x,w,scales,biases,rows=8,threads=256):
    k=x.shape[-1];m=x.size//k;n=w.shape[0]
    if (x.dtype!=mx.float32 or w.dtype!=mx.uint32 or not 1<=m<=4 or k%256
        or w.shape!=(n,k//8) or scales.shape!=(n,k//64) or biases.shape!=scales.shape
        or scales.dtype not in (mx.float16,mx.float32) or biases.dtype!=scales.dtype
        or rows not in (1,2,4,8) or threads not in (64,128,256)):
        raise ValueError('SIMD LUT requires FP32 B1..4, affine int4/group64 K multiple256, valid tile')
    return _KERNEL(inputs=[x,w,scales,biases],template=[('K',k),('N',n),('M',m),('ROWS',rows)],
        grid=(math.ceil(n/(rows*(threads//32)))*threads,1,1),threadgroup=(threads,1,1),
        output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.float32])[0]
