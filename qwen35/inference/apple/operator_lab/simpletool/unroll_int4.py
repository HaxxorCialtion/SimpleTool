"""Explicit constant-loop scalarisation for the original FP32 int4 dot order.

Representative AIR IR retains dynamically indexed local arrays without pragmas.
Unroll row/branch loops, and separately control the eight nibble iterations.
This does not establish hardware register/spill counts. Performance is empirical.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from .fma_int4 import SOURCE as ROW_SOURCE,eligible
from .compact_metadata import CompactInt4Linear,CompactInt4Embedding

MODES=('j1','j2','j4','j8')
SOURCES={};KERNELS={}
for mode in MODES:
 source=ROW_SOURCE.replace('for(uint r=','_Pragma("clang loop unroll(full)")\n    for(uint r=').replace('for(uint m=','_Pragma("clang loop unroll(full)")\n    for(uint m=')
 pragma='clang loop unroll(disable)' if mode=='j1' else 'clang loop unroll(full)' if mode=='j8' else f'clang loop unroll_count({mode[1:]})'
 source=source.replace('for(uint j=0;',f'_Pragma("{pragma}")\n        for(uint j=0;')
 SOURCES[mode]=source;KERNELS[mode]=mx.fast.metal_kernel(name='simpletool_unroll_int4_'+mode,input_names=['x','w','scales','biases'],output_names=['y'],source=source)

def unroll_int4(x,w,scales,biases,mode='j4',rows=8,threads=256):
 k=x.shape[-1];m=x.size//k;n=w.shape[0]
 if (mode not in MODES or x.dtype!=mx.float32 or w.dtype!=mx.uint32 or not 1<=m<=4 or k%64
     or w.shape!=(n,k//8) or scales.shape!=(n,k//64) or biases.shape!=scales.shape
     or scales.dtype not in (mx.float16,mx.float32) or biases.dtype!=scales.dtype
     or rows not in (1,2,4,8) or threads not in (64,128,256)):
  raise ValueError('Unroll candidate requires FP32 B1..4 affine int4/group64 and valid tile')
 return KERNELS[mode](inputs=[x,w,scales,biases],
  template=[('K',k),('N',n),('M',m),('ROWS',rows),('ACC_FMA',False),('DQ_FMA',False)],
  grid=(math.ceil(n/(rows*(threads//32)))*threads,1,1),threadgroup=(threads,1,1),
  output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.float32])[0]

class UnrollLinear(CompactInt4Linear):
 def __call__(self,x):
  if not eligible(self,x):return super().__call__(x)
  y=unroll_int4(x,self.weight,self.scales,self.biases,self._unroll_mode,*self._row_reuse_config)
  return y+self.bias if 'bias' in self else y
class UnrollEmbedding(CompactInt4Embedding):
 def as_linear(self,x):
  if not eligible(self,x):return super().as_linear(x)
  return unroll_int4(x,self.weight,self.scales,self.biases,self._unroll_mode,*self._row_reuse_config)

def install_unroll_int4(model,mode='j4'):
 if mode not in MODES:raise ValueError('Invalid unroll mode')
 modules=[m for _,m in model.named_modules() if isinstance(m,(nn.QuantizedLinear,nn.QuantizedEmbedding))]
 if any(type(m) not in (CompactInt4Linear,CompactInt4Embedding) for m in modules):raise ValueError('Original compact int4 modules required')
 for m in modules:
  m.__class__=UnrollEmbedding if isinstance(m,CompactInt4Embedding) else UnrollLinear;m._unroll_mode=mode
