"""Generated constant-index FP32 int4 QMV; identical per-lane arithmetic order.

Remove source-level dynamic local array indexing. The packed payload and
metadata stay immutable. AIR alloca elimination is not GPU spill evidence.
"""
import math
import mlx.core as mx
import mlx.nn as nn
from .fma_int4 import eligible
from .compact_metadata import CompactInt4Linear,CompactInt4Embedding

SOURCES={};KERNELS={}
def source_for(m,rows):
 if not 1<=m<=4 or rows not in (1,2,4,8):raise ValueError('Scalar QMV M/rows')
 key=(m,rows)
 if key in SOURCES:return SOURCES[key]
 code=['const uint row0=(thread_position_in_grid.x/32)*'+str(rows)+';','const uint lane=thread_index_in_simdgroup;','if(row0>=N) return;']
 for r in range(rows):
  for b in range(m):code.append(f'float acc_{r}_{b}=0.0f;')
 code.append('for(uint word=lane;word<K/8;word+=32) {')
 for r in range(rows):
  code += [f'const uint row_{r}=row0+{r};',f'const uint packed_{r}=row_{r}<N?w[row_{r}*(K/8)+word]:0u;',
           f'const float s_{r}=row_{r}<N?float(scales[row_{r}*(K/64)+word/8]):0.0f;',
           f'const float b_{r}=row_{r}<N?float(biases[row_{r}*(K/64)+word/8]):0.0f;']
 for j in range(8):
  code.append('{')
  for b in range(m):code.append(f'const float input_{b}=x[{b}*K+word*8+{j}];')
  for r in range(rows):
   code.append(f'const float value_{r}=float((packed_{r}>>{4*j})&15)*s_{r}+b_{r};')
   for b in range(m):code.append(f'acc_{r}_{b}+=value_{r}*input_{b};')
  code.append('}')
 code.append('}')
 for r in range(rows):
  for b in range(m):
   code += ['{',f'const float sum=simd_sum(acc_{r}_{b});',f'if(lane==0 && row0+{r}<N) y[{b}*N+row0+{r}]=sum;','}']
 SOURCES[key]='\n'.join(code);return SOURCES[key]

def scalar_int4(x,w,scales,biases,rows=8,threads=256):
 k=x.shape[-1];m=x.size//k;n=w.shape[0]
 if (x.dtype!=mx.float32 or w.dtype!=mx.uint32 or not 1<=m<=4 or k%64
     or w.shape!=(n,k//8) or scales.shape!=(n,k//64) or biases.shape!=scales.shape
     or scales.dtype not in (mx.float16,mx.float32) or biases.dtype!=scales.dtype
     or rows not in (1,2,4,8) or threads not in (64,128,256)):
  raise ValueError('Scalar QMV requires FP32 B1..4 affine int4/group64 and valid tile')
 key=(m,rows)
 if key not in KERNELS:KERNELS[key]=mx.fast.metal_kernel(name=f'simpletool_scalar_int4_m{m}_r{rows}',input_names=['x','w','scales','biases'],output_names=['y'],source=source_for(m,rows))
 return KERNELS[key](inputs=[x,w,scales,biases],template=[('K',k),('N',n)],
  grid=(math.ceil(n/(rows*(threads//32)))*threads,1,1),threadgroup=(threads,1,1),output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.float32])[0]

class ScalarLinear(CompactInt4Linear):
 def __call__(self,x):
  if not eligible(self,x) or (self._scalar_policy=='vocab' and self.weight.shape[0]<65536):return super().__call__(x)
  y=scalar_int4(x,self.weight,self.scales,self.biases,*self._row_reuse_config)
  return y+self.bias if 'bias' in self else y
class ScalarEmbedding(CompactInt4Embedding):
 def as_linear(self,x):
  if not eligible(self,x):return super().as_linear(x)
  cfg=((8 if x.size//x.shape[-1]>=3 else 4),128) if self._scalar_policy=='vocab' else self._row_reuse_config
  return scalar_int4(x,self.weight,self.scales,self.biases,*cfg)
def install_scalar_int4(model,policy='all'):
 if policy not in ('all','vocab'):raise ValueError('Scalar policy all/vocab required')
 modules=[m for _,m in model.named_modules() if isinstance(m,(nn.QuantizedLinear,nn.QuantizedEmbedding))]
 if any(type(m) not in (CompactInt4Linear,CompactInt4Embedding) for m in modules):raise ValueError('Original compact int4 required')
 for m in modules:
  m.__class__=ScalarEmbedding if isinstance(m,CompactInt4Embedding) else ScalarLinear
  m._scalar_policy=policy
