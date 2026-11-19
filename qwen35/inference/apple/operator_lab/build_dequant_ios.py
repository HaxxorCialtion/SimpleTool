"""Standalone iOS compiler validation of explicit-FMA/LUT/static index kernels."""
import hashlib,json,subprocess
from pathlib import Path
from simpletool.fma_int4 import SOURCE as FMA,MODES as FMAMODES
from simpletool.lut_int4 import SOURCE as LUT
from simpletool.unroll_int4 import SOURCES as UNROLL
from simpletool.scalar_int4 import source_for
ROOT=Path(__file__).resolve().parent;parts=['#include <metal_stdlib>\nusing namespace metal;\n'];cases=[]
shapes=[(9216,2560),(2560,9216),(248320,2560),(6144,2048),(2048,6144),(248320,2048)]
for n,k in shapes:
 for m in (1,2,3,4):
  for rows in ((4,8) if n>=65536 and m>=3 else (4,) if n>=65536 else (8,)):
   for metadata in ('half','float'):
    modes={**{name:(FMA,*flags) for name,flags in FMAMODES.items()},'lut':(LUT,False,False),
           **{name:(body,False,False) for name,body in UNROLL.items()},'scalar':(source_for(m,rows),False,False)}
    for mode,(body,acc,dq) in modes.items():
     name=f'deq_n{n}_k{k}_m{m}_{metadata}_{mode}_r{rows}';constants=dict(N=n,K=k,M=m,ROWS=rows)
     prefix=''.join(f'#define {q} {v}\n' for q,v in constants.items())+f'#define ACC_FMA {str(acc).lower()}\n#define DQ_FMA {str(dq).lower()}\n'
     signature=f'kernel void {name}(device const float* x [[buffer(0)]],device const uint* w [[buffer(1)]],device const {metadata}* scales [[buffer(2)]],device const {metadata}* biases [[buffer(3)]],device float* y [[buffer(4)]],uint3 thread_position_in_grid [[thread_position_in_grid]],uint thread_index_in_simdgroup [[thread_index_in_simdgroup]])'
     parts.append(prefix+signature+' {\n'+body+'\n}\n'+''.join(f'#undef {q}\n' for q in (*constants,'ACC_FMA','DQ_FMA')))
     cases.append(dict(name=name,n=n,k=k,batch=m,metadata=metadata,mode=mode,rows=rows))
metal=ROOT/'artifacts/dequant_ios_20261008.metal';metal.write_text(''.join(parts));air=metal.with_suffix('.air');library=metal.with_suffix('.metallib')
commands=[['xcrun','-sdk','iphoneos','metal','-fno-fast-math','-c',str(metal),'-o',str(air)],['xcrun','-sdk','iphoneos','metallib',str(air),'-o',str(library)]]
with (ROOT/'reports/dequant_ios_build_20261008.log').open('w') as f:
 for command in commands:
  code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
  if code:raise SystemExit(code)
out=dict(variants=cases,all_compiled=True,fast_math=False,phone_executed=False,full_swift_candidate_integrated=False,commands=commands,
 metal_sha256=hashlib.sha256(metal.read_bytes()).hexdigest(),metallib_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
 scope='Six representative projection shapes, B1..4, FP32 compute with half/float metadata; standalone shader compile, not full Swift model or phone execution')
(ROOT/'reports/dequant_ios_build_20261008.json').write_text(json.dumps(out,indent=2)+'\n');print(len(cases),'iOS shader variants compiled, not phone run')
