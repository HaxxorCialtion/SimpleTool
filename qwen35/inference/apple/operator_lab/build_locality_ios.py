"""Compile runtime-phase/fused-history kernels for iOS; not device execution."""
import hashlib
import json
import subprocess
from pathlib import Path
from simpletool.locality_gdn import HeadLocalityState,ValueLocalityState

ROOT=Path(__file__).resolve().parent
parts=['#include <metal_stdlib>\nusing namespace metal;\n'];cases=[]
for order,State in [('head',HeadLocalityState),('value',ValueLocalityState)]:
 for hk,hv,dv in ((16,32,128),(16,16,128),(1,1,8)):
  for sb in (1,4):
   for capacity in (1,3,7,15):
    for operation in ('defer','write','fused_defer','fused_write','materialize'):
     name=f'locality_{order}_h{hk}_{hv}_d{dv}_sb{sb}_c{capacity}_{operation}'
     constants=dict(Hk=hk,Hv=hv,Dv=dv,SB=sb,B=4,C=capacity)
     prefix=''.join(f'#define {k} {v}\n' for k,v in constants.items())
     prefix+=f'#define WRITE {str(operation.endswith("write")).lower()}\n'
     if operation=='materialize':
         inputs=['base','history_k','history_g','history_delta','pending'];outputs=['state_out'];body=State.sources['materialize']
     else:
         inputs=['q','k','v','g','beta','base','history_k','history_g','history_delta','pending']
         outputs=['y','delta_out','state_out'];body=State.sources['step']
         if operation.startswith('fused'):
             outputs+=['history_k_out','history_g_out','history_delta_out'];body=State.sources['fused']
     args=[(f'constant int& {a}' if a=='pending' else f'device const float* {a}')+f' [[buffer({i})]]' for i,a in enumerate(inputs)]
     args += [f'device float* {a} [[buffer({len(inputs)+i})]]' for i,a in enumerate(outputs)]
     args += ['uint3 thread_position_in_grid [[thread_position_in_grid]]','uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]']
     parts.append(prefix+f'kernel void {name}('+',\n'.join(args)+') {\n'+body+'\n}\n')
     parts.append(''.join(f'#undef {k}\n' for k in (*constants,'WRITE')))
     cases.append(dict(name=name,dispatch_order=order,key_heads=hk,value_heads=hv,value_dim=dv,stored_batch=sb,capacity=capacity,operation=operation))
metal=ROOT/'artifacts/locality_ios_20261008.metal';metal.write_text(''.join(parts))
air=metal.with_suffix('.air');library=metal.with_suffix('.metallib')
commands=[['xcrun','-sdk','iphoneos','metal','-fno-fast-math','-c',str(metal),'-o',str(air)],
          ['xcrun','-sdk','iphoneos','metallib',str(air),'-o',str(library)]]
with (ROOT/'reports/locality_ios_build_20261008.log').open('w') as f:
 for command in commands:
    code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
    if code:raise SystemExit(code)
report=dict(all_compiled=True,variants=cases,fast_math=False,phone_executed=False,
            kernel_source_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in
                                 [(order+'_'+name,body) for order,State in [('head',HeadLocalityState),('value',ValueLocalityState)] for name,body in State.sources.items()]},
            metallib_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),commands=commands)
(ROOT/'reports/locality_ios_build_20261008.json').write_text(json.dumps(report,indent=2)+'\n')
print(len(cases),'iOS variants compiled; dispatch permutations, runtime phase; not full Swift integration or phone executed')
