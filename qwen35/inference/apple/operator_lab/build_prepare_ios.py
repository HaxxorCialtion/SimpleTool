"""Standalone iOS compile of FP32 convolution/QK/history fusion."""
import hashlib,json,subprocess
from pathlib import Path
from simpletool.gdn_prepare import SOURCE
ROOT=Path(__file__).resolve().parent
parts=['#include <metal_stdlib>\nusing namespace metal;\n'];variants=[]
for hk,hv in [(1,1),(1,2),(8,16),(16,16),(16,32)]:
    for B in (1,2,3,4):
        D=(2*hk+hv)*128;name=f'prepare_b{B}_hk{hk}_hv{hv}'
        constants=dict(B=B,D=D,Hk=hk,Hv=hv)
        parts.append(''.join(f'#define {k} {v}\n' for k,v in constants.items()))
        parts.append(f'''kernel void {name}(device const float* history [[buffer(0)]],device const float* qkv [[buffer(1)]],
 device const float* weight [[buffer(2)]],device float* q [[buffer(3)]],device float* k [[buffer(4)]],
 device float* v [[buffer(5)]],device float* next_history [[buffer(6)]],
 uint3 thread_position_in_grid [[thread_position_in_grid]],uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {{
{SOURCE}
}}
''')
        parts.append(''.join(f'#undef {k}\n' for k in constants))
        variants.append(dict(name=name,batch=B,key_heads=hk,value_heads=hv))
metal=ROOT/'artifacts/gdn_prepare_ios_20261008.metal';metal.write_text(''.join(parts))
air=metal.with_suffix('.air');lib=metal.with_suffix('.metallib')
commands=[['xcrun','-sdk','iphoneos','metal','-fno-fast-math','-c',str(metal),'-o',str(air)],
          ['xcrun','-sdk','iphoneos','metallib',str(air),'-o',str(lib)]]
with (ROOT/'reports/gdn_prepare_ios_build_20261008.log').open('w') as f:
    for command in commands:
        code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
        if code:raise SystemExit(code)
result=dict(variants=variants,all_compiled=True,phone_executed=False,full_swift_integrated=False,compute='float32',
            commands=commands,source_sha256=hashlib.sha256(metal.read_bytes()).hexdigest(),
            metallib_sha256=hashlib.sha256(lib.read_bytes()).hexdigest())
(ROOT/'reports/gdn_prepare_ios_build_20261008.json').write_text(json.dumps(result,indent=2)+'\n')
print(len(variants),'iOS shaders compiled; no phone run')
