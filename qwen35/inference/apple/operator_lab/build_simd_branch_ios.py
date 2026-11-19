"""Compile candidate bodies for iOS; no device install or model execution."""
import ast,subprocess,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
tree=ast.parse((ROOT/'simpletool/simd_branch_qmv.py').read_text())
body=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE' for t in n.targets))
parts=['#include <metal_stdlib>\nusing namespace metal;\n'];cases=[]
for batch in range(1,5):
    for rows in (2,4,8):
        for shared in (False,True):
            for dtype in ('half','float'):
                name=f'candidate_b{batch}_r{rows}_s{int(shared)}_{dtype}'
                prefix=f'#define M {batch}\n#define ROWS {rows}\n#define N 79\n#define K 2048\n#define SHARED {str(shared).lower()}\n'
                signature=f'''kernel void {name}(device const float* x [[buffer(0)]],device const uint* w [[buffer(1)]],
device const {dtype}* scales [[buffer(2)]],device const {dtype}* biases [[buffer(3)]],device float* y [[buffer(4)]],
uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {{\n'''
                parts.append(prefix+signature+body+'\n}\n#undef M\n#undef ROWS\n#undef N\n#undef K\n#undef SHARED\n')
                cases.append(dict(batch=batch,rows=rows,shared=shared,metadata=dtype,name=name))
metal=ROOT/'artifacts/simd_branch_ios_20261008.metal';metal.write_text(''.join(parts))
air=metal.with_suffix('.air');library=metal.with_suffix('.metallib');log=ROOT/'reports/simd_branch_ios_build_20261008.log'
sdk=subprocess.check_output(['xcrun','--sdk','iphoneos','--show-sdk-path'],text=True).strip()
commands=[['xcrun','-sdk','iphoneos','metal','-c',str(metal),'-o',str(air)],
          ['xcrun','-sdk','iphoneos','metallib',str(air),'-o',str(library)],
          ['xcrun','swiftc','-target','arm64-apple-ios17.0','-sdk',sdk,'-typecheck',str(ROOT/'native/bench_payload_bandwidth.swift')]]
with log.open('w') as output:
    for command in commands:
        code=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT).returncode
        if code:raise SystemExit(code)
report=dict(all_compiled=True,variants=cases,swift_calibration_typecheck=True,phone_executed=False,
            source_sha256=hashlib.sha256(body.encode()).hexdigest(),metallib_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
(ROOT/'reports/simd_branch_ios_build_20261008.json').write_text(json.dumps(report,indent=2)+'\n')
print(len(cases),'iOS Metal variants compiled; native calibration Swift typecheck passed')
