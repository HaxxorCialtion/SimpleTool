"""Build native Metal projection candidates from existing owned kernel sources."""
import hashlib
import json
import subprocess
from pathlib import Path

from simpletool.fma_int4 import SOURCE
from simpletool.scalar_int4 import source_for
from simpletool.unroll_int4 import SOURCES

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'artifacts/weighted_qmv_20261008'
out.mkdir(exist_ok=True)
prefix = '''#include <metal_stdlib>
using namespace metal;
constant uint K [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant bool ACC_FMA=false;
constant bool DQ_FMA=false;
'''


def wrapper(name, body):
    return f'''kernel void {name}(device const float* x [[buffer(0)]],
 device const uint* w [[buffer(1)]], device const half* scales [[buffer(2)]],
 device const half* biases [[buffer(3)]], device float* y [[buffer(4)]],
 uint3 thread_position_in_grid [[thread_position_in_grid]],
 uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]) {{
{body}
}}
'''


parts = [prefix]
for batch in range(1, 5):
    for rows in (1, 2, 4, 8):
        for mode, body in [('baseline', SOURCE), ('j8', SOURCES['j8']), ('scalar', source_for(batch, rows))]:
            parts.append(f'#define M {batch}\n#define ROWS {rows}\n')
            parts.append(wrapper(f'weighted_{mode}_m{batch}_r{rows}', body))
            parts.append('#undef M\n#undef ROWS\n')
metal = out / 'weighted.metal'
metal.write_text(''.join(parts))
records = []
for sdk in ('macosx', 'iphoneos'):
    air, library = out / f'{sdk}.air', out / f'{sdk}.metallib'
    commands = [['xcrun', '-sdk', sdk, 'metal', '-fno-fast-math', '-c', str(metal), '-o', str(air)],
                ['xcrun', '-sdk', sdk, 'metallib', str(air), '-o', str(library)]]
    with (out / f'{sdk}_build.log').open('w') as f:
        for command in commands:
            code = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT).returncode
            if code:
                raise SystemExit(code)
    records.append(dict(sdk=sdk, commands=commands, sha256=hashlib.sha256(library.read_bytes()).hexdigest()))
    print(sdk, '48 functions compiled; N/K specialization is runtime, not phone run', flush=True)
binary = out / 'bench_weighted_qmv'
cmd = ['xcrun', 'swiftc', '-O', '-framework', 'Metal', str(ROOT / 'native/bench_weighted_qmv.swift'), '-o', str(binary)]
with (out / 'swift_build.log').open('w') as f:
    code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if code:
        raise SystemExit(code)
result = dict(libraries=records, source_sha256=hashlib.sha256(metal.read_bytes()).hexdigest(),
              binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), fast_math=False,
              phone_executed=False, functions=48, swift_command=cmd)
(ROOT / 'reports/weighted_qmv_build_20261008.json').write_text(json.dumps(result, indent=2) + '\n')
print('Native binary built', flush=True)
