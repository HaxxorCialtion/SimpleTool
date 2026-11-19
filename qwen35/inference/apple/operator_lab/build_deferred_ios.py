"""Compile exact-order deferred GDN bodies for iOS; no device execution."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
tree = ast.parse((ROOT / 'simpletool/deferred_gdn.py').read_text())
values = {}
def literal(node):
    if isinstance(node, ast.Constant): return node.value
    if isinstance(node, ast.Name): return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return literal(node.left) + literal(node.right)
    raise ValueError(ast.dump(node))
for node in tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in ('SETUP', 'REPLAY', 'STEP', 'MATERIALIZE'):
            values[node.targets[0].id] = literal(node.value)
parts = ['#include <metal_stdlib>\nusing namespace metal;\n']
cases = []
for hk, hv, dv in ((16, 32, 128), (16, 16, 128), (1, 1, 8)):
    for sb in (1, 4):
        for pending in (0, 1, 3, 7, 15):
            for operation in ('defer', 'write', 'materialize'):
                name = f'gdn_{hk}_{hv}_{dv}_sb{sb}_p{pending}_{operation}'
                constants = dict(Hk=hk, Hv=hv, Dv=dv, SB=sb, P=pending)
                defines = ''.join(f'#define {k} {v}\n' for k, v in constants.items())
                defines += f'#define WRITE {str(operation == "write").lower()}\n'
                if operation == 'materialize':
                    inputs = ('base', 'history_k', 'history_g', 'history_delta')
                    outputs = ('state_out',)
                    body = values['MATERIALIZE']
                else:
                    inputs = ('q', 'k', 'v', 'g', 'beta', 'base', 'history_k', 'history_g', 'history_delta')
                    outputs = ('y', 'delta_out', 'state_out')
                    body = values['STEP']
                args = [f'device const float* {a} [[buffer({i})]]' for i, a in enumerate(inputs)]
                args += [f'device float* {a} [[buffer({len(inputs)+i})]]' for i, a in enumerate(outputs)]
                args += ['uint3 thread_position_in_grid [[thread_position_in_grid]]',
                         'uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]']
                parts.append(defines + f'kernel void {name}(' + ',\n'.join(args) + ') {\n' + body + '\n}\n')
                parts.append(''.join(f'#undef {k}\n' for k in (*constants, 'WRITE')))
                cases.append(dict(name=name, key_heads=hk, value_heads=hv, value_dim=dv,
                                  stored_batch=sb, pending=pending, operation=operation))
metal = ROOT / 'artifacts/deferred_gdn_ios_20261008.metal'
metal.write_text(''.join(parts))
air, library = metal.with_suffix('.air'), metal.with_suffix('.metallib')
commands = [['xcrun', '-sdk', 'iphoneos', 'metal', '-fno-fast-math', '-c', str(metal), '-o', str(air)],
            ['xcrun', '-sdk', 'iphoneos', 'metallib', str(air), '-o', str(library)]]
with (ROOT / 'reports/deferred_gdn_ios_build_20261008.log').open('w') as output:
    for command in commands:
        code = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT).returncode
        if code: raise SystemExit(code)
report = dict(all_compiled=True, variants=cases, phone_executed=False, full_swift_integration=False,
              fp32=True, fast_math=False, commands=commands,
              kernel_source_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k, v in values.items()},
              metallib_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
(ROOT / 'reports/deferred_gdn_ios_build_20261008.json').write_text(json.dumps(report, indent=2)+'\n')
print(len(cases), 'iOS shader variants compiled; not executed on iPhone')
