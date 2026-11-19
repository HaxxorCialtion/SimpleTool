"""Recheck saved actual-tokenizer packets through every Swift execution mode.

Gold packets were produced by make_swift_packet.py using matching int4 serial
inference; no generated command text is executed. GPU runs stay sequential.
"""
import argparse,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('--binary',type=Path,default=ROOT/'artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI')
p.add_argument('--output',type=Path,default=ROOT/'reports/swift_static_release_suite_20261008.json')
a=p.parse_args();rows=[]
for mode in ('dynamic','staticCache','compiled'):
    for size in ('2b','4b'):
        for i in range(5):
            stem=f'swift_static_release_{mode}_{size}_{i}_20261008'
            packet=ROOT/'reports'/f'swift_suite_{size}_{i}_20261008.packet.json'
            out=ROOT/'reports'/f'{stem}.json';env=dict(os.environ,ST_SWIFT_EXECUTION=mode)
            env.pop('ST_SWIFT_TRACE',None);env.pop('ST_SWIFT_BENCHMARK',None)
            with (ROOT/'reports'/f'{stem}.log').open('w') as log:
                code=subprocess.run([str(a.binary.resolve()),str(ROOT/f'artifacts/{size}_int4_compact'),str(packet),str(out)],
                                    env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            data=json.loads(out.read_text()) if out.exists() else {}
            passed=(code==0 and bool(data.get('gold_token_equal')) and
                    all(v for mode_values in data['gold_token_equal'].values() for v in mode_values.values()) and
                    all(v for mode_values in data['decoded_gold_equal'].values() for v in mode_values.values()))
            rows.append(dict(mode=mode,size=size,case=i,passed=passed,exit_code=code,result=str(out.relative_to(ROOT))))
            print(mode,size,i,passed,flush=True)
            a.output.write_text(json.dumps(rows,indent=2)+'\n')
if not all(row['passed'] for row in rows):raise SystemExit(2)
