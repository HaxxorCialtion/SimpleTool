"""Sequential full-model natural-stop parity for explicit projection policies."""
import argparse,json,os,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--execution',choices=('dynamic','compiled'),default='compiled')
p.add_argument('--binary',type=Path,default=ROOT/'artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI')
a=p.parse_args();rows=[]
for policy in ('stream','fused','both'):
    for size in ('2b','4b'):
        for i in range(5):
            stem=f'swift_projection_{a.execution}_{policy}_{size}_{i}_20261008'
            packet=ROOT/'reports'/f'swift_suite_{size}_{i}_20261008.packet.json';out=ROOT/'reports'/f'{stem}.json'
            env=dict(os.environ,ST_SWIFT_EXECUTION=a.execution,ST_SWIFT_PROJECTIONS=policy)
            for key in ('ST_SWIFT_TRACE','ST_SWIFT_BENCHMARK'):env.pop(key,None)
            with (ROOT/'reports'/f'{stem}.log').open('w') as log:
                code=subprocess.run([str(a.binary.resolve()),str(ROOT/f'artifacts/{size}_int4_compact'),str(packet),str(out)],
                                    env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            data=json.loads(out.read_text()) if out.exists() else {}
            passed=code==0 and bool(data.get('gold_token_equal')) and all(
                value for sub in data['gold_token_equal'].values() for value in sub.values()) and all(
                value for sub in data['decoded_gold_equal'].values() for value in sub.values())
            rows.append(dict(policy=policy,execution=a.execution,size=size,case=i,passed=passed,exit_code=code,result=str(out.relative_to(ROOT))))
            print(policy,size,i,passed,flush=True)
            (ROOT/f'reports/swift_projection_{a.execution}_suite_20261008.json').write_text(json.dumps(rows,indent=2)+'\n')
if not all(r['passed'] for r in rows):raise SystemExit(2)
