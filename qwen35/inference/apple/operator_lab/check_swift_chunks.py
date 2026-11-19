"""Sequential real-model checks for GPU autoregressive blocks; commands are data."""
import argparse,json,os,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--kind',choices=('natural','state'),default='natural')
p.add_argument('--policies',nargs='+',choices=('baseline','both'),default=['baseline','both'])
a=p.parse_args();binary=ROOT/'artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI'
binary_sha=hashlib.sha256(binary.read_bytes()).hexdigest();rows=[]
report=ROOT/f'reports/swift_chunks_{a.kind}_suite_20261008.json'
for size in ('2b','4b'):
    for policy in a.policies:
        cases=[(1,0)] if a.kind=='state' else [(chunk,i) for chunk in (1,2,4,8) for i in range(5)]
        for chunk,i in cases:
            stem=f'swift_chunks_{a.kind}_{policy}_{size}_{chunk}_{i}_20261008'
            out=ROOT/f'reports/{stem}.json';packet=ROOT/f'reports/swift_suite_{size}_{i}_20261008.packet.json'
            env=dict(os.environ,ST_SWIFT_EXECUTION='compiled',ST_SWIFT_PROJECTIONS=policy,ST_SWIFT_DECODE_CHUNK=str(chunk))
            for key in ('ST_SWIFT_TRACE','ST_SWIFT_BENCHMARK'):env.pop(key,None)
            if a.kind=='state':env['ST_SWIFT_BENCHMARK']='chunkState'
            with (ROOT/f'reports/{stem}.log').open('w') as log:
                code=subprocess.run([str(binary),str(ROOT/f'artifacts/{size}_int4_compact'),str(packet),str(out)],
                                    env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            data=json.loads(out.read_text()) if out.exists() else {}
            if a.kind=='state':passed=code==0 and data.get('all_passed') is True
            else:
                passed=code==0 and data.get('decode_chunk_size')==chunk and all(
                    set(data.get(field,{}))=={'serial','batch4'} and all(
                        set(values)=={'speech','expression','action','command'} and all(value is True for value in values.values())
                        for values in data[field].values()) for field in ('gold_token_equal','decoded_gold_equal'))
            rows.append(dict(size=size,policy=policy,chunk=chunk,case=i,passed=passed,exit_code=code,result=str(out.relative_to(ROOT))))
            report.write_text(json.dumps(dict(binary_sha256=binary_sha,kind=a.kind,cases=rows),indent=2)+'\n')
            print(size,policy,chunk,i,passed,flush=True)
if not all(row['passed'] for row in rows):raise SystemExit(2)
