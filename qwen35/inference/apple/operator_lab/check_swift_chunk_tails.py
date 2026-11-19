"""Cap previously validated serial gold to force partial blocks of 3 and 7."""
import json,os,hashlib,subprocess
from pathlib import Path
from tokenizers import Tokenizer
ROOT=Path(__file__).resolve().parent
binary=ROOT/'artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI'
rows=[];report=ROOT/'reports/swift_chunks_tail_suite_20261008.json'
binary_sha=hashlib.sha256(binary.read_bytes()).hexdigest()
for size,count in [('2b',31),('4b',11)]:
    source=ROOT/f'reports/swift_suite_{size}_0_20261008.packet.json';packet=json.loads(source.read_text())
    tokenizer=Tokenizer.from_file(str(ROOT/f'artifacts/{size}_int4_compact/tokenizer.json'))
    assert all(tokenizer.decode(tokens,skip_special_tokens=False)==packet['gold_text'][h] for h,tokens in packet['gold'].items())
    packet['max_tokens']=count;packet['gold']={h:tokens[:count] for h,tokens in packet['gold'].items()}
    packet['gold_text']={h:tokenizer.decode(tokens,skip_special_tokens=False) for h,tokens in packet['gold'].items()}
    packet['python_batch_tokens']={h:tokens[:count] for h,tokens in packet['python_batch_tokens'].items()}
    packet['derived_from']=str(source.relative_to(ROOT));packet['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    packet['gold_method']+='; capped existing validated serial outputs, tokenizer-only text decoding'
    path=ROOT/f'reports/swift_chunks_tail_{size}_packet_20261008.json';path.write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n')
    for policy in ['baseline','both']:
        reference=None
        for chunk in [1,2,4,8]:
            stem=f'swift_chunks_tail_{size}_{policy}_{chunk}_20261008';out=ROOT/f'reports/{stem}.json'
            env=dict(os.environ,ST_SWIFT_EXECUTION='compiled',ST_SWIFT_PROJECTIONS=policy,ST_SWIFT_DECODE_CHUNK=str(chunk))
            for key in ['ST_SWIFT_TRACE','ST_SWIFT_BENCHMARK']:env.pop(key,None)
            with (ROOT/f'reports/{stem}.log').open('w') as log:
                code=subprocess.run([str(binary),str(ROOT/f'artifacts/{size}_int4_compact'),str(path),str(out)],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            d=json.loads(out.read_text()) if out.exists() else {}
            if chunk==1:reference=d
            passed=code==0 and all(set(d.get(field,{}))=={'serial','batch4'} and all(
                set(vals)=={'speech','expression','action','command'} and all(value is True for value in vals.values())
                for vals in d[field].values()) for field in ('gold_token_equal','decoded_gold_equal'))
            passed=passed and all(d.get(field)==reference.get(field) for field in ('outputs','decoded','stop_reasons'))
            rows.append(dict(size=size,policy=policy,chunk=chunk,max_tokens=count,passed=passed,result=str(out.relative_to(ROOT))))
            report.write_text(json.dumps(dict(binary_sha256=binary_sha,cases=rows),indent=2)+'\n')
            print(size,policy,chunk,passed,flush=True)
if not all(r['passed'] for r in rows):raise SystemExit(2)
