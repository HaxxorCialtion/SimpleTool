"""Sequential owned Python-gold / native Swift real-input parity suite."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=[]
for size in ('2b','4b'):
    prompts=[x['prompt'] for x in json.loads((ROOT/f'reports/{size}_stream_swiglu_cases_20261008.json').read_text())]
    for i,prompt in enumerate(prompts):
        base=ROOT/f'reports/swift_suite_{size}_{i}_20261008'
        packet=base.with_suffix('.packet.json');result=base.with_suffix('.result.json')
        commands=[
            [sys.executable,str(ROOT/'make_swift_packet.py'),'--model',str(ROOT/f'artifacts/{size}_int4_compact'),
             '--output',str(packet),'--tokens','64','--natural-stop','--question',prompt],
            [str(ROOT/'artifacts/swift_build/Build/Products/Debug/SimpleToolSwiftCLI'),
             str(ROOT/f'artifacts/{size}_int4_compact'),str(packet),str(result)]]
        codes=[]
        for phase,command in zip(('gold','native'),commands):
            with base.with_suffix(f'.{phase}.log').open('w') as log:
                codes.append(subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode)
            if codes[-1]:break
        row={'size':size,'prompt':prompt,'exit_codes':codes,'result':str(result.relative_to(ROOT))}
        if codes==[0,0]:
            data=json.loads(result.read_text());gold=json.loads(packet.read_text())
            row['token_equal']=data['gold_token_equal'];row['decoder_equal']=data['decoded_gold_equal']
            row['python_batch_gold_equal']=gold['python_batch_gold_equal']
            row['batch_trace']=data['active_batch_by_step']['batch4']
            row['passed']=all(v for mode in row['token_equal'].values() for v in mode.values()) and all(
                v for mode in row['decoder_equal'].values() for v in mode.values())
        else:row['passed']=False
        rows.append(row);print(size,i,row['passed'],flush=True)
        (ROOT/'reports/swift_real_suite_20261008.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
if not all(row['passed'] for row in rows):raise SystemExit(2)
