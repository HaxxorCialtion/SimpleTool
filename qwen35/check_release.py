"""CPU-only source, protocol and frozen-data acceptance. No model execution."""
import ast,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(args,cwd):
 print('\nCHECK',cwd.relative_to(ROOT),*args,flush=True)
 subprocess.run([sys.executable,*args],cwd=cwd,check=True)

def main():
 for directory in ['reference','training','inference/cuda']:
  for p in (ROOT/directory).rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
 models=json.loads((ROOT/'models/registry.json').read_text())['models']
 assert len(models)==11 and all(m['public'] for m in models)
 assert sum(m['weight_shards'] for m in models)==191
 run(['-m','unittest','discover','-s','tests','-v'],ROOT)
 run(['-m','unittest','test_protocol','test_runtime_content','test_server','-v'],ROOT/'inference/cuda')
 run(['check_frozen.py'],ROOT/'evaluation/public4786')
 run(['-m','unittest','test_adapter','-v'],ROOT/'evaluation/public4786')
 run(['-m','unittest','discover','-s','tests','-v'],ROOT/'evaluation/public4786/upstream')
 for name in ['cf-zh','cf-en']:
  run(['-m','unittest','test_benchmark','-v'],ROOT/'evaluation/diagnostic800/frozen'/name/'scripts')
 for name in ['hard','multiarg']:
  code="import json; from pathlib import Path; from benchmark_lib import score_case,validate_rows; rows=[json.loads(x) for x in Path('../cases.jsonl').read_text().splitlines()]; validate_rows(rows); assert len(rows)==200; assert all(score_case(r['expected'],r)['exact_correct'] for r in rows); print('200 literal oracles passed')"
  run(['-c',code],ROOT/'evaluation/diagnostic800/frozen'/name/'scripts')
 print('\nRelease CPU acceptance passed. GPU/iPhone inference not rerun.',flush=True)
if __name__=='__main__':main()
