"""Re-score a historical tools-only outputs.jsonl without editing original outputs."""
import argparse
import collections
import json
from pathlib import Path
from benchlib import ROOT, sha, score_row, aggregate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-prefix', action='store_true', help='Smoke only; never label as full benchmark')
    a = p.parse_args()
    assert not a.output.exists(), 'Choose a new destination; do not overwrite old reports'
    assert sha(ROOT/'cases.jsonl') == json.loads((ROOT/'preflight.json').read_text())['cases_sha256']
    cases = [json.loads(x) for x in (ROOT/'cases.jsonl').read_text().splitlines()]
    raw = [json.loads(x) for x in a.outputs.read_text().splitlines()]
    assert len(raw) == len(cases) or (a.allow_prefix and 0 < len(raw) < len(cases))
    assert [r['uid'] for r in raw] == [c['uid'] for c in cases[:len(raw)]]
    for r, c in zip(raw, cases):
        assert [h['name'] for h in r['heads']] == [h for h in c['heads'] if h != 'content']
    rows = [score_row(r,c) for r,c in zip(raw,cases)]
    datasets = {n: aggregate([r for r in rows if r['dataset'] == n]) for n in dict.fromkeys(r['dataset'] for r in rows)}
    stops = dict(collections.Counter(h['stop'] for r in raw for h in r['heads']))
    result = dict(complete=len(raw)==len(cases), all=aggregate(rows), datasets=datasets,
                  stops=stops, cases_sha256=sha(ROOT/'cases.jsonl'), outputs_sha256=sha(a.outputs),
                  adapter_sha256=sha(ROOT/'benchlib.py'), scored_rows=rows)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['complete','all','stops']},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
