"""Verify frozen data and optionally the actual target tokenizer, without GPU inference."""
import argparse
import collections
import json
from pathlib import Path
from benchlib import ROOT, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', help='Optional actual model directory for tokenizer/context preflight')
    p.add_argument('--context', type=int, default=16384)
    a = p.parse_args()
    receipt = json.loads((ROOT/'preflight.json').read_text())
    assert sha(ROOT/'cases.jsonl') == receipt['cases_sha256']
    cases = [json.loads(s) for s in (ROOT/'cases.jsonl').read_text().splitlines()]
    assert len(cases) == len({c['uid'] for c in cases}) == 4786
    assert dict(collections.Counter(c['dataset'] for c in cases)) == receipt['dataset_counts']
    sources = json.loads((ROOT/'source-receipt.json').read_text())
    for rel, digest in sources['files'].items():
        assert sha(ROOT/'upstream'/rel) == digest, rel
    for c in cases:
        assert len(c['ground_truth']) == 1
        maxarity = max(len(t['function']['parameters']['properties']) for t in c['tools'])
        assert maxarity <= 6
        assert c['heads'] == ['content', 'function']+[f'arg{i}' for i in range(1, maxarity+1)]
        for t in c['tools']:
            s = t['function']['parameters']
            req = s.get('required', [])
            assert list(s['properties'])[:len(req)] == req
    result = dict(status='passed', cases=len(cases), cases_sha256=sha(ROOT/'cases.jsonl'),
                  source_hashes_verified=True, target_tokenizer_checked=False)
    if a.model:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
        for head in ['content', 'function']+[f'arg{i}' for i in range(1,7)]:
            for token in (f'<{head}>', f'</{head}>'):
                ids = tok.encode(token, add_special_tokens=False)
                assert len(ids) == 1 and tok.decode(ids, skip_special_tokens=False) == token, token
        longest = max(len(tok.encode(c['prompt'], add_special_tokens=False))+1 for c in cases)
        assert longest+128 <= a.context, (longest, a.context)
        result.update(target_tokenizer_checked=True, max_prompt_tokens=longest, context=a.context,
                      tokenizer_json_sha256=sha(Path(a.model)/'tokenizer.json'))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
