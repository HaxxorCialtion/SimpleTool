#!/usr/bin/env python3
"""CPU regression against historical predictions; never a new model accuracy run."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simpletool_eval.data import load_pair, read_jsonl, selected_names, sha256
from simpletool_eval.evaluate import score_dataset
from simpletool_eval import protocol, conversion, rules, rules_three_groups
from simpletool_eval.run import prompt_for


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Original benchmarks directory')
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, default=Path('reports/legacy_parity.json'))
    args = p.parse_args()
    base = args.source/'now_benchmark'
    provenance = json.loads(Path('simpletool_eval/provenance.json').read_text())
    # Compare parsed definitions, without importing original GPU inference modules.
    for filename, info in provenance.items():
        original = base/Path(info['source']).name
        if sha256(original) != info['source_sha256']:
            raise AssertionError(f'Original source changed: {original}')
        def definitions(path):
            found = {}
            for node in ast.parse(path.read_text()).body:
                name = node.name if isinstance(node, ast.FunctionDef) else node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None
                if name in info['definitions']:
                    found[name] = ast.dump(node, include_attributes=False)
            return found
        assert definitions(original) == definitions(Path('simpletool_eval')/filename), filename
    original = load_module(base/'051_evaluate_multimodel_final.py')
    report = {'kind': 'historical prediction scorer regression; NOT fresh HF inference',
              'definition_parity': True, 'datasets': {}}
    total = 0
    for name in selected_names():
        ds = Path(name).stem
        path = base/'converted_results_all_models'/f'simpletool_RT-Qwen3-4B_{ds}.jsonl'
        rows, gt, _ = load_pair(args.data, name)
        preds, _ = read_jsonl(path)
        new, _ = score_dataset(rows, gt, preds)
        old = original.evaluate_file(str(path), str(args.data/'possible_answer'/name), {r['id']: r for r in rows})
        assert all(new[k] == v for k, v in old.items()), (ds, old, new)
        report['datasets'][ds] = {'prediction_sha256': sha256(path), 'predictions': len(preds), 'scored': new['total'], 'metrics_identical': True}
        total += len(preds)
    report['predictions_checked'] = total
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'PASS: verbatim source definitions; {total} historical predictions across {len(report["datasets"])} datasets match original scoring')

if __name__ == '__main__':
    main()
