"""Score predictions using the original local rules, with auditable denominators."""
import argparse
import json
from pathlib import Path

from .conversion import convert_raw_heads_to_fc
from .data import load_pair, read_jsonl, selected_names, sha256
from . import rules, rules_three_groups


def score_dataset(rows, gt, predictions, *, allow_partial=False):
    inputs = {r['id']: r for r in rows}
    predicted = {r['id']: r for r in predictions}
    if len(predicted) != len(predictions):
        raise ValueError('Duplicate prediction IDs')
    unknown = sorted(set(predicted)-set(inputs))
    missing = sorted(set(inputs)-set(predicted))
    if unknown or (missing and not allow_partial):
        raise ValueError(f'Prediction coverage mismatch: unknown={unknown[:5]}, missing={len(missing)}')
    scores = {'input_count': len(rows), 'predicted_count': len(predictions), 'total': 0,
              'func_correct': 0, 'overall_correct': 0, 'filtered': 0,
              'missing_prediction_ids': missing, 'missing_ground_truth_ids': [], 'filtered_ids': [],
              'length_limited_samples': 0}
    details = []
    for item in predictions:
        iid = item['id']
        detail = {'id': iid}
        if iid not in gt:
            scores['missing_ground_truth_ids'].append(iid)
            detail['excluded'] = 'missing_ground_truth'
        elif rules.has_tool_exceeding_max_params(inputs[iid]):
            scores['filtered'] += 1
            scores['filtered_ids'].append(iid)
            detail['excluded'] = 'tool_exceeds_six_parameters'
        else:
            result = convert_raw_heads_to_fc(item['raw_heads'], inputs[iid].get('function', [])) if 'raw_heads' in item else item['result']
            decision = rules.evaluate_single_prediction(result, gt[iid])
            scores['total'] += 1
            for key in ['func_correct', 'overall_correct']:
                scores[key] += int(decision[key])
            detail.update(result=result, **decision)
        if 'length' in item.get('finish_reasons', {}).values():
            scores['length_limited_samples'] += 1
        details.append(detail)
    for key in ['func', 'overall']:
        scores[key+'_acc'] = scores[key+'_correct']/scores['total'] if scores['total'] else 0.0
    return scores, details


def aggregate(results):
    expected = {Path(n).stem for n in selected_names()}
    if set(results) != expected or any(r['missing_prediction_ids'] or not r['total'] for r in results.values()):
        return None
    output = {}
    for key, module in [('five_groups', rules), ('three_groups', rules_three_groups)]:
        metrics = {}
        for metric in ['func_acc', 'overall_acc']:
            groups = {name: module.calculate_group_score(results, name, metric) for name in module.BENCHMARK_GROUPS}
            metrics[metric] = {'groups': groups, 'macro_average': sum(groups.values())/len(groups),
                               'legacy_nonzero_average': module.calculate_final_avg(results, metric)}
        output[key] = metrics
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, default=Path('reports/RT-Qwen3-4B'))
    p.add_argument('--datasets', nargs='+')
    p.add_argument('--allow-partial', action='store_true', help='Smoke/debug only; no full aggregate')
    args = p.parse_args()
    results, artifacts = {}, {}
    args.output.mkdir(parents=True, exist_ok=True)
    for name in selected_names(args.datasets):
        dataset = Path(name).stem
        path = args.predictions/f'{dataset}.jsonl'
        rows, gt, info = load_pair(args.data, name)
        preds, _ = read_jsonl(path)
        scores, details = score_dataset(rows, gt, preds, allow_partial=args.allow_partial)
        results[dataset] = scores
        artifacts[dataset] = {'prediction_sha256': sha256(path), 'input_sha256': info['input_sha256'], 'ground_truth_sha256': info['ground_truth_sha256']}
        (args.output/f'{dataset}.details.jsonl').write_text(''.join(json.dumps(d, ensure_ascii=False)+'\n' for d in details))
    combined = aggregate(results)
    report = {'scorer': 'SimpleTool local legacy rules (not official BFCL execution/AST scoring)',
              'datasets': results, 'aggregates': combined, 'artifacts': artifacts,
              'data_manifest_sha256': sha256(args.data/'manifest.json')}
    if (args.predictions/'run.json').exists():
        report['run'] = json.loads((args.predictions/'run.json').read_text())
    (args.output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    lines = ['# SimpleTool accuracy reproduction', '', report['scorer'], '',
             'Full coverage' if combined else '**Partial evaluation: full-suite aggregate unavailable.**', '',
             '| Dataset | Input | Scored | Filtered | Missing GT | Function | Overall |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, s in results.items():
        lines.append(f'| {name} | {s["input_count"]} | {s["total"]} | {s["filtered"]} | {len(s["missing_ground_truth_ids"])} | {s["func_acc"]:.2%} | {s["overall_acc"]:.2%} |')
    if combined:
        for name, value in combined.items():
            lines += ['', f'## {name}', '']
            for group, score in value['overall_acc']['groups'].items():
                lines.append(f'- {group}: {score:.2%}')
            lines += ['', f'Overall macro average (including zero groups): {value["overall_acc"]["macro_average"]:.2%}',
                      f'Original script nonzero-only average: {value["overall_acc"]["legacy_nonzero_average"]:.2%}']
    (args.output/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
