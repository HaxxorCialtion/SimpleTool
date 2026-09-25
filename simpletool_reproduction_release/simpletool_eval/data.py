"""Strict inputs with explicit, fingerprinted compatibility for legacy JSONL errors."""
import hashlib
import json
from pathlib import Path

from .protocol import DATASETS_TO_EVAL
from .rules import has_tool_exceeding_max_params


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path, *, allow_invalid=False):
    rows, errors = [], []
    with open(path, encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if not allow_invalid:
                    raise ValueError(f'{path}:{number}: {exc}') from exc
                errors.append({'line': number, 'error': str(exc)})
                continue
            if not isinstance(row, dict) or not isinstance(row.get('id'), str):
                raise ValueError(f'{path}:{number}: expected object with string id')
            rows.append(row)
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{path}: duplicate IDs')
    return rows, errors


def describe_pair(base, name):
    base = Path(base)
    inputs, errors = read_jsonl(base / name)
    gt, gt_errors = read_jsonl(base / 'possible_answer' / name, allow_invalid=True)
    ids, gt_ids = {r['id'] for r in inputs}, {r['id'] for r in gt}
    filtered = [r['id'] for r in inputs if r['id'] in gt_ids and has_tool_exceeding_max_params(r)]
    return {
        'filename': name,
        'input_sha256': sha256(base / name),
        'ground_truth_sha256': sha256(base / 'possible_answer' / name),
        'input_count': len(inputs), 'ground_truth_count': len(gt),
        'invalid_input_lines': errors, 'invalid_ground_truth_lines': gt_errors,
        'missing_ground_truth_ids': sorted(ids - gt_ids),
        'extra_ground_truth_ids': sorted(gt_ids - ids),
        'filtered_ids': filtered,
        'scored_count': len(ids & gt_ids) - len(filtered),
    }


def load_pair(base, name):
    base = Path(base)
    manifest = json.loads((base / 'manifest.json').read_text())
    info = manifest['datasets'][Path(name).stem]
    for relative, key in [(name, 'input_sha256'), ('possible_answer/' + name, 'ground_truth_sha256')]:
        if sha256(base / relative) != info[key]:
            raise ValueError(f'Dataset fingerprint mismatch: {relative}')
    inputs, _ = read_jsonl(base / name)
    gt, errors = read_jsonl(base / 'possible_answer' / name, allow_invalid=True)
    if errors != info['invalid_ground_truth_lines']:
        raise ValueError(f'Unrecorded invalid ground truth: {name}')
    return inputs, {r['id']: r['ground_truth'] if isinstance(r['ground_truth'], list) else [r['ground_truth']] for r in gt}, info


def selected_names(names=None):
    if not names:
        return DATASETS_TO_EVAL
    lookup = {Path(n).stem: n for n in DATASETS_TO_EVAL}
    return [lookup[name] for name in names]
