"""Export a completed run or explicitly selected interrupted checkpoint as standalone FP16."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer

from merge_8head import checkpoint_index, lora_pairs, text_key

ROOT = Path(__file__).resolve().parent
TOKENS = [t for h in ['content', 'function', *[f'arg{i}' for i in range(1, 7)]]
          for t in (f'<{h}>', f'</{h}>')] + ['<|null|>']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_identity(run, checkpoint=None):
    if checkpoint is None:
        names = ['training_complete.json', 'run_config.json', 'final/adapter_config.json',
                 'final/adapter_model.safetensors', 'final/head_tokens.safetensors']
    else:
        relative = checkpoint.relative_to(run)
        names = ['INTERRUPTED.json', 'run_config.json'] + [
            str(relative / name) for name in ['train_config.json', 'adapter_config.json',
                                             'adapter_model.safetensors', 'head_tokens.safetensors']]
        names += [str(f.relative_to(run)) for f in sorted((checkpoint / 'tokenizer').iterdir()) if f.is_file()]
    if (run / 'RESUME_RECEIPT.json').exists():
        names.append('RESUME_RECEIPT.json')
    hashes = {name: digest(run / name) for name in names}
    return {'run': str(run), 'sha256': hashes}


def set_dtype(obj):
    if isinstance(obj, dict):
        assert not obj.get('quantization_config'), 'quantized export is not permitted'
        for k, value in obj.items():
            if k in ('dtype', 'torch_dtype'):
                obj[k] = 'float16'
            else:
                set_dtype(value)
    elif isinstance(obj, list):
        for value in obj:
            set_dtype(value)


def export_run(tag, stage='p005', run_dir=None, output=None, merged_dir=None, dry_run=False,
               checkpoint=None):
    torch.set_num_threads(4)
    run = Path(run_dir or ROOT / 'runs' / tag / stage).resolve()
    out = Path(output or ROOT / 'fp16' / stage / tag).resolve()
    src = Path(merged_dir or ROOT / 'merged' / stage / tag).resolve()
    run_config = json.loads((run / 'run_config.json').read_text())
    config = run_config['config']
    selected = None
    checkpoint_validation = None
    if checkpoint is None:
        summary = json.loads((run / 'training_complete.json').read_text())
        assert summary['completed_steps'] == summary['planned_steps'] and summary['skipped_steps'] == 0
        assert not (run / 'INTERRUPTED.json').exists(), 'interrupted run needs explicit --checkpoint interrupted'
        adapter = run / 'final'
    else:
        selected = (run / checkpoint).resolve()
        assert selected == run / 'interrupted', 'only the recorded interrupted checkpoint is supported'
        interrupted = json.loads((run / 'INTERRUPTED.json').read_text())
        assert interrupted['complete'] is False and 0 < interrupted['step'] < interrupted['planned']
        assert not (run / 'training_complete.json').exists(), 'ambiguous completed/interrupted source'
        from resume_state import read_state
        state = read_state(selected, config, interrupted['planned'], run_config['world'])
        assert state['step'] == interrupted['step']
        checkpoint_validation = {
            'valid': True, 'mode': 'explicitly selected interrupted LoRA + 17 tied token rows',
            'completed_steps': state['step'], 'planned_steps': state['planned'],
            'training_complete': False, 'interruption_reason': interrupted['reason'],
            'optimizer_metadata_consistent': True, 'validation': None}
        del state
        summary = {'completed_steps': interrupted['step'], 'planned_steps': interrupted['planned'],
                   'training_complete': False, 'interruption_reason': interrupted['reason'],
                   'checkpoint': str(selected), 'validation': None}
        adapter = selected
    identity = source_identity(run, selected)
    base = Path(config['MODEL_DIR']).resolve()
    if dry_run:
        print(json.dumps({'ready': True, 'base': str(base), 'adapter': str(adapter),
                          'output': str(out), 'dtype': 'float16',
                          'training_complete': selected is None,
                          'completed_steps': summary['completed_steps'],
                          'planned_steps': summary['planned_steps'], 'source_identity': identity}), flush=True)
        return
    if out.exists():
        receipt = json.loads((out / 'EXPORT_VALIDATION.json').read_text())
        assert receipt['valid'] and receipt['source_identity'] == identity, 'existing export has different identity'
        for name, expected in json.loads((out / 'MODEL_SHA256.json').read_text()).items():
            assert digest(out / name) == expected, ('existing export hash mismatch', name)
        print('Already exported and hashes rechecked:', out, flush=True)
        return
    if selected is None:
        subprocess.run([sys.executable, str(ROOT / 'validate_checkpoint.py'), str(run)], check=True)
    else:
        checked = 0
        for file in sorted(adapter.glob('*.safetensors')):
            with safe_open(str(file), framework='pt') as h:
                for key in h.keys():
                    assert torch.isfinite(h.get_tensor(key)).all(), (file.name, key)
                    checked += 1
        checkpoint_validation['checked_tensors'] = checked
    if src.exists():
        info = json.loads((src / 'MERGE_INFO.json').read_text())
        assert Path(info['adapter']).resolve() == adapter and Path(info['base']).resolve() == base
    else:
        subprocess.run([sys.executable, str(ROOT / 'merge_8head.py'), '--base', str(base),
                        '--adapter', str(adapter), '--out', str(src), '--materialize'], check=True)
    tmp = out.with_name(out.name + '.partial')
    tmp.mkdir(parents=True, exist_ok=False)
    pairs, scale = lora_pairs(adapter)
    with safe_open(str(adapter / 'head_tokens.safetensors'), framework='pt') as h:
        ids, rows = h.get_tensor('ids').long(), h.get_tensor('rows')
    assert len(ids) == len(TOKENS) == 17 and len(ids.unique()) == 17
    bmap, smap = checkpoint_index(base), checkpoint_index(src)
    assert bmap == smap
    n = modified = embedding_rows = 0
    dtypes, handles = {}, {}
    for shard in sorted(set(smap.values())):
        exported = {}
        with safe_open(str(src / shard), framework='pt') as merged:
            for key in merged.keys():
                bf = bmap[key]
                if bf not in handles:
                    handles[bf] = safe_open(str(base / bf), framework='pt')
                original = handles[bf].get_tensor(key)
                tk, expected = text_key(key), original
                if tk in pairs:
                    pair = pairs[tk]
                    expected = (original.float() + (pair['B'].float() @ pair['A'].float()) * scale).to(original.dtype)
                    modified += 1
                if tk == 'model.embed_tokens.weight':
                    expected = expected.clone()
                    expected[ids] = rows.to(expected.dtype)
                    embedding_rows += 1
                actual = merged.get_tensor(key)
                assert torch.equal(actual, expected), ('merge mismatch', key)
                t = actual.half() if actual.is_floating_point() else actual
                assert not t.is_floating_point() or torch.isfinite(t).all(), ('non-finite FP16', key)
                exported[key] = t.contiguous()
                n += 1
                dtypes[str(t.dtype)] = dtypes.get(str(t.dtype), 0) + 1
        target = tmp / shard
        save_file(exported, str(target), metadata={'format': 'pt'})
        # Reopen the written checkpoint, not just the in-memory conversion.
        with safe_open(str(target), framework='pt') as written:
            assert set(written.keys()) == set(exported)
            for key, value in exported.items():
                assert torch.equal(written.get_tensor(key), value), ('write mismatch', key)
        del exported
        print(tag, 'verified merged weights and FP16 shard', shard, flush=True)
    assert modified == len(pairs) and embedding_rows == 1
    for f in src.iterdir():
        if f.is_file() and f.suffix != '.safetensors':
            shutil.copy2(f, tmp / f.name)
    cfg = json.loads((tmp / 'config.json').read_text())
    set_dtype(cfg)
    cfg['torch_dtype'] = 'float16'
    (tmp / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    tok = AutoTokenizer.from_pretrained(tmp, local_files_only=True)
    assert [tok.convert_tokens_to_ids(t) for t in TOKENS] == ids.tolist()
    base_cfg = json.loads((base / 'config.json').read_text())
    text_cfg = base_cfg.get('text_config', base_cfg)
    assert text_cfg.get('tie_word_embeddings'), 'tied head-row export contract requires tied embeddings'
    for name in ['run_config.json']:
        shutil.copy2(run / name, tmp / name)
    if selected is None:
        for name in ['training_summary.json', 'CHECKPOINT_VALIDATION.json']:
            shutil.copy2(run / name, tmp / name)
    else:
        (tmp / 'training_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        (tmp / 'CHECKPOINT_VALIDATION.json').write_text(json.dumps(checkpoint_validation, indent=2) + '\n')
        shutil.copy2(run / 'INTERRUPTED.json', tmp / 'INTERRUPTED.json')
        shutil.copy2(adapter / 'train_config.json', tmp / 'checkpoint_train_config.json')
        (tmp / 'EXPORT_NOTES.md').write_text(
            f'# Qwen3.5-{tag} eight-head FP16 export\n\n'
            f'Complete standalone model weights exported from checkpoint step '
            f'{summary["completed_steps"]}/{summary["planned_steps"]}. '
            'The training run stopped at its wallclock limit; the user selected this '
            'saved checkpoint for export without further training. The planned epoch '
            'was not completed. No benchmark evaluation was performed.\n\n'
            'Text LoRA and all 17 trained tied input/output token rows are merged. '
            'The original vision architecture, vision weights and processor are preserved. '
            'All floating tensors are FP16; no separate base or adapter is required.\n')
    if (run / 'RESUME_RECEIPT.json').exists():
        shutil.copy2(run / 'RESUME_RECEIPT.json', tmp / 'RESUME_RECEIPT.json')
    data_receipts = {}
    data = Path(config['DATA_DIR'])
    for name in ['INDEX.json', 'PREFLIGHT.json']:
        f = data / name
        if f.exists():
            data_receipts[name] = {'path': str(f), 'sha256': digest(f)}
    receipt = {'valid': True, 'dtype': 'float16', 'training_base_dtype': 'bfloat16',
               'quantized': False, 'kv_cache_dtype': 'not applicable: weights only, set by evaluator',
               'tensor_count': n, 'dtypes': dtypes, 'verified_lora_weights': modified,
               'verified_head_rows': len(ids), 'base_architecture_and_vision_preserved': True,
               'all_tensors_finite': True, 'written_tensors_reopened_and_verified': True,
               'source_identity': identity, 'data_receipts': data_receipts,
               'completed_steps': summary['completed_steps'], 'planned_steps': summary['planned_steps'],
               'resumed_from_step': summary.get('resumed_from_step', 0),
               'benchmark_evaluation_performed': False}
    if selected is not None:
        receipt.update(training_complete=False, selected_checkpoint=str(selected),
                       interruption_reason=summary['interruption_reason'],
                       user_selected_checkpoint_export=True)
    (tmp / 'EXPORT_VALIDATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (tmp / 'MODEL_SHA256.json').write_text(json.dumps(
        {f.name: digest(f) for f in sorted(tmp.iterdir()) if f.is_file()}, indent=2) + '\n')
    assert source_identity(run, selected) == identity, 'source checkpoint changed during export'
    assert not any(f.is_symlink() for f in tmp.iterdir()), 'export must be standalone'
    tmp.rename(out)
    print(json.dumps({'export_complete': True, 'output': str(out), **receipt}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tag', required=True, choices=['0p8b', '2b', '4b'])
    ap.add_argument('--stage', default='p005')
    ap.add_argument('--run-dir')
    ap.add_argument('--output')
    ap.add_argument('--merged-dir')
    ap.add_argument('--checkpoint', help='explicitly export the recorded interrupted checkpoint: interrupted')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    export_run(a.tag, a.stage, a.run_dir, a.output, a.merged_dir, a.dry_run, a.checkpoint)
