"""Batched vLLM reproduction with exact legacy prompt and head decoding rules."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

from .data import load_pair, read_jsonl, selected_names, sha256
from .protocol import build_multihead_prompt

HEADS = ['content', 'function'] + [f'arg{i}' for i in range(1, 7)]


def prompt_for(item):
    question = item.get('question')
    if not isinstance(question, list) or not question or not isinstance(question[0], list):
        raise ValueError(f'Unexpected question schema: {item["id"]}')
    messages = question[0]
    if not any(m.get('role') == 'user' for m in messages):
        raise ValueError(f'Missing user message: {item["id"]}')
    history = next((m['history'] for m in messages if m.get('history')), [])
    return build_multihead_prompt(messages, item.get('function', []), history)


def model_fingerprint(path):
    paths = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in {'.json', '.safetensors', '.jinja', '.txt'})
    if not any(p.suffix == '.safetensors' for p in paths):
        raise ValueError(f'Missing safetensors weights: {path}')
    print('Fingerprinting model weights...', flush=True)
    return {p.name: sha256(p) for p in paths}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('models/RT-Qwen3-4B'))
    p.add_argument('--data', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, default=Path('results/RT-Qwen3-4B'))
    p.add_argument('--datasets', nargs='+', help='Dataset stems; default: all 18')
    p.add_argument('--limit', type=int, help='Smoke test only; cannot produce full aggregate')
    p.add_argument('--batch-size', type=int, default=128, help='Samples per generate call (8 heads per sample)')
    p.add_argument('--max-tokens', type=int, default=128, help='Per head; matches 01_inference_multimodel.py')
    p.add_argument('--max-model-len', type=int, default=32768)
    p.add_argument('--max-num-seqs', type=int, default=256)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    p.add_argument('--tensor-parallel-size', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--enforce-eager', action='store_true')
    p.add_argument('--disable-cascade-attn', action='store_true', help='Disable the optional cascade attention optimization')
    p.add_argument('--preflight-only', action='store_true', help='Validate data/prompt schema without importing CUDA')
    args = p.parse_args()
    if min(args.batch_size, args.max_tokens, args.max_model_len, args.max_num_seqs, args.tensor_parallel_size) < 1 or (args.limit is not None and args.limit < 1):
        p.error('Sizes and limit must be positive')
    names = selected_names(args.datasets)
    datasets = {}
    for name in names:
        rows, _, _ = load_pair(args.data, name)
        for row in rows:
            prompt_for(row)
        datasets[Path(name).stem] = rows[:args.limit] if args.limit else rows
    print(f'Validated {len(datasets)} datasets, {sum(map(len, datasets.values()))} samples', flush=True)
    if args.preflight_only:
        return
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable; run on a GPU node')
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ['resume', 'preflight_only']}
    config['data_manifest_sha256'] = sha256(args.data/'manifest.json')
    config['model_files_sha256'] = model_fingerprint(args.model)
    config['code_sha256'] = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    config['versions'] = {pkg: importlib.metadata.version(pkg) for pkg in ['vllm', 'torch', 'transformers', 'huggingface-hub']}
    config['gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv,noheader'], text=True).strip()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output/'run.json'
    if manifest_path.exists():
        if not args.resume or json.loads(manifest_path.read_text())['config'] != config:
            raise ValueError('Existing run: use --resume with identical configuration or a fresh output directory')
    else:
        if list(args.output.glob('*.jsonl')):
            raise ValueError('Prediction files exist without run manifest')
        manifest_path.write_text(json.dumps({'status': 'started', 'created_at': datetime.now(timezone.utc).isoformat(), 'config': config}, indent=2)+'\n')
    pending = {}
    for dataset, rows in datasets.items():
        path = args.output/f'{dataset}.jsonl'
        done = {}
        if path.exists():
            done = {r['id']: r for r in read_jsonl(path)[0]}
            if set(done) - {r['id'] for r in rows}:
                raise ValueError(f'Unexpected existing IDs: {dataset}')
            for row in done.values():
                if set(row['raw_heads']) != set(HEADS):
                    raise ValueError(f'Incomplete heads: {dataset}/{row["id"]}')
        pending[dataset] = [r for r in rows if r['id'] not in done]
    if any(pending.values()):
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        stops = [f'</{head}>' for head in HEADS] + ['<|im_end|>', tokenizer.eos_token]
        sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens, stop=list(dict.fromkeys(s for s in stops if s)), seed=args.seed)
        llm = LLM(model=str(args.model.resolve()), dtype='auto', max_model_len=args.max_model_len,
                  max_num_batched_tokens=8192, max_num_seqs=args.max_num_seqs,
                  enable_prefix_caching=True, gpu_memory_utilization=args.gpu_memory_utilization,
                  tensor_parallel_size=args.tensor_parallel_size, seed=args.seed, enforce_eager=args.enforce_eager,
                  disable_cascade_attn=args.disable_cascade_attn)
        for dataset, rows in pending.items():
            for start in range(0, len(rows), args.batch_size):
                batch = rows[start:start+args.batch_size]
                prompts = [prompt_for(row)+f'<{head}>' for row in batch for head in HEADS]
                tokens = tokenizer(prompts, add_special_tokens=False)['input_ids']
                if any(len(t)+args.max_tokens > args.max_model_len for t in tokens):
                    raise ValueError(f'{dataset}: context overflow; increase --max-model-len, no truncation performed')
                outputs = llm.generate(prompts, sampling, use_tqdm=True)
                if len(outputs) != len(prompts):
                    raise RuntimeError('Incomplete vLLM response batch')
                records = []
                for i, row in enumerate(batch):
                    generated = [o.outputs[0] for o in outputs[i*len(HEADS):(i+1)*len(HEADS)]]
                    records.append({'id': row['id'], 'raw_heads': {h: o.text for h, o in zip(HEADS, generated)},
                                    'finish_reasons': {h: o.finish_reason for h, o in zip(HEADS, generated)},
                                    'token_counts': {h: len(o.token_ids) for h, o in zip(HEADS, generated)}})
                with open(args.output/f'{dataset}.jsonl', 'a', encoding='utf-8') as stream:
                    for record in records:
                        stream.write(json.dumps(record, ensure_ascii=False)+'\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                print(f'{dataset}: {min(start+len(batch), len(rows))}/{len(rows)} pending samples complete', flush=True)
    manifest = json.loads(manifest_path.read_text())
    manifest.update(status='complete', completed_at=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')

if __name__ == '__main__':
    main()
