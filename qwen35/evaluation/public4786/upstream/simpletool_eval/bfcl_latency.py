"""Single-question BFCL-v3 latency, with concurrent heads only within that question."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

from .conversion import convert_raw_heads_to_fc
from .data import load_pair, selected_names, sha256
from .protocol import build_history_string, get_parameter_order
from .run import prompt_for, model_fingerprint
from .rules import evaluate_single_prediction
from .strict import strict_prediction

BFCL = [Path(n).stem for n in selected_names() if n.startswith('BFCL_')]
MODES = ['sequential', 'cache_reset', 'warm_identical']
STOP_STRINGS = [f'</{h}>' for h in ['content', 'function'] + [f'arg{i}' for i in range(1, 7)]] + ['<|null|>', '<|im_end|>']
V2_SYSTEM = 'You are a real-time function calling assistant. Convert user commands into function calls using the available tools.'


def heads_for(item):
    count = max((len(get_parameter_order(t)) for t in item.get('function', [])), default=0)
    if count > 6:
        raise ValueError('Tool exceeds six argument heads')
    return ['function'] + [f'arg{i}' for i in range(1, count + 1)]


def build_prompt(item, version):
    if len(item.get('question', [])) != 1:
        raise ValueError(f'Not a single-turn row: {item.get("id")}')
    if version == 'v1':
        return prompt_for(item)
    messages = item['question'][0]
    query = next(m['content'] for m in messages if m['role'] == 'user')
    functions = item.get('function', [])
    tools = '\n'.join(json.dumps(t if 'type' in t else {'type': 'function', 'function': t}, ensure_ascii=False) for t in functions)
    history = next((m['history'] for m in messages if m.get('history')), [])
    history_text = ', '.join(build_history_string(history, functions))
    # Public 02_server.py's V2_DEFAULT_SYSTEM and V2 system/user template.
    return (f'<|im_start|>system\n{V2_SYSTEM}\n\n## Available Tools:\n\n{tools}<|im_end|>\n'
            f'<|im_start|>user\nhistory: [{history_text}]\n\n{query}<|im_end|>\n<|im_start|>assistant\n')


def reset_cache(llm):
    # vLLM 0.11 discards the scheduler boolean at EngineCore as well.
    # The synchronous call completes; verify successful resets in engine logs.
    result = llm.reset_prefix_cache()
    if result is False:
        raise RuntimeError('Prefix cache reset explicitly failed')


def timed_question(llm, prompts, sampling):
    # There is exactly one synchronous generate call and no async/thread/executor.
    # Every prompt in this list belongs to the same question.
    start = time.perf_counter_ns()
    outputs = llm.generate(prompts, sampling, use_tqdm=False)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    if len(outputs) != len(prompts) or any(len(o.outputs) != 1 for o in outputs):
        raise RuntimeError('Incomplete head outputs')
    return outputs, elapsed


def summarize(records):
    if not records:
        raise ValueError('Cannot summarize empty measurements')
    vals = sorted(r['latency_ms'] for r in records)
    scored = [r for r in records if r['strict_correct'] is not None]
    return {'count': len(vals), 'mean_ms': statistics.mean(vals), 'p50_ms': statistics.median(vals),
            'p95_ms': vals[math.ceil(.95 * len(vals)) - 1], 'min_ms': vals[0], 'max_ms': vals[-1],
            'serial_hz': 1000 * len(vals) / sum(vals),
            'scored': len(scored), 'missing_gt': len(records) - len(scored),
            'legacy_correct': sum(r['legacy_correct'] for r in scored),
            'strict_correct': sum(r['strict_correct'] for r in scored),
            'function_correct': sum(r['function_correct'] for r in scored),
            'length_limited_calls': sum('length' in r['finish_reasons'].values() for r in records),
            'mean_max_output_tokens': statistics.mean(max(r['token_counts'].values()) for r in records),
            'head_count_distribution': dict(Counter(r['head_count'] for r in records))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1] / 'data')
    p.add_argument('--version', choices=['v1', 'v2'], required=True)
    p.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    p.add_argument('--datasets', nargs='+', choices=BFCL, default=BFCL)
    p.add_argument('--max-model-len', type=int, default=8192)
    p.add_argument('--max-tokens', type=int, default=128)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--limit', type=int, help='Smoke test only, per dataset; no full-suite claim')
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args()
    if len(set(a.modes)) != len(a.modes) or len(set(a.datasets)) != len(a.datasets):
        p.error('Duplicate modes/datasets are not allowed')
    if a.warmup < 1 or a.max_tokens < 1 or a.max_model_len <= a.max_tokens or (a.limit is not None and a.limit < 1):
        p.error('Invalid sizes')
    work = []; coverage = {}; gt_by_dataset = {}
    for name in selected_names(a.datasets):
        ds = Path(name).stem
        rows, gt, _ = load_pair(a.data, name)
        kept = []; excluded = []
        for item in rows:
            max_args = max((len(get_parameter_order(t)) for t in item.get('function', [])), default=0)
            if max_args > 6:
                excluded.append(item['id']); continue
            base = build_prompt(item, a.version)
            heads = heads_for(item)
            kept.append((ds, item, heads, [base + f'<{h}>' for h in heads]))
        coverage[ds] = {'input': len(rows), 'capacity_excluded_ids': excluded,
                        'eligible': len(kept), 'missing_gt_ids': [x[1]['id'] for x in kept if x[1]['id'] not in gt]}
        chosen = kept[:a.limit] if a.limit else kept
        coverage[ds]['measured_ids'] = [x[1]['id'] for x in chosen]
        work.extend(chosen); gt_by_dataset[ds] = gt
    print(f'Preflight: {len(work)} eligible single-turn questions, {len(coverage)} datasets, question concurrency=1', flush=True)
    if a.preflight_only:
        return
    a.output.mkdir(parents=True, exist_ok=False)
    metadata = {'status': 'initializing', 'created_at': datetime.now(timezone.utc).isoformat(),
                'config': {k: str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(a).items()},
                'question_concurrency': 1, 'stop_strings': STOP_STRINGS, 'skip_special_tokens': False, 'head_policy': 'function + max candidate schema arity; no content',
                'data_manifest_sha256': sha256(a.data/'manifest.json'), 'coverage': coverage,
                'model_files_sha256': model_fingerprint(a.model),
                'source_sha256': {f.name: sha256(f) for f in Path(__file__).parent.glob('*.py')},
                'versions': {n: importlib.metadata.version(n) for n in ['vllm','torch','transformers','compressed-tensors','pydantic']},
                'environment': {n: os.environ.get(n) for n in ['USE_TF','USE_FLAX','OMP_NUM_THREADS','VLLM_WORKER_MULTIPROC_METHOD','PYTHONPATH']},
                'gpu': subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version,power.limit,clocks.max.sm,clocks.max.memory','--format=csv'],text=True)}
    def save():
        (a.output/'run.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    save()
    from vllm import LLM, SamplingParams
    start = time.perf_counter()
    llm = LLM(model=str(a.model.resolve()), dtype='auto', max_model_len=a.max_model_len,
              max_num_seqs=8, max_num_batched_tokens=8192, gpu_memory_utilization=.8,
              enable_prefix_caching=True, seed=0)
    metadata['engine_init_seconds'] = time.perf_counter() - start
    # Author-confirmed protocol: null or any defined head closer ends a stream.
    # Keep special tokens visible so added-token null/closers participate in stop matching.
    sp = SamplingParams(temperature=0, seed=0, max_tokens=a.max_tokens,
                        stop=STOP_STRINGS, skip_special_tokens=False)
    tokenizer = llm.get_tokenizer()
    for _, item, _, prompts in work:
        if max(len(tokenizer.encode(text,add_special_tokens=False)) for text in prompts)+a.max_tokens > a.max_model_len:
            raise ValueError(f'Context overflow: {item["id"]}; no truncation allowed')
    # Synthetic warmup is not any benchmark question. Reset cache afterwards.
    synthetic = {'id': 'synthetic-warmup', 'question': [[{'role':'user','content':'Say hello using ping.'}]],
                 'function':[{'name':'ping','parameters':{'type':'object','properties':{'message':{'type':'string'}}}}]}
    prefix = build_prompt(synthetic,a.version)
    for _ in range(a.warmup):
        llm.generate([prefix+'<function>', prefix+'<arg1>'],sp,use_tqdm=False)
    metadata['status']='running'; save()
    all_summary = {}
    for mode in a.modes:
        reset_cache(llm)
        records=[]; phase_start=time.perf_counter()
        with (a.output/f'{mode}.jsonl').open('w') as stream:
            for i,(ds,item,heads,prompts) in enumerate(work):
                if mode == 'cache_reset': reset_cache(llm)
                prime_ms = None
                if mode == 'warm_identical':
                    # Deliberately favorable regime: a complete untimed priming call.
                    # Its cost is retained separately and in phase wall time.
                    _,prime_ms=timed_question(llm,prompts,sp)
                outputs,ms=timed_question(llm,prompts,sp)
                raw={h:o.outputs[0].text for h,o in zip(heads,outputs)}
                calls=convert_raw_heads_to_fc(raw,item.get('function',[]))
                gt=gt_by_dataset[ds].get(item['id'])
                history=any(m.get('history') for m in item['question'][0])
                legacy=evaluate_single_prediction(calls,gt) if gt is not None else None
                record={'dataset':ds,'id':item['id'],'mode':mode,'order':i,'latency_ms':ms,'prime_ms':prime_ms,
                        'head_count':len(heads),'raw_heads':raw,
                        'prompt_tokens':{h:len(o.prompt_token_ids) for h,o in zip(heads,outputs)},
                        'token_counts':{h:len(o.outputs[0].token_ids) for h,o in zip(heads,outputs)},
                        'finish_reasons':{h:o.outputs[0].finish_reason for h,o in zip(heads,outputs)},
                        'stop_reasons':{h:o.outputs[0].stop_reason for h,o in zip(heads,outputs)},
                        'strict_correct':strict_prediction(calls,gt,history_present=history) if gt is not None else None,
                        'legacy_correct':bool(legacy['overall_correct']) if legacy else None,
                        'function_correct':bool(legacy['func_correct']) if legacy else None}
                records.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
                if (i+1)%100==0 or i+1==len(work):print(f'{mode}: {i+1}/{len(work)}',flush=True)
        phase_wall=time.perf_counter()-phase_start
        all_summary[mode]={'overall':summarize(records),
                           'datasets':{ds:summarize([r for r in records if r['dataset']==ds]) for ds in coverage},
                           'phase_wall_seconds':phase_wall,'phase_wall_hz':len(records)/phase_wall,
                           'priming_generate_seconds':sum(r['prime_ms'] or 0 for r in records)/1000}
        (a.output/'summary.json').write_text(json.dumps(all_summary,indent=2)+'\n')
        print(mode,json.dumps(all_summary[mode]['overall']),flush=True)
    metadata.update(status='complete',completed_at=datetime.now(timezone.utc).isoformat())
    metadata['gpu_after']=subprocess.check_output(['nvidia-smi','--query-gpu=name,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used','--format=csv'],text=True)
    save()

if __name__=='__main__': main()
