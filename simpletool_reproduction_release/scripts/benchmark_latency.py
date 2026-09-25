#!/usr/bin/env python3
"""Measure complete parallel call latency; one call in flight, no HTTP."""
import argparse
import hashlib
import importlib.util
import importlib.metadata
import os
import json
from pathlib import Path
import statistics
import math
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--version', choices=['v1', 'v2'], default='v2')
    p.add_argument('--repeats', type=int, default=30)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--enforce-eager', action='store_true')
    a = p.parse_args()
    if a.repeats < 1 or a.warmup < 1:
        p.error('repeats and warmup must be positive')
    a.repo = a.repo.resolve()
    a.model = a.model.resolve()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('original_benchmark', a.repo/'01_benchmark.py')
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    b.DIR = a.repo/'prompts'
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    metadata = {'args': {k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
                'torch': torch.__version__, 'vllm': vllm.__version__,
                'versions': {n: importlib.metadata.version(n) for n in ['compressed-tensors','transformers','pydantic']},
                'environment': {n: os.environ.get(n) for n in ['USE_TF','USE_FLAX','OMP_NUM_THREADS','VLLM_WORKER_MULTIPROC_METHOD']},
                'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,power.limit,clocks.max.sm,clocks.max.memory', '--format=csv'], text=True),
                'sha256': {}, 'status': 'initializing'}
    for f in [Path(__file__), a.repo/'01_benchmark.py', *sorted(b.DIR.glob('*')), a.model/'config.json', a.model/'download_provenance.json', a.model/'compatibility.json']:
        if f.is_file(): metadata['sha256'][str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
    def save():
        (a.output/'run.json').write_text(json.dumps(metadata, indent=2)+'\n')
    save()
    start = time.perf_counter()
    llm = LLM(model=str(a.model), trust_remote_code=True, dtype='auto', gpu_memory_utilization=.8,
              max_model_len=4096, max_num_seqs=8, enable_prefix_caching=True, enforce_eager=a.enforce_eager)
    metadata['engine_init_seconds'] = time.perf_counter()-start
    sp = SamplingParams(temperature=0, max_tokens=128, stop=b.STOPS, include_stop_str_in_output=True)
    rows = []
    stream = (a.output/'samples.jsonl').open('w')
    def run(sc, mode, repeat, measured=True):
        heads = b.HEADS[:1+min(b.max_tool_params(sc['tools']),6)]
        base = b.build_prompt(sc, a.version)
        prompts = [base+op for _,op,_ in heads]
        start = time.perf_counter()
        outputs = llm.generate(prompts, sp, use_tqdm=False)
        elapsed = (time.perf_counter()-start)*1000
        row = {'scenario': sc['name'], 'mode': mode, 'repeat': repeat, 'latency_ms': elapsed,
               'head_count': len(heads), 'outputs': {h[0]: {'text': out.outputs[0].text,
               'tokens': len(out.outputs[0].token_ids), 'prompt_tokens': len(out.prompt_token_ids),
               'finish_reason': out.outputs[0].finish_reason} for h,out in zip(heads,outputs)}}
        row['function_correct'] = b.clean(outputs[0].outputs[0].text)==sc['expected']
        if measured:
            rows.append(row); stream.write(json.dumps(row,ensure_ascii=False)+'\n'); stream.flush()
        return row
    scenarios = b.load_scenarios()
    for sc in scenarios:
        run(sc,'first_seen',0)
        for i in range(a.warmup): run(sc,'warmup',i,False)
        for mode in ['warm_identical', 'cache_reset', 'shared_prefix_new_query']:
            for i in range(a.repeats):
                current = dict(sc)
                if mode == 'cache_reset':
                    # vLLM 0.11's V1 wrapper discards the utility return value.
                    # Call the same synchronous utility directly to verify success.
                    core = llm.llm_engine.engine_core
                    reset_ok = core.call_utility('reset_prefix_cache') if hasattr(core, 'call_utility') else llm.reset_prefix_cache()
                    if reset_ok is False: raise RuntimeError('Cache reset failed')
                elif mode == 'shared_prefix_new_query':
                    # Synthetic query changes: expose repeated-input cache advantage explicitly.
                    current['query'] += f'\nObservation identifier: {i:06d}.'
                run(current,mode,i)
            print(sc['name'],mode,'complete',flush=True)
    stream.close()
    summary=[]
    for sc in scenarios:
        for mode in ['first_seen','warm_identical','cache_reset','shared_prefix_new_query']:
            subset=[r for r in rows if r['scenario']==sc['name'] and r['mode']==mode]
            vals=sorted(r['latency_ms'] for r in subset)
            summary.append({'scenario': sc['name'], 'mode':mode,'n':len(vals),'mean_ms':statistics.mean(vals),
                            'p50_ms':statistics.median(vals),'p95_ms':vals[max(0,math.ceil(.95*len(vals))-1)],
                            'function_correct':sum(r['function_correct'] for r in subset),
                            'length_capped_heads':sum(o['finish_reason']=='length' for r in subset for o in r['outputs'].values())})
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    metadata['gpu_after'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem,memory.used', '--format=csv'], text=True)
    metadata['status']='complete'; save()
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
