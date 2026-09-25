#!/usr/bin/env python3
"""Validate complete BFCL latency outputs and produce a report without a GPU."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simpletool_eval.bfcl_latency import summarize, heads_for
from simpletool_eval.conversion import convert_raw_heads_to_fc
from simpletool_eval.data import load_pair, selected_names, sha256
from simpletool_eval.rules import evaluate_single_prediction
from simpletool_eval.strict import strict_prediction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    a=p.parse_args()
    meta=json.loads((a.run/'run.json').read_text())
    if meta['status']!='complete':raise ValueError('Run is not complete')
    if sha256(a.data/'manifest.json')!=meta['data_manifest_sha256']:raise ValueError('Data manifest differs')
    entries={}
    for name in selected_names(meta['config']['datasets']):
        ds=Path(name).stem;rows,gt,_=load_pair(a.data,name)
        for row in rows:entries[ds,row['id']]=(row,gt.get(row['id']))
    order=[(ds,iid) for ds,c in meta['coverage'].items() for iid in c['measured_ids']]
    summary=json.loads((a.run/'summary.json').read_text())
    for mode in meta['config']['modes']:
        rows=[json.loads(l) for l in (a.run/f'{mode}.jsonl').read_text().splitlines()]
        keys=[(r['dataset'],r['id']) for r in rows]
        if keys!=order or len(set(keys))!=len(keys):raise ValueError('Incomplete/reordered/duplicate IDs')
        for i,r in enumerate(rows):
            if r['order']!=i or r['mode']!=mode or not math.isfinite(r['latency_ms']) or r['latency_ms']<=0:raise ValueError('Invalid timing record')
            row,gt=entries[r['dataset'],r['id']];heads=heads_for(row)
            for key in ['raw_heads','token_counts','prompt_tokens','finish_reasons','stop_reasons']:
                if set(r[key])!=set(heads):raise ValueError('Missing/unexpected head')
            if r['head_count']!=len(heads):raise ValueError('Head count differs')
            calls=convert_raw_heads_to_fc(r['raw_heads'],row['function'])
            if gt is None:
                if any(r[k] is not None for k in ['strict_correct','legacy_correct','function_correct']):raise ValueError('Unscorable row has a score')
            else:
                legacy=evaluate_single_prediction(calls,gt)
                strict=strict_prediction(calls,gt,history_present=any(m.get('history') for m in row['question'][0]))
                if r['strict_correct']!=strict or r['legacy_correct']!=bool(legacy['overall_correct']) or r['function_correct']!=bool(legacy['func_correct']):raise ValueError('Scores differ on replay')
        def normalized(x):return json.loads(json.dumps(x))
        if normalized(summarize(rows))!=summary[mode]['overall']:raise ValueError('Overall summary differs')
        for ds in meta['coverage']:
            if normalized(summarize([r for r in rows if r['dataset']==ds]))!=summary[mode]['datasets'][ds]:raise ValueError('Dataset summary differs')
    lines=['# BFCL-v3 single-question latency results','',
           '**Paper context:** the author states the reported 16 Hz was measured directly on BFCL-v3. The earlier game/arm/avatar measurements are separate supplementary tests. The present public-checkpoint run is reported as observed, not fitted to 16 Hz.','',
           f"Checkpoint: `{Path(meta['config']['model']).name}`, prompt `{meta['config']['version']}`. See `run.json` for full model, data and source hashes.",
           '', 'Stops: `<|null|>` or any defined head closing tag, plus ChatML end/EOS. Per-head stop reasons are retained. Special-token skipping is disabled during matching.',
           '','```text',meta['gpu'].strip(),'```','',
           f"Engine initialization: {meta['engine_init_seconds']:.2f} s (outside request timing). Question concurrency: **1**. Head policy: {meta['head_policy']}.",
           '',f"Dataset input: {sum(c['input'] for c in meta['coverage'].values())}; capacity exclusions: {sum(len(c['capacity_excluded_ids']) for c in meta['coverage'].values())}; timed questions per mode: {len(order)}. Input order and excluded IDs are in `run.json`.",'',
           '| Mode | Calls | Mean ms | p50 ms | p95 ms | Serial Hz | Phase wall Hz | Capped calls |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode,v in summary.items():
        r=v['overall'];lines.append(f"| {mode} | {r['count']} | {r['mean_ms']:.2f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['serial_hz']:.2f} | {v['phase_wall_hz']:.2f} | {r['length_limited_calls']} |")
    lines+=['','`serial_hz` = calls / summed generation seconds, not mean of per-call Hz. `phase_wall_hz` includes scoring/writing/reset overhead and all priming in warm-identical mode. Engine initialization and synthetic warmup are outside both rates.','',
            'Sequential is a first traversal without individual benchmark priming. Cache-reset removes cross-request prefix reuse. Warm-identical measures the second of two identical requests and excludes the first generation from its latency; it is a favorable diagnostic, not first-time query throughput.','',
            '| Mode | Dataset | N | Mean ms | p50 ms | p95 ms | Hz | Local legacy | Local strict |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for mode,v in summary.items():
        for ds,r in v['datasets'].items():
            lines.append(f"| {mode} | {ds} | {r['count']} | {r['mean_ms']:.2f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['serial_hz']:.2f} | {r['legacy_correct']}/{r['scored']} | {r['strict_correct']}/{r['scored']} |")
    lines+=['','Incorrect, unscorable and capped predictions remain in timing aggregates. Local scoring is not official BFCL AST/execution scoring. This covers six recovered single-turn subsets, not every BFCL-v3 category. Quantized v1 latency-run accuracy is distinct from the previously published non-quantized accuracy run.','',
            'See [BFCL_SPEED_PROTOCOL.md](../../../BFCL_SPEED_PROTOCOL.md) for cache, head-count, stop, environment and measurement contracts. The 48 GB RTX 4090 differs from a retail 24 GB card. No HTTP overhead, tool execution or matched autoregressive baseline is measured here.','',
            '## Validation','',
            'All expected IDs occur once per mode in the recorded order. Head sets and positive finite timings were checked. Per-question legacy/strict correctness was replayed from raw outputs, and all reported aggregates were recomputed.']
    (a.run/'summary.md').write_text('\n'.join(lines)+'\n')
    (a.run/'validation.json').write_text(json.dumps({'question_count':len(order),'modes':meta['config']['modes'],'id_and_head_coverage_verified':True,'finite_positive_timings':True,'scores_replayed':True,'summaries_recomputed':True},indent=2)+'\n')
    print('\n'.join(lines[:25]))

if __name__=='__main__':main()
