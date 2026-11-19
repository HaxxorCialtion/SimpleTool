"""Actual naturally stopping short heads; private benchmark, no engine changes."""
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
from simpletool.loading import load_model
from simpletool.decode import HEADS, prompt_ids, prefill, batched_generate
from simpletool.compiled import CompiledStep
from simpletool.stream_grouped_qmv import install_stream_grouped_qmv
from simpletool.fused_swiglu_int4 import install_fused_swiglu_int4

ROOT = Path(__file__).resolve().parent

def main():
    mx.set_cache_limit(256 * 1024 * 1024)
    model, tok = load_model(ROOT/'artifacts/4b_int4_compact', kernel='custom', compact_metadata=True)
    install_stream_grouped_qmv(model)
    install_fused_swiglu_int4(model)
    step = CompiledStep(model)
    records = []
    for prompt in ('取消刚才的操作，不要执行任何命令', '请把明天上午九点设置成提醒'):
        ids = prompt_ids(tok, [{'role':'user', 'content':prompt}])
        cache, prefill_s = prefill(model, ids)
        def run(heads):
            start = time.perf_counter()
            result = batched_generate(model,tok,ids,32,heads=heads,prefix_cache=cache,compiled_step=step)
            return result, time.perf_counter()-start
        reference, _ = run(HEADS)
        singles = {h:run((h,))[0] for h in HEADS}
        equality = {h:reference['channels'][h]['token_ids']==singles[h]['channels'][h]['token_ids'] for h in HEADS}
        assert all(equality.values()), equality
        samples=[]
        for round_no in range(5):
            groups=[HEADS]+[(h,) for h in HEADS]
            groups=groups[round_no:]+groups[:round_no]
            row={}
            for heads in groups:
                result, wall=run(heads)
                assert all(result['channels'][h]['token_ids']==reference['channels'][h]['token_ids'] for h in heads)
                row['four' if len(heads)==4 else heads[0]]=dict(wall_s=wall,stats=result['stats'])
            samples.append(row)
        medians={key:statistics.median(r[key]['wall_s'] for r in samples) for key in ('four',*HEADS)}
        rec=dict(prompt=prompt,one_prefill_s=prefill_s,channels=reference['channels'],
                 lengths={h:len(c['token_ids']) for h,c in reference['channels'].items()},
                 active_batch_by_step=reference['stats']['active_batch_by_step'],
                 token_equal_same_precision=equality,median_warm_generation_s=medians,samples=samples,
                 serial_generation_speedup=sum(medians[h] for h in HEADS)/medians['four'],
                 longest_serial_head_ratio=max(medians[h] for h in HEADS)/medians['four'])
        records.append(rec)
        print(json.dumps({k:v for k,v in rec.items() if k not in ('channels','samples')},ensure_ascii=False),flush=True)
    report=dict(model='4B compact int4',compute='FP32',natural_stops=True,max_tokens=32,rounds=5,
                timer='Warm prefix generation function including fork, marker, decode and final rendering; prefill and tokenizer excluded',
                cache_limit_mib=256,competing_workloads_uncontrolled=True,records=records,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'reports/4b_short_unequal_20261008.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
