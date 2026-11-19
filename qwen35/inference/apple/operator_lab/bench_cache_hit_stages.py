"""Actual 4B inference, incremental cache hits, unequal external stops, paired wall timing.

No random token IDs or estimated kernel durations. EOS is masked only to realize
the requested random lengths. This exercises cancellation, not natural EOS quality.
Detailed mode materializes compaction separately; paired ordinary mode measures
the actual engine without those extra synchronizations.
"""
import argparse
import csv
import hashlib
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache
from simpletool.loading import load_model
from simpletool.decode import HEADS, prompt_ids, prefill, batched_generate, render
from simpletool.cache import clone_single, cache_arrays, fork_batch, memory_accounting
from simpletool.compiled import CompiledStep
from simpletool.stream_grouped_qmv import install_stream_grouped_qmv
from simpletool.fused_swiglu_int4 import install_fused_swiglu_int4
from simpletool.vision import set_positions

ROOT=Path(__file__).resolve().parent
STEM='cache_hit_stages_4b_20261008'
clock=time.perf_counter

class Ledger:
    def __init__(self):self.start=clock();self.events=[]
    def call(self,label,fn,**meta):
        started=clock();value=fn();ended=clock()
        self.events.append(dict(label=label,start_s=started-self.start,wall_s=ended-started,**meta))
        return value

def cache_digest(cache):
    mx.eval(cache_arrays(cache));h=hashlib.sha256()
    for a in cache_arrays(cache):
        h.update(str((a.shape,a.dtype)).encode());h.update(np.asarray(a).tobytes())
    for c in cache:
        if isinstance(c,KVCache):h.update(str(c.offset).encode())
    return h.hexdigest()

def make_inputs(tok):
    records='\n'.join(f'记录{i}：周{i%7+1}上午安排喝水提醒，下午检查日历。用户可以取消提醒、修改时间、调整音量和亮度。上一轮已经确认设备设置，请只处理最新请求。' for i in range(120))
    body=tok.encode(records,add_special_tokens=False)
    def make(n):
        def messages(k):
            return [{'role':'system','content':'你是SimpleLove手机助手。根据最后的请求回答，四个通道各自简短输出，历史记录只作为上下文。'},
                    {'role':'user','content':tok.decode(body[:k],skip_special_tokens=False)+'\n最新请求：请把明天上午九点设置成喝水提醒。'}]
        k=n-len(prompt_ids(tok,messages(0)))
        for _ in range(12):
            m=messages(k);ids=prompt_ids(tok,m);delta=n-len(ids)
            if delta==0:return dict(messages=m,ids=ids)
            k+=delta
        for candidate in range(max(1,k-24),k+25):
            m=messages(candidate);ids=prompt_ids(tok,m)
            if len(ids)==n:return dict(messages=m,ids=ids)
        raise RuntimeError(f'Cannot construct exactly {n} real template tokens')
    inputs={n:make(n) for n in (768,1024,1280,1536)}
    for hit in (512,768,1024):
        assert inputs[hit+256]['ids'][:hit]==inputs[hit+512]['ids'][:hit]
    return inputs

def ordinary(model,tok,compiled,cached,inp,hit,upper,limits):
    started=clock();ids=prompt_ids(tok,inp['messages'])
    assert ids==inp['ids'] and ids[:hit]==cached['ids']
    full,pp=prefill(model,ids,prefix_cache=cached['cache'],start_offset=hit)
    result=batched_generate(model,tok,ids,upper,prefix_cache=full,compiled_step=compiled,
        suppress_tokens=compiled.suppress_tokens,ignore_stops=True,head_token_limits=limits)
    json.dumps(result,ensure_ascii=False)
    return result,clock()-started,pp

def detailed(model,tok,compiled,cached,inp,hit,upper,limits):
    log=Ledger()
    ids=log.call('tokenize_template',lambda:prompt_ids(tok,inp['messages']))
    def lookup():
        if ids[:hit]!=cached['ids']:raise AssertionError('prefix is not a real cache hit')
        return cached['cache']
    seed=log.call('cache_lookup_validate',lookup)
    full=log.call('incremental_cache_clone',lambda:clone_single(seed))
    set_positions(model)
    for offset in range(hit,len(ids),256):
        def chunk(offset=offset):
            hidden=model.model(mx.array(ids[offset:offset+256])[None],cache=full)
            mx.eval(hidden,*cache_arrays(full))
        log.call('prefill_chunk',chunk,input_tokens=min(256,len(ids)-offset),offset=offset)
    markers=log.call('marker_tokenize',lambda:[tok.encode(f'<{h}>',add_special_tokens=False) for h in HEADS])
    assert len(set(map(len,markers)))==1
    caches=log.call('fork_handles_and_gdn_graph',lambda:fork_batch(full,4))
    states=log.call('fork_materialize_static_suffix',lambda:compiled.initialize(caches,4,upper+len(markers[0])))
    raw=[[] for _ in HEADS];active=list(range(4));used=0;steps=[];finish={}
    def forward(current,step_no,kind):
        nonlocal states,used
        output=log.call('model_graph_call',lambda:compiled.forward(current,states,mx.array(used,mx.int32)),step=step_no,active=len(active),input_kind=kind)
        prediction,states=output
        log.call('model_eval_wait',lambda:mx.eval(prediction,states),step=step_no,active=len(active),input_kind=kind)
        used+=1
        return prediction
    for position in range(len(markers[0])-1):
        forward(mx.array([[m[position]] for m in markers]),0,'marker_hidden')
    current=mx.array([[m[-1]] for m in markers]);kind='final_marker_first_token'
    for number in range(1,max(limits.values())+1):
        step_start=clock();event_start=len(log.events);prior=[HEADS[i] for i in active]
        prediction=forward(current,number,kind)
        chosen=log.call('token_readback',prediction.tolist,step=number,active=len(active))
        if number==1:first_token_s=clock()-log.start
        def stopping():
            keep=[];survivors=[];finished=[]
            for row,(idx,token) in enumerate(zip(active,chosen)):
                raw[idx].append(token)
                if len(raw[idx])>=limits[HEADS[idx]]:
                    finished.append(HEADS[idx]);finish[HEADS[idx]]=clock()-log.start
                else:keep.append(row);survivors.append(idx)
            return keep,survivors,finished
        keep,survivors,finished=log.call('stop_controller',stopping,step=number,active=len(active))
        if keep and len(keep)!=len(active):
            states=log.call('batch_compaction',lambda:compiled.filter(states,keep),step=number,active_before=len(active),active_after=len(keep))
            log.call('batch_compaction_eval_wait',lambda:mx.eval(states),step=number)
        row=dict(step=number,active_heads=prior,finished_heads=finished,step_wall_s=clock()-step_start,
                 elapsed_s=clock()-log.start,events=log.events[event_start:])
        steps.append(row)
        if not keep:break
        active=survivors;current=mx.array([[chosen[row]] for row in keep]);kind='previous_generated_token'
    channels=log.call('output_text_decode',lambda:{h:render(tok,r,h,'external_token_limit') for h,r in zip(HEADS,raw)})
    payload=dict(channels=channels,target_lengths=limits,precision='W4A32',cache_hit_tokens=hit,new_prefill_tokens=len(ids)-hit)
    log.call('json_serialize',lambda:json.dumps(payload,ensure_ascii=False))
    wall=clock()-log.start
    sums=defaultdict(float)
    for event in log.events:sums[event['label']]+=event['wall_s']
    return dict(channels=channels,request_wall_s=wall,stage_s=dict(sums),unattributed_controller_s=wall-sum(sums.values()),
                first_token_ready_s=first_token_s,head_finished_s=finish,steps=steps,events=log.events)

def median(v):return statistics.median(v)

def summarize(records):
    groups=defaultdict(list)
    for r in records:groups[(r['cache_hit_tokens'],r['new_prefill_tokens'],r['interval'],r['profile'])].append(r)
    profile_summaries=[]
    for key,rows in groups.items():
        values=[r['ordinary_wall_s'] for r in rows];center=median(values);mad=median([abs(v-center) for v in values])
        threshold=max(3*1.4826*mad,.02*center)
        for r in rows:r['outlier']=abs(r['ordinary_wall_s']-center)>threshold
        kept=[r for r in rows if not r['outlier']]
        assert len(kept)>=len(rows)//2+1
        stage_keys=set().union(*(r['detailed']['stage_s'] for r in kept))
        profile_summaries.append(dict(cache_hit_tokens=key[0],new_prefill_tokens=key[1],interval=key[2],profile=key[3],
            target_lengths=rows[0]['target_lengths'],samples=len(rows),kept=len(kept),rejected=[r['sample_id'] for r in rows if r['outlier']],
            ordinary_wall_median_s=median([r['ordinary_wall_s'] for r in kept]),all_wall_median_s=center,
            detailed_wall_median_s=median([r['detailed']['request_wall_s'] for r in kept]),
            stage_median_s={k:median([r['detailed']['stage_s'].get(k,0) for r in kept]) for k in stage_keys},
            timing_overhead_median_s=median([r['detailed']['request_wall_s']-r['ordinary_wall_s'] for r in kept])))
    cells=[]
    for hit in (512,768,1024):
        for new in (256,512):
            for interval in ('3-12','12-24'):
                profiles=[p for p in profile_summaries if (p['cache_hit_tokens'],p['new_prefill_tokens'],p['interval'])==(hit,new,interval)]
                rows=[r for r in records if (r['cache_hit_tokens'],r['new_prefill_tokens'],r['interval'])==(hit,new,interval) and not r['outlier']]
                center=median([p['ordinary_wall_median_s'] for p in profiles])
                representative=min(rows,key=lambda r:abs(r['ordinary_wall_s']-center))
                keys=set().union(*(p['stage_median_s'] for p in profiles))
                cells.append(dict(cache_hit_tokens=hit,new_prefill_tokens=new,interval=interval,samples=21,kept=len(rows),
                    ordinary_wall_median_of_profile_medians_s=center,detailed_wall_median_of_profile_medians_s=median([p['detailed_wall_median_s'] for p in profiles]),
                    stage_median_of_profile_medians_s={k:median([p['stage_median_s'].get(k,0) for p in profiles]) for k in keys},
                    ordinary_wall_min_max_s=[min(r['ordinary_wall_s'] for r in rows),max(r['ordinary_wall_s'] for r in rows)],
                    representative_sample_id=representative['sample_id'],representative=representative))
    return profile_summaries,cells

def write_tables(records,cells):
    flat=[];step_rows=[]
    for r in records:
        base={k:r[k] for k in ('sample_id','cache_hit_tokens','new_prefill_tokens','interval','profile','round','ordinary_wall_s','ordinary_prefill_s','outlier')}
        base['target_lengths']=json.dumps(r['target_lengths']);base['detailed_wall_s']=r['detailed']['request_wall_s']
        flat.append({**base,**{k+'_s':v for k,v in r['detailed']['stage_s'].items()},'unattributed_controller_s':r['detailed']['unattributed_controller_s']})
        for s in r['detailed']['steps']:
            costs=defaultdict(float)
            for e in s['events']:costs[e['label']]+=e['wall_s']
            step_rows.append({**{k:base[k] for k in ('sample_id','cache_hit_tokens','new_prefill_tokens','interval','profile','round','outlier')},
                'step':s['step'],'active_heads':','.join(s['active_heads']),'finished_heads':','.join(s['finished_heads']),
                'step_wall_s':s['step_wall_s'],'elapsed_s':s['elapsed_s'],**{k+'_s':v for k,v in costs.items()}})
    for suffix,rows in [('requests',flat),('steps',step_rows)]:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (ROOT/'reports'/f'{STEM}_{suffix}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    keys=['cache_hit_tokens','new_prefill_tokens','interval','samples','kept','ordinary_wall_median_of_profile_medians_s','detailed_wall_median_of_profile_medians_s']
    stage_keys=sorted(set().union(*(c['stage_median_of_profile_medians_s'] for c in cells)))
    rows=[{**{k:c[k] for k in keys},**{k+'_s':c['stage_median_of_profile_medians_s'].get(k,0) for k in stage_keys}} for c in cells]
    with (ROOT/'reports'/f'{STEM}_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--rounds',type=int,default=7);args=p.parse_args()
    if args.rounds!=7:raise ValueError('This receipt specifies seven measured repeats per profile')
    mx.set_cache_limit(256*1024*1024)
    start=clock();model,tok=load_model(ROOT/'artifacts/4b_int4_compact',kernel='custom',compact_metadata=True)
    install_stream_grouped_qmv(model);install_fused_swiglu_int4(model);load_s=clock()-start
    banned=tuple(tok.convert_tokens_to_ids(t) for t in ('<|endoftext|>','<|im_end|>'))
    assert all(type(t) is int for t in banned)
    compiled=CompiledStep(model,banned)
    inputs=make_inputs(tok);rng=random.Random(20261008)
    profiles={name:[dict(zip(HEADS,rng.sample(range(lo,hi+1),4))) for _ in range(3)] for name,lo,hi in [('3-12',3,12),('12-24',12,24)]}
    setup=dict(model_load_install_s=load_s,cached_prefix_creation={},profiles=profiles)
    cache={}
    for hit in (512,768,1024):
        ids=inputs[hit+256]['ids'][:hit];c,t=prefill(model,ids)
        cache[hit]=dict(ids=ids,cache=c,sha256=cache_digest(c))
        setup['cached_prefix_creation'][hit]=t
        print('cached prefix created',hit,round(t,3),flush=True)
    (ROOT/'reports'/f'{STEM}_inputs.json').write_text(json.dumps(dict(inputs=inputs,profiles=profiles,setup=setup),ensure_ascii=False,indent=2)+'\n')
    configs=[(hit,new,interval,index) for hit in (512,768,1024) for new in (256,512) for interval in profiles for index in range(3)]
    warmups=[]
    for index,(hit,new,interval,profile) in enumerate(configs):
        limits=profiles[interval][profile];upper=int(interval.split('-')[1]);inp=inputs[hit+new]
        a,t,_=ordinary(model,tok,compiled,cache[hit],inp,hit,upper,limits)
        b=detailed(model,tok,compiled,cache[hit],inp,hit,upper,limits)
        assert all(a['channels'][h]['token_ids']==b['channels'][h]['token_ids'] for h in HEADS)
        assert {h:len(a['channels'][h]['token_ids']) for h in HEADS}==limits
        warmups.append(dict(config=[hit,new,interval,profile],ordinary_wall_s=t,detailed_wall_s=b['request_wall_s']))
        if (index+1)%6==0:print('warmups',index+1,'/',len(configs),flush=True)
    (ROOT/'reports'/f'{STEM}_warmups.json').write_text(json.dumps(warmups,indent=2)+'\n')
    records=[];partial=ROOT/'reports'/f'{STEM}_raw.jsonl'
    with partial.open('w') as f:
        for round_no in range(7):
            order=list(configs);random.Random(20261008+round_no).shuffle(order)
            for hit,new,interval,profile in order:
                limits=profiles[interval][profile];upper=int(interval.split('-')[1]);inp=inputs[hit+new]
                if round_no%2:
                    b=detailed(model,tok,compiled,cache[hit],inp,hit,upper,limits)
                    a,t,pp=ordinary(model,tok,compiled,cache[hit],inp,hit,upper,limits)
                else:
                    a,t,pp=ordinary(model,tok,compiled,cache[hit],inp,hit,upper,limits)
                    b=detailed(model,tok,compiled,cache[hit],inp,hit,upper,limits)
                assert all(a['channels'][h]['token_ids']==b['channels'][h]['token_ids'] for h in HEADS)
                assert {h:len(a['channels'][h]['token_ids']) for h in HEADS}==limits
                row=dict(sample_id=f'{hit}_{new}_{interval}_p{profile}_r{round_no}',cache_hit_tokens=hit,new_prefill_tokens=new,
                    interval=interval,profile=profile,round=round_no,target_lengths=limits,ordinary_wall_s=t,ordinary_prefill_s=pp,
                    token_equal_profiled_vs_actual_engine=True,detailed=b)
                records.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            print('measured round',round_no+1,'/ 7, pairs',len(records),flush=True)
    integrity={hit:cache_digest(c['cache'])==c['sha256'] for hit,c in cache.items()};assert all(integrity.values())
    profiles_summary,cells=summarize(records)
    write_tables(records,cells)
    report=dict(model='SimpleLove 4B compact affine int4 group64',compute='W4A32, FP32 GDN, FP16 lossless metadata storage',
        scope='Mac M4 Pro; real model execution with externally controlled random stop lengths, not natural generation quality',
        rounds_per_profile=7,profiles_per_interval=3,measured_pairs=len(records),warmups=warmups,setup=setup,cached_prefix_unchanged=integrity,
        timer='perf_counter wall clock; model evaluated at each forward. Request starts before tokenizer and ends after JSON serialization; excludes model load, existing-prefix construction, HTTP transport and report disk writes.',
        instrumentation='Detailed mode synchronizes compaction separately. Ordinary paired request uses actual batched_generate and lazy compaction; no isolated GPU-time claim.',
        outlier_rule='Within identical cache/input/length vector: two-sided 3*1.4826*MAD, with 2% of median floor. All raw samples retained; warmth excluded independently.',
        caveats=['EOS tokens masked to guarantee requested range; external_token_limit is the stop reason. Output may be truncated and is not semantic/tool-quality acceptance.',
                 'Int4 unpack/dequant/projections and argmax included in compiled forward; their individual wall times cannot be separated without changing fused execution.',
                 'External GPU load, thermals and power state uncontrolled; rejected samples are statistical outliers, not proof of external-load causes.',
                 'Stage medians do not necessarily sum to request median; each representative record has an exact exclusive ledger and residual.'],
        profile_summaries=profiles_summary,cells=cells,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'reports'/f'{STEM}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for c in cells:print('SUMMARY',c['cache_hit_tokens'],c['new_prefill_tokens'],c['interval'],round(c['ordinary_wall_median_of_profile_medians_s']*1000,2),'ms kept',c['kept'],flush=True)

if __name__=='__main__':main()
