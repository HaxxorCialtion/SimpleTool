"""Sequential, one logical request at a time; exact terminal fork on TP1/2/4.
All model output is recorded data; no external tool execution. Run with launch.sh.
"""
import os,json,time,argparse,dataclasses,statistics,hashlib,traceback,math
from pathlib import Path
from benchmark_lib import score_case,normalize_prediction,render_prompt
H=['function','arg1','arg2','arg3','arg4','arg5','arg6']

def pct(v,q):
    if not v:return None
    v=sorted(v);i=(len(v)-1)*q;l=int(i);return v[l]+(v[min(l+1,len(v)-1)]-v[l])*(i-l)

def stats(v):
    v=[x for x in v if x is not None]
    return {'n':len(v),'p50':pct(v,.5),'p95':pct(v,.95),'p99':pct(v,.99),'mean':statistics.mean(v) if v else None}

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--tp',type=int,required=True)
    p.add_argument('--output',required=True);p.add_argument('--mode',choices=['screen','full','diagnose'],default='full')
    p.add_argument('--prefill-graphs',choices=['on','off'],default='on');p.add_argument('--repeats',type=int,default=10);p.add_argument('--soak',type=int,default=1000)
    p.add_argument('--kv-gib',type=float,default=4);p.add_argument('--gdn',default='flashinfer');p.add_argument('--async-scheduling',action='store_true')
    p.add_argument('--moe-backend',default='auto')
    p.add_argument('--allow-whole-cold-drift',action='store_true')
    a=p.parse_args();R=Path(a.output);R.mkdir(parents=True,exist_ok=False)
    import shutil
    (R/'source').mkdir()
    for name in ['bench.py','tp_worker.py','fast_prefill.py','PAYLOAD_MANIFEST.json']:
        shutil.copy2(Path(__file__).resolve().parent.parent/name,R/'source'/name)
    shutil.copy2(Path(__file__),R/'source'/Path(__file__).name)
    import torch,vllm
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import RequestOutputKind
    import prefill_runtime as rt
    rt.install()
    T=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    markers={h:T.encode('<'+h+'>',add_special_tokens=False) for h in H}
    endings={h:T.encode('</'+h+'>',add_special_tokens=False) for h in H}
    assert all(len(v)==1 for v in [*markers.values(),*endings.values()])
    cfg=dict(model=a.model,dtype='bfloat16',kv_cache_dtype='auto',mamba_ssm_cache_dtype='auto',
        tensor_parallel_size=a.tp,distributed_executor_backend='mp' if a.tp>1 else 'uni',
        worker_extension_cls='tp_worker.SimpleToolWorker',enable_prefix_caching=True,prefix_match_unit=1,
        prefix_caching_hash_algo='sha256',additional_config={'gdn_prefill_backend':a.gdn},
        max_model_len=4096,max_num_seqs=8,max_num_batched_tokens=4096,
        gpu_memory_utilization=.96,kv_cache_memory_bytes=int(a.kv_gib*2**30),
        enforce_eager=False,async_scheduling=a.async_scheduling,limit_mm_per_prompt={'image':0,'video':0},
        disable_log_stats=True,seed=1113,kernel_config={'enable_jit_warmup':False,'moe_backend':a.moe_backend},
        compilation_config={'mode':3,'cudagraph_mode':'FULL_DECODE_ONLY',
           'cudagraph_capture_sizes':[1,2,3,4,5,6,7,8],'max_cudagraph_capture_size':8})
    model_metadata={}
    for name in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json','generation_config.json']:
        path=Path(a.model)/name
        if path.exists():
            model_metadata[name]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (R/'config.json').write_text(json.dumps({'args':vars(a),'llm':cfg,'versions':{'torch':torch.__version__,'vllm':vllm.__version__,'cuda':torch.version.cuda},
       'model_metadata_hashes':model_metadata,
       'model_resolved_path':str(Path(a.model).resolve()),
       'validation_protocol':'split-boundary-reference-v3; synthetic tool parity stops at its own closing token; raw post-terminal and whole-prompt parity preserved separately; real tool cold parity remains strict',
       'ideal_workload_warmup_version':2,
       'real_validation_includes_schema_sized_batch':True,
       'real_validation_audits_same_split_reference':True,
       'prefill_capture_protocol':'native-distributed-graph-context-v2',
       'head_body_contract':'Raw body text; only a final closing marker is removed. Leading and trailing whitespace are significant for string arguments.',
       'input_precision':'Model artifact quantization is authoritative; model dtype BF16, KV/GDN auto. No content head in this benchmark.',
       'model_weight_quantization':json.loads((Path(a.model)/'config.json').read_text()).get('quantization_config'),
       'timing':'server pretokenized in-process critical path including scheduling, prime publication and tool assembly; tokenize cost separately added on natural tasks. Excludes HTTP/network, startup, capture and telemetry RPC.'},indent=2)+'\n')
    startup=time.perf_counter();llm=LLM(**cfg);eng=llm.llm_engine
    core=eng.engine_core.engine_core;sched=core.scheduler;executor=core.model_executor
    def rpc(name,**kw):return executor.collective_rpc(name,kwargs=kw)
    print('LOADED',round(time.perf_counter()-startup,2),flush=True)
    (R/'READY').write_text(str(time.time()))
    counter=0;rows=[];schedules=[]
    original_schedule=sched.schedule
    def schedule(*args,**kwargs):
        out=original_schedule(*args,**kwargs)
        if out.num_scheduled_tokens:
            schedules.append({'tokens':dict(out.num_scheduled_tokens),'time':time.perf_counter()})
        return out
    sched.schedule=schedule
    def reset():
        assert not eng.has_unfinished_requests()
        assert eng.reset_prefix_cache() is not False
    def run(common,heads,cap=64,fixed=False,phase='',tools=None,reference=False,write=True,cache_salt=None):
        nonlocal counter
        assert not eng.has_unfinished_requests();counter+=1;g=f'stbench-{counter}'
        ids={g+'-'+h:h for h in heads};prime=g+'-prime';schedules.clear();rt.EVENTS.clear()
        first={};last={};tokens={};finished={};outputs={};step_ms=[];arrival={};public_hit=0
        t=time.perf_counter();ready=None;call=None
        def drain():
            nonlocal ready,call
            while eng.has_unfinished_requests():
                s=time.perf_counter();oo=eng.step();now=time.perf_counter();step_ms.append((now-s)*1000)
                for o in oo:
                    idsout=list(o.outputs[0].token_ids) if o.outputs else []
                    if idsout and o.request_id not in first:first[o.request_id]=(now-t)*1000
                    old=tokens.get(o.request_id,[])
                    if len(idsout)>len(old):arrival.setdefault(o.request_id,[]).append({'at_ms':(now-t)*1000,'count':len(idsout),'new_tokens':len(idsout)-len(old)})
                    tokens[o.request_id]=idsout
                    if o.finished:
                        finished[o.request_id]=True;last[o.request_id]=(now-t)*1000
                        outputs[o.request_id]={'tokens':idsout,'text':T.decode(idsout,skip_special_tokens=False),
                            'cached_tokens':o.num_cached_tokens,'first_ms':first.get(o.request_id),'last_ms':last[o.request_id],
                            'finish_reason':o.outputs[0].finish_reason if o.outputs else None,
                            'stop_reason':o.outputs[0].stop_reason if o.outputs else None,
                            'metrics':dataclasses.asdict(o.metrics) if o.metrics else None}
                        if (a.mode=='diagnose' or phase.startswith('validation_')) and o.outputs:
                            outputs[o.request_id]['logprobs']=[{str(k):{'logprob':v.logprob,'rank':v.rank,'decoded_token':v.decoded_token} for k,v in lp.items()} for lp in (o.outputs[0].logprobs or [])]
                # Runtime emission point: once function completed, determine its
                # schema arity without consulting labels, then wait required heads.
                if tools and ready is None and g+'-function' in outputs:
                    fn=clean('function',outputs[g+'-function']['text'])
                    contract=next((x['function'] for x in tools if x['function']['name']==fn),None)
                    if contract:
                        props=list(contract.get('parameters',{}).get('properties',{}))
                        required=['function']+['arg'+str(i+1) for i in range(len(props))]
                        if all(g+'-'+h in outputs for h in required):
                            call,parse_errors=normalize_prediction({'heads':{h:clean(h,outputs[g+'-'+h]['text']) for h in required}},{'tools':tools})
                            ready=(time.perf_counter()-t)*1000
        try:
            if not reference:
                eng.add_request(prime,{'prompt_token_ids':common,**({'cache_salt':cache_salt} if cache_salt is not None else {})},SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,
                    extra_args={'simpletool_prefill':{'group':g,'role':'prime','N':len(common)}}))
                req=sched.requests[rt.GROUPS[g]['prime_id']]
                managers=sched.kv_cache_manager.coordinator.single_type_managers;pool=managers[0].block_pool
                assert pool.hash_block_size==1
                terminal=pool.get_cached_block(req.block_hashes[-1],[m.kv_cache_group_id for m in managers])
                if terminal is not None:
                    _,public_hit,_=sched.kv_cache_manager.get_computed_blocks(req,max_cache_hit_length=len(common))
                if public_hit==len(common):
                    rt.GROUPS[g]['ready']=True;eng.abort_request([prime])
                    outputs[prime]={'tokens':[],'cached_tokens':public_hit,'complete_public_hit':True}
                drain();prime_done=(time.perf_counter()-t)*1000
                assert not outputs[prime]['tokens']
                # Derive actual prefix reuse from returned request metrics, not
                # terminal-key probes (which only test full hits).
                public_hit=outputs[prime]['cached_tokens']
            else:prime_done=0
            for rid,h in ids.items():
                if reference:reset()
                extra=None if reference else {'simpletool_prefill':{'group':g,'role':'branch','N':len(common)}}
                eng.add_request(rid,{'prompt_token_ids':common+markers[h],**({'cache_salt':cache_salt} if cache_salt is not None else {})},SamplingParams(temperature=0,max_tokens=cap,
                    logprobs=5 if (a.mode=='diagnose' or phase.startswith('validation_')) else None,
                    ignore_eos=fixed,stop_token_ids=[] if fixed else endings[h],skip_special_tokens=False,
                    include_stop_str_in_output=True,output_kind=RequestOutputKind.CUMULATIVE,extra_args=extra))
                if reference:drain()
            if not reference:drain()
            wall=(time.perf_counter()-t)*1000
            hh={h:outputs[rid] for rid,h in ids.items()}
            if not reference:assert all(v['cached_tokens']==len(common) for v in hh.values()),[(h,v['cached_tokens']) for h,v in hh.items()]
            for h,v in hh.items():
                n=len(v['tokens']);v['tpot_host_ms']=(v['last_ms']-v['first_ms'])/(n-1) if n>1 else None
                v['arrivals']=arrival.get(g+'-'+h,[])
                v['terminated_on_head_end']=v['stop_reason']==endings[h][0] or bool(v['tokens'] and v['tokens'][-1]==endings[h][0])
            row={'id':g,'phase':phase,'N':len(common),'B':len(heads),'cap':cap,'fixed':fixed,'reference':reference,
                'prompt_sha256':hashlib.sha256(bytes(json.dumps(common),'ascii')).hexdigest(),
                'prime_ms':prime_done,'public_cached_tokens':public_hit,'public_new_tokens':len(common)-public_hit,
                'heads':hh,'wall_ms':wall,'tools_ready_ms':ready,'call':call,
                'schedule':[{'tokens':z['tokens'],'at_ms':(z['time']-t)*1000} for z in schedules],
                'step_ms':step_ms,'scheduler_events':list(rt.EVENTS)}
            if write:
                rows.append(row)
                with (R/'requests.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                if counter%25==0:
                    (R/'progress.json').write_text(json.dumps({'time':time.time(),'request':counter,'phase':phase,'N':len(common),'B':len(heads)})+'\n')
            return row
        finally:
            if eng.has_unfinished_requests():eng.abort_request([prime,*ids])
            st=rt.GROUPS.pop(g,None)
            if st:rt.PRIMES.discard(st['prime_id'])
    def clean(h,s):
        return s.removesuffix('</'+h+'>')
    def value(s):
        try:return json.loads(s)
        except Exception:return s
    def pin_prefix(common,cache_salt=None):
        nonlocal counter
        counter+=1;g=f'stbench-pin-{counter}';rid=g+'-prime'
        eng.add_request(rid,{'prompt_token_ids':common,**({'cache_salt':cache_salt} if cache_salt is not None else {})},SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,
            extra_args={'simpletool_prefill':{'group':g,'role':'prime','N':len(common)}}))
        req=sched.requests[rt.GROUPS[g]['prime_id']]
        cache,hit,_=sched.kv_cache_manager.get_computed_blocks(req,max_cache_hit_length=len(common))
        assert hit==len(common),('cannot pin missing terminal boundary',hit,len(common))
        pool=sched.kv_cache_manager.coordinator.single_type_managers[0].block_pool
        blocks=list({b.block_id:b for group in cache.blocks for b in group if not b.is_null}.values())
        pool.touch(blocks)
        eng.abort_request([rid]);st=rt.GROUPS.pop(g);rt.PRIMES.discard(st['prime_id'])
        return pool,blocks
    def textids(n,i=0):
        ids=T.encode(f'Observation version {i}. '+('workspace state sensor object target safe. '*(n+1)),add_special_tokens=False)
        return ids[:n]
    def receipt(name):
        d=rpc('sl_receipt');(R/(name+'.json')).write_text(json.dumps(d,indent=2)+'\n')
        pool=sched.kv_cache_manager.coordinator.single_type_managers[0].block_pool
        if hasattr(pool,'get_num_free_blocks'):
            (R/(name+'-pool.json')).write_text(json.dumps({'free_blocks':pool.get_num_free_blocks(),'total_blocks':getattr(pool,'num_gpu_blocks',None)})+'\n')
        return d
    # Content is excluded from validation, workload and timing in this run.
    from make_scenarios import fixture, SPECS
    domains=list(SPECS)
    def packed(domain,i):
        row=fixture(domain,i)
        base=T.encode(row['prefix'],add_special_tokens=False)
        dyn=T.encode(row['dynamic'],add_special_tokens=False)
        end=T.encode(row['end'],add_special_tokens=False)
        pad=T.encode(' .',add_special_tokens=False)
        assert len(pad)==1
        count=512-len(dyn)-len(end)
        assert count>=0,(domain,len(dyn),len(end))
        common=base+dyn+pad*count+end
        assert len(common)-len(base)==512
        return row,base,common,count

    validation=[]
    # Four different phases per domain; original full prefill and independent
    # split references are preserved. No timed result hides a failed gate.
    for domain in domains:
        for i in {'robot':[0,2,3,5], 'npc':[0,1,3,4], 'avatar':[0,1,2,3]}[domain]:
            row,base,common,padding=packed(domain,i)
            maxargs=max(len(t['function']['parameters']['properties']) for t in row['case']['tools'])
            heads=['function']+['arg'+str(k) for k in range(1,maxargs+1)]
            reset();rpc('sl_control',enabled=False,trace=False)
            shared=run(common,heads,64,False,'validation_shared')
            reset();independent=run(common,heads,64,False,'validation_whole_cold',reference=True)
            whole={h:shared['heads'][h]['tokens']==independent['heads'][h]['tokens'] for h in heads}
            split={}
            for h in heads:
                reset();one=run(common,[h],64,False,'validation_same_split')
                split[h]=shared['heads'][h]['tokens']==one['heads'][h]['tokens']
            reset();run(base,['function'],1,True,'validation_prefix_seed')
            cached=run(common,heads,64,False,'validation_cached_shared')
            assert cached['public_cached_tokens']==len(base)
            cached_split={}
            for h in heads:
                reset();run(base,['function'],1,True,'validation_prefix_seed')
                one=run(common,[h],64,False,'validation_cached_independent')
                assert one['public_cached_tokens']==len(base)
                cached_split[h]=cached['heads'][h]['tokens']==one['heads'][h]['tokens']
            result={'case_id':row['id'],'domain':domain,'whole_cold':whole,'same_split':split,
                    'cached_prefix_same_split':cached_split,
                    'cached_vs_whole_cold':{h:cached['heads'][h]['tokens']==independent['heads'][h]['tokens'] for h in heads}}
            validation.append(result)
            (R/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
            if not all(split.values()):
                print('UNCACHED_BATCH_NUMERICAL_ROUTE_DIFFERENCE',json.dumps(result),flush=True)
                assert a.allow_whole_cold_drift,('same-split validation failed',result)
            assert all(cached_split.values()),('cached-prefix same-split validation failed',result)
            if not all(whole.values()):
                print('WHOLE_COLD_NUMERICAL_ROUTE_DIFFERENCE',json.dumps(result),flush=True)
                assert a.allow_whole_cold_drift,('whole-cold validation failed',result)
    reset();rt.ACTIVE=False
    rpc('sl_control',enabled=True,capture=True,verify=True,trace=True)
    # Pure infra: exactly512 fixed public tokens +512 new, unique tokens.
    pure_base=textids(512,12345)
    run(pure_base,['function'],1,True,'infra_seed')
    pool,blocks=pin_prefix(pure_base)
    for b in [1,2,4,7]:
        heads=['function','arg1','arg2','arg3','arg4','arg5','arg6'][:b]
        for i in [90000,90001]:
            common=pure_base+textids(512,i)
            run(common,heads,4,True,'infra_graph_warm_cached')
            run(common,heads,4,True,'infra_graph_warm_cold',cache_salt='warm-'+str(b)+'-'+str(i))
    receipt('infra-warm-workers')
    rpc('sl_control',enabled=True,capture=False,verify=False,trace=False)
    for b in [1,2,4,7]:
        heads=['function','arg1','arg2','arg3','arg4','arg5','arg6'][:b]
        for cap in [16,32]:
            for i in range(a.repeats):
                common=pure_base+textids(512,100000+b*10000+cap*100+i)
                cold=run(common,heads,cap,True,f'infra_cold_P512_S512_B{b}_T{cap}',cache_salt='cold-'+str(counter))
                assert cold['public_cached_tokens']==0
                warm=run(common,heads,cap,True,f'infra_cached_P512_S512_B{b}_T{cap}')
                assert warm['public_cached_tokens']==512 and warm['public_new_tokens']==512
    receipt('infra-end-workers')
    pool.free_blocks(blocks);reset()

    pins=[];prefixes={};warm_inputs=[]
    rpc('sl_control',enabled=True,capture=True,verify=True,trace=True)
    for domain in domains:
        row,base,common,padding=packed(domain,90000)
        run(base,['function'],1,True,'scenario_seed_'+domain)
        pin_pool,pin_blocks=pin_prefix(base);pins.append((pin_pool,pin_blocks));prefixes[domain]=base
        heads=['function']+['arg'+str(k) for k in range(1,1+max(len(t['function']['parameters']['properties']) for t in row['case']['tools']))]
        for i in [90000,90001,90002,90003]:
            row,_,common,_=packed(domain,i)
            run(common,heads,8,False,'scenario_graph_warm_'+domain,tools=row['case']['tools'])
        warm_inputs.append({'domain':domain,'prefix_tokens':len(base),'new_tokens':512,'N':len(common),'B':len(heads),'padding_tokens':padding})
    (R/'workload-config.json').write_text(json.dumps({'scenarios':warm_inputs,'unique_test_observations_per_domain':60,
        'suffix_tokens':512,'padding':'Explicit inert diagnostic punctuation tokens; IDs are the actual inference inputs.',
        'timing':'No content. Host pretokenized tools-ready; no HTTP/network, environment step or actual tool execution.'},indent=2)+'\n')
    before=receipt('scenario-warm-workers')
    rpc('sl_control',enabled=True,capture=False,verify=False,trace=False)
    quality=[];prompt_hashes=set()
    for i in range(60):
        # Interleave domains so all three pinned boundary states must survive.
        for domain in domains:
            row,base,common,padding=packed(domain,i)
            assert base==prefixes[domain]
            heads=['function']+['arg'+str(k) for k in range(1,1+max(len(t['function']['parameters']['properties']) for t in row['case']['tools']))]
            q=run(common,heads,128,False,'real_'+domain,tools=row['case']['tools'])
            assert q['public_cached_tokens']==len(base) and q['public_new_tokens']==512
            assert q['prompt_sha256'] not in prompt_hashes
            prompt_hashes.add(q['prompt_sha256'])
            q.update(case_id=row['id'],domain=domain,expected=row['expected'],padding_tokens=padding,
                score=score_case({'heads':{h:clean(h,v['text']) for h,v in q['heads'].items()}},row['case']))
            q['correct']=q['score']['exact_correct'];q['function_correct']=q['score']['name_correct']
            q['tokenize_ms']=None
            with (R/'quality.jsonl').open('a') as f:f.write(json.dumps(q,ensure_ascii=False)+'\n')
            if i<2:
                example={**row,'prefix_tokens':len(base),'suffix_tokens':512,'padding_tokens':padding,
                    'input_token_ids':common,'actual':{h:v['text'] for h,v in q['heads'].items()},'score':q['score']}
                with (R/'examples.jsonl').open('a') as f:f.write(json.dumps(example,ensure_ascii=False)+'\n')
            quality.append(q)
    after=receipt('scenario-measured-workers')
    for x,y in zip(before,after):
        a_modes=x['fast_prefill']['cumulative_modes'];b_modes=y['fast_prefill']['cumulative_modes']
        assert x['fast_prefill']['graphs']==y['fast_prefill']['graphs']
        assert sum(v for k,v in a_modes.items() if k.startswith('eager-'))==sum(v for k,v in b_modes.items() if k.startswith('eager-'))
    for i in range(a.soak):
        domain=domains[i%3];row,base,common,padding=packed(domain,200000+i)
        heads=['function']+['arg'+str(k) for k in range(1,1+max(len(t['function']['parameters']['properties']) for t in row['case']['tools']))]
        q=run(common,heads,128,False,'soak_'+domain,tools=row['case']['tools'])
        assert q['public_cached_tokens']==len(base) and q['public_new_tokens']==512
        if (i+1)%100==0:
            receipt('soak-workers-'+str(i+1));write_summary(R,rows)
            print('SCENARIO_SOAK',i+1,flush=True)
    end=receipt('final-workers')
    for x,y in zip(after,end):
        am=x['fast_prefill']['cumulative_modes'];bm=y['fast_prefill']['cumulative_modes']
        assert x['fast_prefill']['graphs']==y['fast_prefill']['graphs']
        assert sum(v for k,v in am.items() if k.startswith('eager-'))==sum(v for k,v in bm.items() if k.startswith('eager-'))
    for pin_pool,pin_blocks in pins:pin_pool.free_blocks(pin_blocks)
    write_summary(R,rows)
    (R/'DONE').write_text(str(time.time()))
    print('DONE_SCENARIOS',R,flush=True)

def write_summary(R,rows):
    phases=sorted(set(x['phase'] for x in rows));out={}
    for ph in phases:
        rs=[r for r in rows if r['phase']==ph]
        out[ph]={'requests':len(rs),'wall_ms':stats([r['wall_ms'] for r in rs]),
            'prime_ms':stats([r['prime_ms'] for r in rs]),'tools_ready_ms':stats([r['tools_ready_ms'] for r in rs]),
            'tpot_host_ms':stats([h['tpot_host_ms'] for r in rs for h in r['heads'].values()]),
            'new_public_tokens':stats([r['public_new_tokens'] for r in rs]),
            'head_ttft_ms':stats([h['first_ms'] for r in rs for h in r['heads'].values()])}
    q=R/'quality.jsonl'
    if q.exists():
        qq=list(map(json.loads,q.read_text().splitlines()));out['quality']={'n':len(qq),'correct':sum(x['correct'] for x in qq),'required_correct':sum(x['score']['required_correct'] for x in qq),'function_correct':sum(x['function_correct'] for x in qq)}
    (R/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();raise
