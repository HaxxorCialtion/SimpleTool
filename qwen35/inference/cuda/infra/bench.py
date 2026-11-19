"""Sequential, one logical request at a time; exact terminal fork on TP1/2/4.
All model output is recorded data; no external tool execution. Run with launch.sh.
"""
import os,json,time,argparse,dataclasses,statistics,hashlib,traceback,math
from pathlib import Path
from benchmark_lib import score_case,normalize_prediction,render_prompt
H=['content','function','arg1','arg2','arg3','arg4','arg5','arg6']

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
    a=p.parse_args();R=Path(a.output);R.mkdir(parents=True,exist_ok=False)
    import shutil
    (R/'source').mkdir()
    for name in ['bench.py','tp_worker.py','fast_prefill.py','PAYLOAD_MANIFEST.json']:
        shutil.copy2(Path(__file__).parent/name,R/'source'/name)
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
       'input_precision':'FP8 block weights, BF16 model dtype and KV, native GDN dtype; FP8 GEMM kernels may dynamically quantize activation operands',
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
    def run(common,heads,cap=64,fixed=False,phase='',tools=None,reference=False,write=True):
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
                        if a.mode=='diagnose' and o.outputs:
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
                eng.add_request(prime,{'prompt_token_ids':common},SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,
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
                eng.add_request(rid,{'prompt_token_ids':common+markers[h]},SamplingParams(temperature=0,max_tokens=cap,
                    logprobs=5 if a.mode=='diagnose' else None,
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
    def pin_prefix(common):
        nonlocal counter
        counter+=1;g=f'stbench-pin-{counter}';rid=g+'-prime'
        eng.add_request(rid,{'prompt_token_ids':common},SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,
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
    # No claim of quality or timing until exact-fork and graph parity pass.
    rt.ACTIVE=True;rpc('sl_control',trace=True)
    sanity=[]
    for n in [256,512,1024]:
        common=textids(n);reset();raw=run(common,H,cap=4,fixed=True,phase='sanity_raw')
        reset();ref=run(common,H,cap=4,fixed=True,phase='sanity_independent',reference=True)
        parity={h:raw['heads'][h]['tokens']==ref['heads'][h]['tokens'] for h in H}
        (R/('cold-prefill-parity-N'+str(n)+'.json')).write_text(json.dumps(parity,indent=2)+'\n')
        # Compare the optimization with the same N-token prefill + marker
        # algorithm run independently for every branch. A single N+1 cold
        # prefill changes GDN/FP8 kernel shapes; its output remains evidence,
        # but is not a bitwise reference for splitting that prefill boundary.
        split_heads={}
        for h in H:
            reset()
            split_heads[h]=run(common,[h],cap=4,fixed=True,phase='sanity_split_single')['heads'][h]
        raw_split_parity={h:raw['heads'][h]['tokens']==split_heads[h]['tokens'] for h in H}
        (R/('split-fork-raw-parity-N'+str(n)+'.json')).write_text(json.dumps(raw_split_parity,indent=2)+'\n')
        def through_head_end(h,tokens):
            # Fixed-length microbenchmarks intentionally continue after the
            # stop token. Such tokens are outside the runtime head contract.
            # Keep the raw comparison above, and require exact tokens through
            # the first actual closing marker, including that marker itself.
            stop=endings[h][0]
            return tokens[:tokens.index(stop)+1] if stop in tokens else tokens
        split_parity={h:through_head_end(h,raw['heads'][h]['tokens'])==through_head_end(h,split_heads[h]['tokens']) for h in H}
        (R/('split-fork-parity-N'+str(n)+'.json')).write_text(json.dumps(split_parity,indent=2)+'\n')
        if a.mode!='diagnose':
            assert all(v for h,v in split_parity.items() if h!='content'),'batched tool fork differs from independent split-boundary reference'
        else:
            reset();single=run(common,['function'],cap=4,fixed=True,phase='diagnostic_fork_single')
            (R/('diagnostic-function-N'+str(n)+'.json')).write_text(json.dumps({'batched':raw['heads']['function'],'single_fork':single['heads']['function'],'independent':ref['heads']['function']},indent=2)+'\n')
        reset();rpc('sl_control',enabled=a.prefill_graphs=='on',capture=a.prefill_graphs=='on',verify=a.prefill_graphs=='on',trace=True)
        graph=run(common,H,cap=4,fixed=True,phase='sanity_graph')
        assert {h:v['tokens'] for h,v in raw['heads'].items()}=={h:v['tokens'] for h,v in graph['heads'].items()},'prefill graph tokens differ'
        sanity.append({'N':n,'tool_head_token_parity':all(split_parity[h] for h in H if h!='content'),'content_token_parity':split_parity['content'],'whole_prompt_cold_parity':parity,'split_boundary_parity':split_parity,'raw_post_terminal_included_parity':raw_split_parity,'graph_token_parity':True,'workers':receipt('sanity-workers-'+str(n))})
        rpc('sl_control',enabled=False,trace=True)
    # Independent real tool inputs confirm that decisions and argument bodies
    # are invariant to public-prefix reuse and branch batching. Content is
    # recorded separately because it is excluded from the tool contract.
    natural=[]
    for suite in ['hard','multiarg']:
        cases=list(map(json.loads,(Path(__file__).parent/'data'/(suite+'.jsonl')).read_text().splitlines()))
        for c in [cases[0],cases[-1]]:
            common=T.encode(c['prompt']+'<think>\n\n</think>\n\n',add_special_tokens=False)
            reset();rpc('sl_control',enabled=False,trace=False)
            raw=run(common,H,cap=32,fixed=False,phase='sanity_real_fork')
            reset();ref=run(common,H,cap=32,fixed=False,phase='sanity_real_independent',reference=True)
            pp={h:raw['heads'][h]['tokens']==ref['heads'][h]['tokens'] for h in H}
            # Production quality submits only the maximum arity allowed by
            # its tool definitions, without content. Check that exact batch
            # layout as well as the full eight-head diagnostic layout.
            maxargs=max(len(t['function'].get('parameters',{}).get('properties',{})) for t in c['tools'])
            tool_heads=['function']+['arg'+str(k) for k in range(1,maxargs+1)]
            reset();actual=run(common,tool_heads,cap=32,fixed=False,phase='sanity_real_schema_batch')
            actual_pp={h:actual['heads'][h]['tokens']==ref['heads'][h]['tokens'] for h in tool_heads}
            # Separate numerical changes from splitting N+marker into N,marker
            # from the actual cache-sharing/batched-decode transformation.
            split_reference={}
            for h in H:
                reset()
                split_reference[h]=run(common,[h],cap=32,fixed=False,phase='sanity_real_split_single')['heads'][h]
            split_pp={h:raw['heads'][h]['tokens']==split_reference[h]['tokens'] for h in H}
            actual_split_pp={h:actual['heads'][h]['tokens']==split_reference[h]['tokens'] for h in tool_heads}
            natural.append({'case':c['id'],'per_head_token_parity':pp,'schema_sized_heads':tool_heads,'schema_sized_tool_parity':actual_pp,
                            'same_split_per_head_token_parity':split_pp,'same_split_schema_sized_tool_parity':actual_split_pp})
            (R/'sanity-real.json').write_text(json.dumps(natural,indent=2)+'\n')
            if a.mode!='diagnose':
                assert all(v for h,v in split_pp.items() if h!='content'),'real batched tool fork differs from independent split-boundary reference'
                assert all(actual_split_pp.values()),'schema-sized production tool batch differs from same-split reference'
                assert all(v for h,v in pp.items() if h!='content'),'real tool fork parity failed'
                assert all(actual_pp.values()),'schema-sized production tool batch parity failed'
    (R/'sanity-real.json').write_text(json.dumps(natural,indent=2)+'\n')
    (R/'sanity.json').write_text(json.dumps(sanity,indent=2)+'\n')
    if a.mode=='diagnose':
        receipt('diagnostic-workers')
        (R/'DIAGNOSTIC_DONE').write_text(str(time.time()))
        print('DIAGNOSTIC_ONLY_NOT_ACCEPTED',R,flush=True)
        return
    rt.ACTIVE=False;rpc('sl_control',enabled=a.prefill_graphs=='on',capture=False,verify=False,trace=False)
    for n in ([512] if a.mode=='screen' else [256,512,1024]):
        for b in ([1,8] if a.mode=='screen' else [1,2,4,8]):
            heads=['function','arg1','arg2','arg3','arg4','arg5','arg6','content'][:b]
            common=textids(n);reset()
            # Warm native graphs/GEMM kernels; not included in warm summaries.
            for _ in range(2):run(common,heads,16,True,'warmup')
            for cap in ([32] if a.mode=='screen' else [16,64]):
                for i in range(a.repeats):
                    reset();run(textids(n,i),heads,cap,True,f'micro_cold_N{n}_B{b}_T{cap}')
                    # Reuse entire common state only as a separately labeled ceiling.
                    run(textids(n,i),heads,cap,True,f'micro_fullhit_N{n}_B{b}_T{cap}')
            receipt(f'micro-workers-N{n}-B{b}')
            rpc('sl_control',enabled=a.prefill_graphs=='on',capture=False,trace=True)
            rt.ACTIVE=True;reset()
            run(textids(n,7722),heads,32,True,f'profile_N{n}_B{b}')
            receipt(f'profile-workers-N{n}-B{b}')
            rt.ACTIVE=False;rpc('sl_control',enabled=a.prefill_graphs=='on',capture=False,trace=False)
    if a.mode=='full':
        # Known boundary: prefill immutable fixed task region, cache terminal
        # GDN/attention state, append changed environment/query on each request.
        for n in [256,512,1024]:
            base=textids(n,9182);reset();rpc('sl_control',enabled=a.prefill_graphs=='on',capture=False)
            run(base,['function'],1,True,'fixed_boundary_seed')
            pinned_pool,pinned_blocks=pin_prefix(base)
            # Capture the cached-boundary continuation shape separately. Cold
            # full-prefix graphs do not represent a short cached suffix graph.
            rpc('sl_control',enabled=a.prefill_graphs=='on',capture=a.prefill_graphs=='on',verify=a.prefill_graphs=='on',trace=True)
            for warm_i in [0,10]:
                tail=T.encode(f' Environment frame={warm_i}; target=box{warm_i%7}; action=move. Query: next action.',add_special_tokens=False)
                run(base+tail,['function','arg1','arg2','arg3'],2,True,'fixed_boundary_graph_warmup')
            receipt('fixed-boundary-warm-workers-'+str(n))
            rpc('sl_control',enabled=a.prefill_graphs=='on',capture=False,verify=False,trace=False)
            for i in range(40):
                tail=T.encode(f' Environment frame={i}; target=box{i%7}; action=move. Query: next action.',add_special_tokens=False)
                r=run(base+tail,['function','arg1','arg2','arg3'],32,True,f'fixed_boundary_N{n}')
                assert r['public_cached_tokens']>=n,'fixed boundary state was not retained'
            pinned_pool.free_blocks(pinned_blocks)
        receipt('fixed-boundary-workers')
        task_cases=list(map(json.loads,(Path(__file__).parent/'data'/'hard.jsonl').read_text().splitlines()))[:2]
        prefix_text=task_cases[0]['prompt'].split('<|im_start|>user\n',1)[0]
        prefix=T.encode(prefix_text,add_special_tokens=False)
        assert all(c['prompt'].startswith(prefix_text) for c in task_cases)
        reset();run(prefix,['function'],1,True,'real_fixed_task_seed')
        pinned_pool,pinned_blocks=pin_prefix(prefix)
        # A configured single-task space can warm its changing-observation
        # continuation kernels too. Use disjoint frame ids so measured requests
        # never receive a complete-prompt hit from warmup. All warm and measured
        # frame ids have four digits: this tokenizer splits numeric digits, so
        # three-digit warmups do not cover one/two-digit measured shapes.
        rpc('sl_control',enabled=a.prefill_graphs=='on',capture=a.prefill_graphs=='on',verify=a.prefill_graphs=='on',trace=True)
        for warm_i in [9000,9001,9900,9901]:
            c=task_cases[warm_i%2]
            warm_case={**c['case'],'environment':{'frame_id':warm_i,**c['case']['environment']}}
            common=T.encode(render_prompt(warm_case)+'<think>\n\n</think>\n\n',add_special_tokens=False)
            assert common[:len(prefix)]==prefix
            maxargs=max(len(t['function']['parameters']['properties']) for t in c['tools'])
            run(common,['function']+['arg'+str(k) for k in range(1,maxargs+1)],32,False,'real_fixed_task_graph_warmup',tools=c['tools'])
        real_warm_receipt=receipt('real-fixed-task-warm-workers')
        rpc('sl_control',enabled=a.prefill_graphs=='on',capture=False,verify=False,trace=False)
        for i in range(30):
            c=task_cases[i%2];input_case={**c['case'],'environment':{'frame_id':1000+i,**c['case']['environment']}}
            common=T.encode(render_prompt(input_case)+'<think>\n\n</think>\n\n',add_special_tokens=False)
            assert common[:len(prefix)]==prefix,'token boundary is not stable'
            maxargs=max(len(t['function']['parameters']['properties']) for t in c['tools'])
            h=['function']+['arg'+str(k) for k in range(1,maxargs+1)]
            result=run(common,h,128,False,'real_fixed_task',tools=c['tools'])
            result['score']=score_case({'heads':{k:clean(k,v['text']) for k,v in result['heads'].items()}},c['case'])
            with (R/'fixed-task-quality.jsonl').open('a') as f:f.write(json.dumps(result,ensure_ascii=False)+'\n')
        pinned_pool.free_blocks(pinned_blocks)
        real_measured_receipt=receipt('real-fixed-task-workers')
        if a.prefill_graphs=='on':
            for before,after in zip(real_warm_receipt,real_measured_receipt):
                miss=lambda w:sum(v for k,v in w['fast_prefill']['cumulative_modes'].items() if k.startswith('eager-'))
                assert miss(after)==miss(before),'known real-task continuation missed its prewarmed graph shape'
        # Natural tool calls use frozen public benchmark prompts, exact order,
        # blank think as used for matching 5% checkpoints. Never use labels to
        # decide the number of heads or generation stopping.
        for suite in ['cf-zh','cf-en','hard','multiarg']:
            for case in map(json.loads,(Path(__file__).parent/'data'/(suite+'.jsonl')).read_text().splitlines()):
                prompt=case['prompt']+'<think>\n\n</think>\n\n'
                tokenize_start=time.perf_counter();common=T.encode(prompt,add_special_tokens=False);tokenize_ms=(time.perf_counter()-tokenize_start)*1000
                maxargs=max(len(t['function'].get('parameters',{}).get('properties',{})) for t in case['tools'])
                heads=['function']+['arg'+str(i) for i in range(1,maxargs+1)]
                row=run(common,heads,128,False,'quality-'+suite,tools=case['tools'])
                row['tokenize_ms']=tokenize_ms;row['tokenize_plus_tools_ready_ms']=tokenize_ms+row['tools_ready_ms'] if row['tools_ready_ms'] is not None else None
                row['case_id']=case['id'];row['expected']=case['expected']
                row['score']=score_case({'heads':{h:clean(h,v['text']) for h,v in row['heads'].items()}},case['case'])
                row['correct']=row['score']['exact_correct'];row['function_correct']=row['score']['name_correct']
                row['required_correct']=row['score']['required_correct']
                with (R/'quality.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            receipt('quality-workers-'+suite)
        # Steady load, one logical request. Change both tokens and >24 shapes,
        # include fresh long contexts to pressure state/KV cache, and revisit a
        # pinned shape. Capture remains OFF: no growing graph cache or warmup
        # work in production request paths. Count all eager fallbacks.
        rpc('sl_control',enabled=a.prefill_graphs=='on',capture=False)
        for i in range(a.soak):
            n=512 if i%4==0 else [256,384,512,768,1024,1536][(i//4)%6]+(i%31)
            r=run(textids(n,i+9999),['function','arg1','arg2','arg3'],16,True,'soak')
            if (i+1)%100==0:
                rr=receipt('soak-workers-'+str(i+1))
                recent=rows[-100:];print('SOAK',i+1,stats([x['wall_ms'] for x in recent]),flush=True)
                write_summary(R,rows)
    receipt('final-workers');write_summary(R,rows)
    (R/'DONE').write_text(str(time.time()))
    print('DONE',R,flush=True)

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
