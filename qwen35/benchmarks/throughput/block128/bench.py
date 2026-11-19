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
        max_model_len=4096,max_num_seqs=1024,max_num_batched_tokens=4096,
        gpu_memory_utilization=.96,kv_cache_memory_bytes=int(a.kv_gib*2**30),
        enforce_eager=False,async_scheduling=a.async_scheduling,limit_mm_per_prompt={'image':0,'video':0},
        disable_log_stats=True,seed=1113,kernel_config={'enable_jit_warmup':False,'moe_backend':a.moe_backend},
        compilation_config={'mode':3,'cudagraph_mode':'FULL_DECODE_ONLY',
           'cudagraph_capture_sizes':[1,32,256,384,512,576,640,704,768,896,1024],'max_cudagraph_capture_size':1024})
    canonical = ['function','arg1','arg2','arg3','arg4','arg5','arg6','content']
    branch_names = []
    for i in range(1024):
        original=canonical[i%8]
        alias=original if i<8 else original+'@'+str(i//8)
        branch_names.append(alias)
        markers[alias]=markers[original]
        endings[alias]=endings[original]
    model_metadata={}
    for name in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json','generation_config.json']:
        path=Path(a.model)/name
        if path.exists():
            model_metadata[name]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (R/'config.json').write_text(json.dumps({'args':vars(a),'llm':cfg,'versions':{'torch':torch.__version__,'vllm':vllm.__version__,'cuda':torch.version.cuda},
       'model_metadata_hashes':model_metadata,
       'model_resolved_path':str(Path(a.model).resolve()),
       'validation_protocol':'throughput extension to 1024: record independent same-split parity through head end; graph/eager hidden+GDN exact check; assert full public hits and single-token scheduling' ,
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

    import random,subprocess,platform
    batches=[1,32,256,384,512,576,640,704,768,896,1024]
    N=512;cap=64
    (R/'machine.json').write_text(json.dumps({'gpu':subprocess.getoutput('nvidia-smi -q'),
        'python':platform.python_version(),'host':platform.node(),'candidate_batches':batches,
        'marker_order':canonical,'max_batch':1024,
        'notes':['B>8 repeats eight trained markers; not independent distinct tools or independent-user prompts',
                 'fixed-length ignore_eos; one logical group at a time; public prefill emits zero tokens',
                 'host-observed TPOT includes harness/engine scheduling and sampling; no HTTP/network',
                 'large-batch test uses 80 GiB cache instead of previous 8 GiB; controls remeasured',
                 'decode throughput stopping condition: paired bootstrap 95pct upper CI below 1 after confirmation']},indent=2))
    def save(name,v): (R/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
    checked=0;measured=[];validations=[];pair_reports=[];profiles={};sampled=set()
    def check(row):
        nonlocal checked
        B=row['B']; g=row['id']
        ss=[z for z in row['schedule'] if any(k.startswith(g+'-') and not k.startswith(g+'-prime') for k in z['tokens'])]
        pp=[z for z in row['schedule'] if any(k.startswith(g+'-prime') for k in z['tokens'])]
        assert len(ss)==row['cap'],('decode stage count',B,len(ss),row['cap'])
        assert all(len(z['tokens'])==B and all(v==1 for v in z['tokens'].values()) for z in ss),('batch split or branch prefill',B)
        assert len(pp)==1 and sum(pp[0]['tokens'].values())==N,('cold public prefill split',B,pp)
        assert row['public_cached_tokens']==0
        assert all(h['cached_tokens']==N and len(h['tokens'])==row['cap'] for h in row['heads'].values()),('incomplete fork or generation',B)
        checked+=1
    def metrics(row):
        heads=list(row['heads'].values())
        first=min(h['first_ms'] for h in heads);last=max(h['last_ms'] for h in heads)
        return {'id':row['id'],'phase':row['phase'],'N':N,'B':row['B'],'T':row['cap'],
            'tpot_ms':statistics.median(h['tpot_host_ms'] for h in heads),
            'aggregate_decode_tps':row['B']*(row['cap']-1)*1000/(last-first),
            'prime_ms':row['prime_ms'],'marker_stage_ms':max(h['first_ms'] for h in heads)-row['prime_ms'],
            'wall_ms':row['wall_ms'],'first_all_ms':max(h['first_ms'] for h in heads),
            'decode_steps':len([z for z in row['schedule'] if any('-prime' not in k for k in z['tokens'])]),
            'cached_tokens_min':min(h['cached_tokens'] for h in heads),'layout_valid':True}
    def trial(B,i,phase='screen',Tcap=64,keep=True):
        reset()
        row=run(textids(N,i),branch_names[:B],Tcap,True,phase,write=False)
        try:check(row)
        except Exception:
            save("FAILED-layout-B"+str(B)+".json",row)
            raise
        m=metrics(row)
        if keep:
            measured.append(m)
            with (R/'metrics.jsonl').open('a') as f:f.write(json.dumps(m)+'\n')
            if B not in sampled:
                save('timeline-B'+str(B)+'.json',row);sampled.add(B)
        save('progress.json',{'time':time.time(),'request':counter,'B':B,'phase':phase,'checked':checked})
        return m,row
    def summary(ms):
        return {'n':len(ms),**{k:stats([m[k] for m in ms]) for k in ['tpot_ms','aggregate_decode_tps','prime_ms','marker_stage_ms','wall_ms']}}
    def ci(ratios):
        rng=random.Random(111617);n=len(ratios)
        boots=[statistics.median(rng.choices(ratios,k=n)) for _ in range(4000)]
        return [pct(boots,.025),pct(boots,.975)]
    def compare(best,B,n,phase):
        ra=[];rb=[];ratios=[];rng=random.Random(1116+B)
        for rep in range(n):
            order=[best,B];rng.shuffle(order);pair={}
            for b in order:
                m,_=trial(b,10000+rep,phase)
                pair[b]=m
            ra.append(pair[best]);rb.append(pair[B]);ratios.append(pair[B]['aggregate_decode_tps']/pair[best]['aggregate_decode_tps'])
            if (rep+1)%5==0:print('PAIR',phase,best,B,rep+1,n,flush=True)
        out={'reference_B':best,'candidate_B':B,'phase':phase,'pairs':n,
             'ratio_p50':statistics.median(ratios),'ratio_ci95':ci(ratios),
             'reference':summary(ra),'candidate':summary(rb),'ratios':ratios}
        pair_reports.append(out);save('comparisons.json',pair_reports)
        return out
    def prepare(B):
        rpc('sl_control',enabled=False,capture=False,verify=False,trace=False)
        _,eager=trial(B,9000,'validation-eager',Tcap=4,keep=False)
        rpc('sl_control',enabled=True,capture=True,verify=True,trace=False)
        _,graph=trial(B,9000,'validation-graph',Tcap=4,keep=False)
        assert {h:v['tokens'] for h,v in eager['heads'].items()}=={h:v['tokens'] for h,v in graph['heads'].items()},('public graph/eager token parity',B)
        aliases={h:graph['heads'][h]['tokens']==graph['heads'][h.split('@')[0]]['tokens'] for h in graph['heads'] if '@' in h}
        validations.append({'B':B,'public_graph_eager_tokens_exact':True,'layout_valid':True,
            'alias_parity_failures':[h for h,v in aliases.items() if not v],
            'prefix_hit_tokens':N})
        save('validation.json',validations)
        rpc('sl_control',enabled=True,capture=True,verify=False,trace=False)
        for i in range(3):trial(B,9100+i,'warmup',keep=False)
        rpc('sl_control',enabled=True,capture=False,verify=False,trace=False)
        receipt('prepared-B'+str(B))
        print('PREPARED',B,flush=True)
    stop=None;best=512
    # Fresh low-batch controls use the same cache reservation and decoder captures.
    for B in [1,32,256,384,512]:
        prepare(B)
        ms=[]
        for rep in range(20):m,_=trial(B,10000+rep,'control');ms.append(m)
        profiles[B]=summary(ms);save('profiles.json',profiles)
        print('CONTROL',B,profiles[B]['tpot_ms']['p50'],profiles[B]['aggregate_decode_tps']['p50'],flush=True)
    for B in batches:
        if B<=512:continue
        prepare(B)
        out=compare(best,B,12,'screen')
        profiles[B]=out['candidate'];save('profiles.json',profiles)
        print('SCREEN',B,out['candidate']['tpot_ms']['p50'],out['candidate']['aggregate_decode_tps']['p50'],'ratio',out['ratio_p50'],'ci',out['ratio_ci95'],flush=True)
        if out['ratio_p50']<1:
            confirm=compare(best,B,24,'confirm')
            profiles[B]=confirm['candidate'];save('profiles.json',profiles)
            print('CONFIRM',B,confirm['ratio_p50'],confirm['ratio_ci95'],flush=True)
            if confirm['ratio_ci95'][1]<1:
                stop={'kind':'confirmed_throughput_decline','peak_reference_B':best,'first_confirmed_lower_B':B,'comparison':confirm}
                # A further larger point checks that the decline is not merely a local dip.
                nxt=next((x for x in batches if x>B),None)
                if nxt:
                    prepare(nxt);more=compare(best,nxt,12,'post-decline')
                    profiles[nxt]=more['candidate'];save('profiles.json',profiles)
                    stop['larger_check']=more
                    if more['ratio_ci95'][0]>1:
                        print('DECLINE_WAS_LOCAL_CONTINUE',B,nxt,flush=True)
                        best=nxt;stop=None;continue
                break
        if out['ratio_p50']>1:best=B
    if stop is None:stop={'kind':'configured_batch_limit_without_confirmed_decline','largest_B':max(profiles),'best_B':best}
    receipt('final-workers')
    save('completion.json',{'stop':stop,'checked_requests':checked,'measured_requests':len(measured),
        'completed_unix':time.time(),'tools_executed':False,'profiles':profiles,'validation_cells':len(validations)})
    (R/'DONE').write_text(stop['kind']+'\n')
    print('DONE',stop['kind'],flush=True)

if __name__=='__main__':main()
