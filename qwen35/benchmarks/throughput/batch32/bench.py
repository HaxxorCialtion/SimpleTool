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
        max_model_len=4096,max_num_seqs=32,max_num_batched_tokens=4096,
        gpu_memory_utilization=.96,kv_cache_memory_bytes=int(a.kv_gib*2**30),
        enforce_eager=False,async_scheduling=a.async_scheduling,limit_mm_per_prompt={'image':0,'video':0},
        disable_log_stats=True,seed=1113,kernel_config={'enable_jit_warmup':False,'moe_backend':a.moe_backend},
        compilation_config={'mode':3,'cudagraph_mode':'FULL_DECODE_ONLY',
           'cudagraph_capture_sizes':[1,2,3,4,5,6,7,8,12,16,24,32],'max_cudagraph_capture_size':32})
    canonical = ['function','arg1','arg2','arg3','arg4','arg5','arg6','content']
    branch_names = []
    for i in range(32):
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
       'validation_protocol':'batch32 extension: record independent same-split parity through head end; graph/eager hidden+GDN exact check; assert full public hits and single-token scheduling' ,
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
    batches=[1,2,4,8,12,16,24,32]
    Ns=[256,512,1024]
    caps=[16,64]
    (R/'machine.json').write_text(json.dumps({'gpu':subprocess.getoutput('nvidia-smi -q'),
        'python':platform.python_version(),'host':platform.node(),
        'batch_sizes':batches,'marker_mapping':{x:canonical[i%8] for i,x in enumerate(branch_names)},
        'notes':['B>8 repeats the eight trained markers, not a 32-head model',
                 'B1 is a SimpleTool function branch, not ordinary JSON AR',
                 'fixed-length ignore_eos; no tools executed; no HTTP/network timing',
                 'one logical request at a time; N-token prefill emits zero tokens',
                 'different shapes can change FP8 numerical results; parity is measured explicitly']},indent=2))
    def save_json(name,obj):
        (R/name).write_text(json.dumps(obj,indent=2)+'\n')
    def check_layout(row):
        bs=row['B'];N=row['N'];group=row['id']
        stages=[z for z in row['schedule'] if any(k.startswith(group+'-') and not k.startswith(group+'-prime') for k in z['tokens'])]
        assert stages, 'missing decode schedule'
        assert len(stages[0]['tokens'])==bs, ('marker batch split',bs,stages[0])
        assert all(v==1 for z in stages for v in z['tokens'].values()),'branch prefill recomputed'
        assert all(h['cached_tokens']==N for h in row['heads'].values())
        prime_stages=[z for z in row['schedule'] if any(k.startswith(group+'-prime') for k in z['tokens'])]
        if row['public_new_tokens']:
            assert len(prime_stages)==1,('public prefill split',prime_stages)
            assert sum(prime_stages[0]['tokens'].values())==row['public_new_tokens']
        assert all(len(z['tokens'])==bs for z in stages),('fixed batch size changed',bs)
        assert all(len(h['tokens'])==row['cap'] for h in row['heads'].values())
    # Exercise the native large-batch graphs and validate publication/cache shape.
    checks=[]
    for N in Ns:
        common=textids(N,9000)
        reset();rpc('sl_control',enabled=False,capture=False,verify=False,trace=False)
        refs={}
        for h in canonical:
            reset();refs[h]=run(common,[h],4,True,'validation-single',write=True)['heads'][h]['tokens']
        for bs in batches:
            heads=branch_names[:bs]
            reset();r=run(common,heads,4,True,'validation-native',write=True);check_layout(r)
            equal={}
            for i,h in enumerate(heads):
                original=canonical[i%8];end=endings[h][0]
                def trim(v):return v[:v.index(end)+1] if end in v else v
                equal[h]=trim(r['heads'][h]['tokens'])==trim(refs[original])
            # Graph parity is strict for each batch, including repeated aliases.
            reset();rpc('sl_control',enabled=True,capture=True,verify=True,trace=False)
            graph=run(common,heads,4,True,'validation-prefill-graph',write=True);check_layout(graph)
            assert {h:v['tokens'] for h,v in graph['heads'].items()}=={h:v['tokens'] for h,v in r['heads'].items()},'graph/eager token mismatch'
            checks.append({'N':N,'B':bs,'same_split_single_parity':equal,'graph_eager_token_parity':True,
                'layout_valid':True,'common_cached_tokens':N})
            rpc('sl_control',enabled=False,capture=False,verify=False,trace=False)
            save_json('validation.json',checks)
        receipt('validated-N'+str(N))
    rpc('sl_control',enabled=True,capture=True,verify=False,trace=False)
    for N in Ns:
        for bs in batches:
            for w in range(3):
                reset();run(textids(N,9100+w),branch_names[:bs],64,True,'warmup',write=True)
    receipt('warm-workers')
    # No graph capture, profiling or GPU-memory RPC in measured requests.
    rpc('sl_control',enabled=True,capture=False,verify=False,trace=False)
    rng=random.Random(1116)
    for N in [512,256,1024]:
        for cap in caps:
            for rep in range(a.repeats):
                order=list(batches);rng.shuffle(order)
                for bs in order:
                    for hit in [False,True]:
                        if not hit:reset()
                        row=run(textids(N,10000+rep),branch_names[:bs],cap,True,
                            f'measure-{N}-{cap}-{bs}-'+('fullhit' if hit else 'cold'))
                        check_layout(row)
                        if hit:assert row['public_cached_tokens']==N
                        else:assert row['public_cached_tokens']==0
                print('MEASURE',N,cap,rep+1,a.repeats,flush=True)
        receipt('measured-N'+str(N))
    # Round-robin batches under a sustained real benchmark workload, not dummy occupancy.
    for i in range(a.soak):
        bs=batches[i%len(batches)];reset()
        r=run(textids(512,20000+i),branch_names[:bs],64,True,'soak')
        check_layout(r)
    receipt('final-workers')
    summary={}
    for row in rows:
        if row['phase'].startswith('measure-'):
            summary.setdefault(row['phase'],[]).append(row)
    output=[]
    for key,rs in summary.items():
        group_tp=[statistics.median(h['tpot_host_ms'] for h in r['heads'].values()) for r in rs]
        decode_span=[max(h['last_ms'] for h in r['heads'].values())-min(h['first_ms'] for h in r['heads'].values()) for r in rs]
        aggregate=[r['B']*(r['cap']-1)*1000/span for r,span in zip(rs,decode_span)]
        output.append({'phase':key,'N':rs[0]['N'],'B':rs[0]['B'],'T':rs[0]['cap'],
            'n_requests':len(rs),'tpot_ms':stats(group_tp),'aggregate_decode_tps':stats(aggregate),
            'prime_ms':stats([r['prime_ms'] for r in rs]),'wall_ms':stats([r['wall_ms'] for r in rs]),
            'first_all_ms':stats([max(h['first_ms'] for h in r['heads'].values()) for r in rs]),
            'marker_stage_ms':stats([max(h['first_ms'] for h in r['heads'].values())-r['prime_ms'] for r in rs]),
            'per_head_tpot_ms':{h:stats([r['heads'][h]['tpot_host_ms'] for r in rs]) for h in rs[0]['heads']}})
    output.sort(key=lambda r:(r['N'],r['T'],r['phase'].endswith('fullhit'),r['B']))
    save_json('summary.json',output)
    save_json('completion.json',{'requests':len(rows),'validation_cells':len(checks),
       'parity_failed_cells':[{k:r[k] for k in ['N','B','same_split_single_parity']} for r in checks if not all(r['same_split_single_parity'].values())],
       'completed_unix':time.time(),'tools_executed':False})
    (R/'DONE').write_text('completed and verified batch/cache scheduling; see validation.json for numerical parity\n')
    for r in output:
        if r['N']==512 and r['T']==64 and r['phase'].endswith('cold'):
            print('RESULT',r['B'],r['tpot_ms']['p50'],r['aggregate_decode_tps']['p50'],flush=True)

if __name__=='__main__':main()
