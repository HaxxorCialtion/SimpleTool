"""Single-owner, TP4 exact-fork engine; never runs a generated tool."""
import dataclasses
import hashlib
import json
import time
import uuid
from pathlib import Path
from protocol import HEADS,ALL_HEADS,END,Invalid,require,number,scenario,dynamic,assemble,necessary,request_heads

class Runtime:
    def __init__(self,config):
        import torch,vllm
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.sampling_params import RequestOutputKind
        import prefill_runtime as rt
        self.config=config;self.rt=rt;rt.install();self.SP=SamplingParams;self.OK=RequestOutputKind
        self.T=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
        self.markers={h:self.T.encode('<'+h+'>',add_special_tokens=False) for h in ALL_HEADS}
        self.ends={h:self.T.encode('</'+h+'>',add_special_tokens=False) for h in ALL_HEADS}
        require(all(len(v)==1 for v in [*self.markers.values(),*self.ends.values()]),'head markers must be single tokens in this runtime')
        self.pad=self.T.encode(' .',add_special_tokens=False);require(len(self.pad)==1,'expected inert pad token')
        cfg=dict(enable_expert_parallel=config.get('enable_expert_parallel',False),all2all_backend=config.get('all2all_backend','allgather_reducescatter'),model=config['model'],dtype='bfloat16',kv_cache_dtype=config['kv_cache_dtype'],mamba_ssm_cache_dtype=config['mamba_ssm_cache_dtype'],
          tensor_parallel_size=config['tp'],distributed_executor_backend='mp' if config['tp']>1 else 'uni',
          worker_extension_cls='tp_worker.SimpleToolWorker',enable_prefix_caching=True,prefix_match_unit=1,
          prefix_caching_hash_algo='sha256',additional_config={'gdn_prefill_backend':config['gdn_backend']},
          max_model_len=config['max_model_len'],max_num_seqs=config['max_num_seqs'],max_num_batched_tokens=config['max_num_batched_tokens'],
          gpu_memory_utilization=config['gpu_memory_utilization'],kv_cache_memory_bytes=int(config['kv_gib']*2**30),
          enforce_eager=config['enforce_eager'],async_scheduling=False,limit_mm_per_prompt={'image':0,'video':0},
          disable_log_stats=True,seed=config['seed'],kernel_config={'enable_jit_warmup':False,'moe_backend':config['moe_backend']},
          compilation_config={'mode':3,'cudagraph_mode':'NONE' if config['enforce_eager'] else 'FULL_DECODE_ONLY',
           'cudagraph_capture_sizes':config['decode_capture_sizes'],'max_cudagraph_capture_size':max(config['decode_capture_sizes'])})
        self.llm=LLM(**cfg);self.eng=self.llm.llm_engine
        self.core=self.eng.engine_core.engine_core;self.sched=self.core.scheduler;self.executor=self.core.model_executor
        self.scenarios={};self.n=0;self.metrics={'requests':0,'errors':0}
        self.engine_config=cfg;self.versions={'torch':torch.__version__,'vllm':vllm.__version__,'cuda':torch.version.cuda}
        self.control(capture=False)

    def rpc(self,name,**kw):return self.executor.collective_rpc(name,kwargs=kw)
    def control(self,capture=False,verify=False):
        return self.rpc('sl_control',enabled=self.config['prefill_graphs'],capture=capture,verify=verify,trace=False)
    def _group(self):
        self.n+=1;return 'slrobot-'+str(self.n)+'-'+uuid.uuid4().hex[:8]
    def _cleanup(self,g,ids):
        if self.eng.has_unfinished_requests():self.eng.abort_request(ids)
        st=self.rt.GROUPS.pop(g,None)
        if st:self.rt.PRIMES.discard(st['prime_id'])
    def _deadline(self,deadline):
        if deadline is not None and time.monotonic()>deadline:raise TimeoutError('inference deadline exceeded; generated calls must not execute')

    def _prime(self,g,common,salt,deadline=None):
        rid=g+'-prime'
        self.eng.add_request(rid,{'prompt_token_ids':common,'cache_salt':salt},self.SP(temperature=0,max_tokens=1,ignore_eos=True,
          extra_args={'simpletool_prefill':{'group':g,'role':'prime','N':len(common)}}))
        req=self.sched.requests[self.rt.GROUPS[g]['prime_id']]
        managers=self.sched.kv_cache_manager.coordinator.single_type_managers;pool=managers[0].block_pool
        require(pool.hash_block_size==1,'runtime requires prefix_match_unit=1')
        terminal=pool.get_cached_block(req.block_hashes[-1],[m.kv_cache_group_id for m in managers])
        hit=0
        if terminal is not None:_,hit,_=self.sched.kv_cache_manager.get_computed_blocks(req,max_cache_hit_length=len(common))
        if hit==len(common):
            self.rt.GROUPS[g]['ready']=True;self.eng.abort_request([rid]);return hit
        result=None
        while self.eng.has_unfinished_requests():
            self._deadline(deadline)
            for o in self.eng.step():
                if o.request_id==rid and o.finished:result=o
        require(result is not None,'prime did not finish')
        require(not result.outputs or not result.outputs[0].token_ids,'prime generated tokens')
        return result.num_cached_tokens

    def _pin(self,common,salt):
        g=self._group();rid=g+'-prime'
        try:
            self._prime(g,common,salt)
            # A new request obtains references to the checkpoint after publication.
        finally:self._cleanup(g,[rid])
        g=self._group();rid=g+'-prime'
        try:
            self.eng.add_request(rid,{'prompt_token_ids':common,'cache_salt':salt},self.SP(temperature=0,max_tokens=1,ignore_eos=True,
                extra_args={'simpletool_prefill':{'group':g,'role':'prime','N':len(common)}}))
            req=self.sched.requests[self.rt.GROUPS[g]['prime_id']]
            cache,hit,_=self.sched.kv_cache_manager.get_computed_blocks(req,max_cache_hit_length=len(common))
            require(hit==len(common),'fixed prefix checkpoint could not be pinned')
            pool=self.sched.kv_cache_manager.coordinator.single_type_managers[0].block_pool
            blocks=list({b.block_id:b for group in cache.blocks for b in group if not b.is_null}.values())
            pool.touch(blocks)
            return pool,blocks
        finally:self._cleanup(g,[rid])

    def add_scenario(self,body):
        spec=scenario(body);sid=spec['id']
        require(sid not in self.scenarios,'scenario already exists; delete it explicitly before replacing')
        require(len(self.scenarios)<self.config['max_scenarios'],'max_scenarios reached')
        prefix=self.T.encode(spec['prefix'],add_special_tokens=False)
        require(len(prefix)+spec['max_suffix_tokens']+max([spec['max_tokens'],*spec['head_max_tokens'].values()])+1<=self.config['max_model_len'],
                'scenario maximum token budget exceeds max_model_len')
        require(len(spec['heads'])<=self.config['max_num_seqs'],'scenario heads exceed max_num_seqs')
        spec['prefix_ids']=prefix;spec['cache_salt']='scenario-'+sid+'-'+spec['revision'];spec['warmup_results']=[]
        pin=self._pin(prefix,spec['cache_salt']);spec['_pin']=pin
        self.scenarios[sid]=spec
        try:
            if spec['warmup_observations']:
                self.control(capture=True,verify=self.config['verify_warmup'])
                for obs in spec['warmup_observations']:
                    r=self.infer({'scenario_id':sid,'environment':obs,'max_tokens':spec['warmup_cap']},warming=True)
                    spec['warmup_results'].append({'latency_ms':r['timings_ms'],'cache':r['cache'],'valid':r['valid']})
        except BaseException:
            self.remove_scenario(sid);raise
        finally:self.control(capture=False)
        return self.describe(spec)

    def describe(self,s):
        return {k:s[k] for k in ['id','revision','tools','heads','suffix_tokens','max_suffix_tokens','max_tokens','head_max_tokens','temperature','top_p','top_k','seed','stop_unused_heads','warmup_results']}|{'prefix_tokens':len(s['prefix_ids']),'prefix_sha256':hashlib.sha256(s['prefix'].encode()).hexdigest()}
    def list_scenarios(self):return [self.describe(s) for s in self.scenarios.values()]
    def remove_scenario(self,sid):
        require(sid in self.scenarios,'unknown scenario_id')
        s=self.scenarios.pop(sid);pool,blocks=s['_pin'];pool.free_blocks(blocks)
        return {'deleted':sid,'note':'checkpoint references released; cached blocks remain evictable; bounded captured graphs are retained'}

    def _input(self,s,body):
        text=dynamic(s,body);end=self.T.encode(END,add_special_tokens=False)
        if s['suffix_tokens'] is not None:
            ids=self.T.encode(text+'\nDiagnostic padding:\n',add_special_tokens=False)
            padding=s['suffix_tokens']-len(ids)-len(end)
            require(padding>=0,'observation exceeds suffix_tokens; increase scenario budget, no truncation performed')
            ids+=self.pad*padding+end
        else:ids=self.T.encode(text,add_special_tokens=False)+end;padding=0
        require(len(ids)<=s['max_suffix_tokens'],'observation exceeds max_suffix_tokens')
        return s['prefix_ids']+ids,len(ids),padding

    def infer(self,body,deadline=None,warming=False):
        start=time.perf_counter();sid=body.get('scenario_id');require(sid in self.scenarios,'unknown scenario_id')
        s=self.scenarios[sid]
        require(body.get('expected_revision',s['revision'])==s['revision'],'scenario revision mismatch')
        heads=request_heads(s,body)
        require(len(heads)<=self.config['max_num_seqs'],'requested heads exceed max_num_seqs')
        common,suffix,padding=self._input(s,body);tokenize_ms=(time.perf_counter()-start)*1000
        cap=number(body.get('max_tokens',s['max_tokens']),1,1024,'max_tokens',True)
        hcaps=body.get('head_max_tokens',s['head_max_tokens']);require(type(hcaps) is dict and not set(hcaps)-set(s['heads']+['content']),'invalid head_max_tokens')
        for h,n in hcaps.items():number(n,1,1024,h+' max_tokens',True)
        require(len(common)+1+max(hcaps.get(h,cap) for h in heads)<=self.config['max_model_len'],'context plus generation budget exceeds max_model_len')
        temp=number(body.get('temperature',s['temperature']),0,2,'temperature')
        top_p=number(body.get('top_p',s['top_p']),.01,1,'top_p')
        top_k=number(body.get('top_k',s['top_k']),-1,100000,'top_k',True);require(top_k!=0,'top_k cannot be zero')
        seed=number(body.get('seed',s['seed']),0,2**31-1,'seed',True)
        g=self._group();prime=g+'-prime';ids={g+'-'+h:h for h in heads}
        first={};last={};last_seen={};finished={};cancelled=[];ready=None;relevant=None
        t=time.perf_counter();self.rt.EVENTS.clear()
        try:
            hit=self._prime(g,common,s['cache_salt'],deadline);prime_ms=(time.perf_counter()-t)*1000
            require(hit>=len(s['prefix_ids']),'pinned prefix was not reused')
            for rid,h in ids.items():
                self.eng.add_request(rid,{'prompt_token_ids':common+self.markers[h],'cache_salt':s['cache_salt']},self.SP(
                  temperature=temp,top_p=top_p,top_k=top_k,seed=seed,max_tokens=hcaps.get(h,cap),
                  stop_token_ids=self.ends[h],ignore_eos=False,skip_special_tokens=False,include_stop_str_in_output=True,
                  output_kind=self.OK.CUMULATIVE,extra_args={'simpletool_prefill':{'group':g,'role':'branch','N':len(common)}}))
            while self.eng.has_unfinished_requests():
                self._deadline(deadline)
                oo=self.eng.step();now=(time.perf_counter()-t)*1000
                for o in oo:
                    if o.request_id not in ids:continue
                    h=ids[o.request_id];v=o.outputs[0] if o.outputs else None;tokens=list(v.token_ids) if v else []
                    if tokens:first.setdefault(h,now)
                    last_seen[h]=tokens
                    if o.finished:
                        raw=self.T.decode(tokens,skip_special_tokens=False)
                        closed=bool(tokens and tokens[-1]==self.ends[h][0])
                        finished[h]={'text':raw.removesuffix('</'+h+'>'),'raw':raw,'tokens':tokens,'closed':closed,
                          'finish_reason':v.finish_reason if v else None,'stop_reason':v.stop_reason if v else None,
                          'cached_tokens':o.num_cached_tokens,'first_token_ms':first.get(h),'finished_ms':now,
                          'tpot_ms':(now-first[h])/(len(tokens)-1) if len(tokens)>1 else None}
                        require(o.num_cached_tokens==len(common),'branch did not reuse the complete common checkpoint')
                if 'function' in finished:
                    relevant=necessary(s,finished['function']['text'])
                    if relevant is None:relevant=['function']
                    if s['stop_unused_heads'] and not warming:
                        stop=[rid for rid,h in ids.items() if h!='content' and h not in relevant and h not in finished and h not in cancelled]
                        if stop:self.eng.abort_request(stop);cancelled.extend(ids[rid] for rid in stop)
                    if ready is None and all(h in finished for h in relevant):ready=(time.perf_counter()-t)*1000
            decoded={h:v['text'] for h,v in finished.items()}
            call,errors=assemble(s,decoded)
            for h in relevant or s['heads']:
                if h not in finished or not finished[h]['closed']:errors.append(h+': missing closing token (EOS or cap)')
            content=finished.get('content')
            content_errors=[]
            if 'content' in heads and (content is None or not content['closed']):
                content_errors.append('content: missing closing token (EOS or cap)')
            self.metrics['requests']+=1;self.metrics['errors']+=bool(errors)
            return {'scenario_id':sid,'revision':s['revision'],'request_id':body.get('request_id',g),'call':call if not errors else None,
              'candidate_call':call,'valid':not errors,'errors':errors,'heads':finished,'cancelled_heads':cancelled,
              'necessary_heads':relevant,'requested_heads':heads,'content_generated':content is not None,
              'content':content['text'] if content is not None else None,
              'content_valid':not content_errors if 'content' in heads else None,'content_errors':content_errors,'tools_executed':False,
              'tokens':{'prefix':len(s['prefix_ids']),'suffix':suffix,'common':len(common),'padding':padding,'head_markers':len(ids)},
              'cache':{'public_cached_tokens':hit,'public_new_tokens':len(common)-hit,'branch_cached_tokens':{h:v['cached_tokens'] for h,v in finished.items()},'boundary_pinned':True},
              'timings_ms':{'tokenize':tokenize_ms,'public_prefill_and_publish':prime_ms,'tools_ready':ready,
                'content_finished':content['finished_ms'] if content is not None else None,
                'content_ready':content['finished_ms'] if content is not None and content['closed'] else None,
                'engine_all_drain':(time.perf_counter()-t)*1000,'service_compute':(time.perf_counter()-start)*1000},
              'sampling':{'temperature':temp,'top_p':top_p,'top_k':top_k,'seed':seed,'max_tokens':cap,'head_max_tokens':hcaps},'warming':warming}
        finally:self._cleanup(g,[prime,*ids])

    def receipt(self):
        return {'versions':self.versions,'engine_config':self.engine_config,'metrics':self.metrics,
          'scenarios':self.list_scenarios(),'workers':self.rpc('sl_receipt'),
          'limits':{'max_prefill_graphs_per_rank':self.config['max_prefill_graphs'],'max_scenarios':self.config['max_scenarios'],
                    'logical_request_concurrency':1,'async_scheduling':False,'no_content':False,
                    'optional_content':True,'include_content_default':False}}
