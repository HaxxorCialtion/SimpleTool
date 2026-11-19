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
