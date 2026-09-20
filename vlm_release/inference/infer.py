#!/usr/bin/env python3
"""Image/text SimpleTool inference with auditable generated-only branch prefixes.

Latency is wall time of blocking vLLM calls, not time-to-first-token. Parallel
means one engine.generate batch of seven independent AR sequences. Prefix cache
is enabled by default but cache-hit or one-prefill behavior is not assumed.
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

try:
    from .protocol import HEADS, SPECIAL, ProtocolError, active_heads, content_prefix, head_prefixes, strict_decode, tool_schemas
except ImportError:
    from protocol import HEADS, SPECIAL, ProtocolError, active_heads, content_prefix, head_prefixes, strict_decode, tool_schemas


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mode_for(row, mode):
    value = row.get('decode_mode', 'adaptive') if mode == 'dataset' else mode
    value = 'adaptive' if value == 'auto' else value
    if value not in ('direct', 'adaptive'):
        raise ValueError('Unsupported decode mode: '+str(value))
    return value


def prepare_input(processor, row, data_dir, tool_template="native"):
    """Read input fields only. No precomputed row.prompt or target is trusted."""
    from PIL import Image
    messages=copy.deepcopy(row['messages']); tool_schemas(row['tools'])
    paths=[]
    for message in messages:
        if isinstance(message.get('content'), list):
            for part in message['content']:
                if part.get('type') in ('image','image_url'):
                    value=part.get('image') if part['type']=='image' else part['image_url']['url']
                    path=Path(value)
                    if not path.is_absolute():path=data_dir/path
                    path=path.resolve(); paths.append(path)
                    part.clear();part.update(type='image',image=str(path))
    if row.get('image_paths') is not None:
        declared=[(Path(p) if Path(p).is_absolute() else data_dir/Path(p)).resolve() for p in row['image_paths']]
        if paths!=declared:raise ValueError('Message and declared image order differ')
    pictures=[]; hashes=[]; sizes=[]
    for path in paths:
        with Image.open(path) as image:
            pictures.append(image.convert('RGB').copy());sizes.append(list(image.size))
        hashes.append(sha(path))
    if row.get('image_sha256') is not None and hashes!=row['image_sha256']:
        raise ValueError('Image hash mismatch')
    if tool_template not in ('native', 'legacy'):
        raise ValueError('Unknown tool template: '+tool_template)
    template_kwargs = {'tools': row['tools']} if tool_template == 'native' else {}
    if tool_template == 'legacy':
        tool_text = '# Available Tools\n' + '\n'.join(json.dumps(t, ensure_ascii=False) for t in row['tools'])
        system = next((m for m in messages if m['role'] == 'system'), None)
        if system is None:
            messages.insert(0, {'role': 'system', 'content': tool_text})
        elif isinstance(system.get('content'), str):
            system['content'] += '\n\n' + tool_text
        else:
            raise ValueError('Legacy tool template requires a text system message')
    prompt=processor.apply_chat_template(messages,**template_kwargs,tokenize=False,
                                          add_generation_prompt=True,enable_thinking=False)
    return dict(prompt=prompt,pictures=pictures,image_paths=list(map(str,paths)),image_sha256=hashes,image_sizes=sizes)


def request(prefix, pictures):
    value={'prompt':prefix}
    if pictures:value['multi_modal_data']={'image':pictures}
    return value


def percentile(values, fraction):
    if not values:return None
    ordered=sorted(values); pos=(len(ordered)-1)*fraction
    lo=int(pos);hi=min(lo+1,len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo)


def timing(values):
    return {'count':len(values),'p50_seconds':percentile(values,.5),'p95_seconds':percentile(values,.95)}


def run_prediction(engine, sampling_cls, tokenizer, inp, mode, execution, content_tokens, head_tokens, event_callback=None):
    records={}; times={}; prefixes={}; started=time.perf_counter()
    def generate(names, ps, max_tokens):
        params=[]
        for name in names:
            stop=f'</{name}>'; ids=tokenizer.encode(stop,add_special_tokens=False)
            if len(ids)!=1:raise ValueError('Closing tag must be one registered token: '+stop)
            params.append(sampling_cls(temperature=0,max_tokens=max_tokens,stop_token_ids=ids,
                                      skip_special_tokens=False,include_stop_str_in_output=True))
        if event_callback:event_callback('generation_started', {'branches':names})
        begin=time.perf_counter()
        outputs=engine.generate([request(p,inp['pictures']) for p in ps],params,use_tqdm=False)
        duration=time.perf_counter()-begin
        for name,prefix,result,param in zip(names,ps,outputs,params):
            output=result.outputs[0]
            records[name]={'text':output.text,'token_ids':list(output.token_ids),
                           'finish_reason':output.finish_reason,'stop_reason':output.stop_reason,
                           'requested_stop_token_ids':list(param.stop_token_ids),'max_tokens':max_tokens,
                           'prompt_token_count':len(result.prompt_token_ids or [])}
            prefixes[name]=prefix
        if event_callback:event_callback('generation_completed', {'branches':{name:records[name]['text'] for name in names},'seconds':duration})
        return duration
    content=None
    if mode=='adaptive':
        times['content_seconds']=generate(['content'],[content_prefix(inp['prompt'],mode)],content_tokens)
        content=records['content']['text']
    else:times['content_seconds']=0.0
    try:
        active=active_heads(inp['tools'])
        heads=head_prefixes(inp['prompt'],mode,content,active)
    except ProtocolError as exc:
        return dict(legal=False,error=str(exc),raw=records,prefixes=prefixes,timings={**times,'generation_seconds':time.perf_counter()-started})
    if execution=='parallel':
        times['heads_seconds']=generate(list(active),list(heads.values()),head_tokens)
    else:
        times['heads_seconds']=sum(generate([name],[heads[name]],head_tokens) for name in active)
    result=dict(raw=records,prefixes=prefixes,timings={**times,'generation_seconds':time.perf_counter()-started})
    try:
        fn,args=strict_decode({h:records[h]['text'] for h in active},inp['tools'])
        result.update(legal=True,function=fn,arguments=args)
    except ProtocolError as exc:result.update(legal=False,error=str(exc))
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',required=True);ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True,help='New empty output directory')
    ap.add_argument('--mode',choices=['direct','adaptive','dataset'],default='dataset')
    ap.add_argument('--head-execution',choices=['parallel','serial'],default='parallel')
    ap.add_argument('--tool-template',choices=['native','legacy'],default='native')
    ap.add_argument('--limit',type=int);ap.add_argument('--cpu-contract-only',action='store_true')
    ap.add_argument('--max-model-len',type=int,default=8192)
    ap.add_argument('--content-max-tokens',type=int,default=96);ap.add_argument('--head-max-tokens',type=int,default=128)
    ap.add_argument('--gpu-memory-utilization',type=float,default=.8)
    ap.add_argument('--tensor-parallel-size',type=int,default=1)
    ap.add_argument('--disable-prefix-cache',action='store_true')
    args=ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):raise SystemExit('Output must be new/empty')
    args.output.mkdir(parents=True,exist_ok=True)
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(args.model,local_files_only=True)
    tokenizer=processor.tokenizer
    special={t:tokenizer.encode(t,add_special_tokens=False) for t in SPECIAL}
    if any(len(v)!=1 for v in special.values()) or len({v[0] for v in special.values()})!=len(special):
        raise ValueError('Model tokenizer lacks distinct registered SimpleTool tokens')
    rows=[json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if args.limit is not None:rows=rows[:args.limit]
    if not rows:raise ValueError('No records selected')
    prepared=[]
    for row in rows:
        inp=prepare_input(processor,row,args.data.parent,args.tool_template);inp['tools']=row['tools'];prepared.append(inp)
    init=0.0;engine=None;SamplingParams=None;version=None
    if not args.cpu_contract_only:
        from vllm import LLM,SamplingParams,__version__
        version=__version__;start=time.perf_counter()
        kwargs=dict(model=args.model,dtype='bfloat16',max_model_len=args.max_model_len,
                    tensor_parallel_size=args.tensor_parallel_size,gpu_memory_utilization=args.gpu_memory_utilization,
                    enforce_eager=True,enable_prefix_caching=not args.disable_prefix_cache,max_num_seqs=16)
        max_images=max(len(p['pictures']) for p in prepared)
        if max_images:kwargs['limit_mm_per_prompt']={'image':max_images}
        engine=LLM(**kwargs);init=time.perf_counter()-start
    results=[]
    with (args.output/'predictions.jsonl').open('w') as stream:
        for index,(row,inp) in enumerate(zip(rows,prepared)):
            mode=mode_for(row,args.mode);begin=time.perf_counter()
            common={k:inp[k] for k in ['prompt','image_paths','image_sha256','image_sizes']}
            if args.cpu_contract_only:
                # Empty generated content is a contract probe, never a quality prediction.
                ps=head_prefixes(inp['prompt'],mode,'</content>' if mode=='adaptive' else None,
                                 active_heads(inp['tools']))
                probes=([content_prefix(inp['prompt'],mode)] if mode=='adaptive' else [])+list(ps.values())
                lengths=[]
                for prefix in probes:
                    batch=processor(text=[prefix],**({'images':inp['pictures']} if inp['pictures'] else {}),return_tensors='pt')
                    if inp['pictures'] and ('pixel_values' not in batch or len(batch['image_grid_thw'])!=len(inp['pictures'])):
                        raise ValueError('Images not encoded into processor inputs')
                    lengths.append(int(batch['input_ids'].shape[-1]))
                budget=max(lengths)+args.content_max_tokens+args.head_max_tokens
                if budget>args.max_model_len:raise ValueError('Input exceeds conservative generation token budget')
                prediction=dict(contract_passed=True,probe_prefixes=ps,prefix_token_lengths=lengths,
                                conservative_token_budget=budget,quality_prediction=False)
            else:
                prediction=run_prediction(engine,SamplingParams,tokenizer,inp,mode,args.head_execution,
                                          args.content_max_tokens,args.head_max_tokens)
                prediction['exact_tool']=prediction.get('function')==row['function']
                prediction['exact_arguments']=prediction.get('arguments')==row['arguments']
                prediction['exact_call']=prediction.get('legal',False) and prediction['exact_tool'] and prediction['exact_arguments']
            prediction.update(id=row.get('id',str(index)),mode=mode,head_execution=args.head_execution,
                              first_record=index==0,input=common,record_seconds=time.perf_counter()-begin)
            stream.write(json.dumps(prediction,ensure_ascii=False)+'\n');stream.flush();results.append(prediction)
    warm=results[1:]
    summary=dict(cpu_contract_only=args.cpu_contract_only,records=len(results),model=args.model,vllm_version=version,
                 data_sha256=sha(args.data),initialization_seconds=init,mode=args.mode,head_execution=args.head_execution,tool_template=args.tool_template,
                 prefix_caching_enabled=not args.disable_prefix_cache,
                 cache_caveat='Cache enabled is not proof of a cache hit or exactly one prefill. Serial warms cache differently.',
                 latency_scope='Blocking engine calls; preparation before evaluation excluded. First record reported separately.',
                 original_image_sizes_preserved=True,native_thinking=False)
    if not args.cpu_contract_only:
        summary.update(legal_rate=sum(r['legal'] for r in results)/len(results),
                       exact_call_rate=sum(r['exact_call'] for r in results)/len(results),
                       exact_tool_rate=sum(r['exact_tool'] for r in results)/len(results),
                       exact_arguments_rate=sum(r['exact_arguments'] for r in results)/len(results),
                       cold_first_record=results[0]['timings'],
                       warm_timings={key:timing([r['timings'][key] for r in warm if key in r['timings']])
                                     for key in ['content_seconds','heads_seconds','generation_seconds']})
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
