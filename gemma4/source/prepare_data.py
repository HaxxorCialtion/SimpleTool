"""Deterministic 5% mixture; canonical tool signatures and native Gemma branches.

CPU workers tokenize bounded batches; arrays are mmap-compatible, never padded.
Normal packs target 4096; complete longer branches are retained alone up to 8192.
"""
import argparse
import bisect
from collections import Counter
import concurrent.futures
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from tokenizers import Tokenizer
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('SIMPLETOOL_TRAINING_DATA', 'path/to/original-jsonl-pool'))
SOURCES=[('09_train_ready.jsonl',402137),('09b_train_ready.jsonl',988712),('final_simplelove.jsonl',115110)]
HEADS=('content','function','arg1','arg2','arg3','arg4','arg5','arg6')
SPECIAL_TOKENS=tuple(t for h in HEADS for t in (f'<{h}>',f'</{h}>'))+('<|null|>',)
SEED=20261108
THOUGHT='<|channel>thought\n<channel|>'
MAX=8192
TARGET=4096

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16<<20),b''):h.update(block)
    return h.hexdigest()

def normalize_schema(schema):
    result=copy.deepcopy(schema)
    if isinstance(result,dict):
        for k,v in list(result.items()):
            if isinstance(v,dict):result[k]=normalize_schema(v)
            elif isinstance(v,list):result[k]=[normalize_schema(x) if isinstance(x,dict) else x for x in v]
        if 'properties' in result:
            props=result['properties'];required=result.get('required',[])
            assert isinstance(props,dict) and isinstance(required,list)
            assert len(set(required))==len(required) and set(required)<=set(props),'invalid required signature'
            order=required+sorted(set(props)-set(required))
            result['properties']={k:result['properties'][k] for k in order}
    return result

def canonical_tools(tools):
    normalized=copy.deepcopy(tools)
    for t in normalized:
        f=t.get('function',t)
        f['parameters']=normalize_schema(f.get('parameters',{'type':'object','properties':{}}))
    return normalized

def body(value,head):
    a,b=f'<{head}>',f'</{head}><|im_end|>'
    assert isinstance(value,str) and value.startswith(a) and value.endswith(b),('malformed head',head)
    return value[len(a):-len(b)]

def normalize_row(row,name):
    prompt=row.get('prompt') or row.get('prefix') or ''
    assert prompt.endswith('<|im_start|>assistant\n'),'assistant prefix'
    assert not any(t in prompt or any(t in str(row.get(f'head_{i}_{h}','')) for i,h in enumerate(HEADS))
                   for t in ['<think>','</think>','<thinking>','</thinking>','<|channel>']), 'thinking/protocol remnants'
    # Parse one original ChatML shell, preserving every role and message body.
    messages=[];position=0
    pattern=re.compile(r'<\|im_start\|>([^\n]+)\n(.*?)<\|im_end\|>\n?',re.S)
    for match in pattern.finditer(prompt):
        assert match.start()==position,'unparsed ChatML'
        role,content=match.groups();assert role in {'system','user','assistant','tool'},role
        messages.append((role,content));position=match.end()
    assert prompt[position:]=='<|im_start|>assistant\n' and messages and messages[-1][0] in {'user','tool'}
    tools=row['tools'];assert isinstance(tools,list) and tools
    names=[t.get('function',t)['name'] for t in tools];assert len(set(names))==len(names),'duplicate tool'
    normalized=canonical_tools(tools)
    by_name={t.get('function',t)['name']:t for t in normalized}
    original={t.get('function',t)['name']:t.get('function',t) for t in tools}
    # Replace complete JSON tool declarations, never unrelated prose/dates.
    found=set();decoder=json.JSONDecoder();new_messages=[]
    for role,content in messages:
        if role=='system':
            replacements=[];cursor=0
            while cursor<len(content):
                start=content.find('{',cursor)
                if start<0:break
                try:obj,end=decoder.raw_decode(content[start:])
                except json.JSONDecodeError:cursor=start+1;continue
                f=obj.get('function',{}) if isinstance(obj,dict) else {}
                tool_name=f.get('name')
                if tool_name in by_name:
                    assert obj==tools[names.index(tool_name)],'prompt/metadata tool mismatch'
                    replacements.append((start,start+end,json.dumps(by_name[tool_name],ensure_ascii=False,separators=(',',':'))))
                    found.add(tool_name)
                cursor=start+end
            for a,b,value in reversed(replacements):content=content[:a]+value+content[b:]
        new_messages.append((role,content))
    assert found==set(names),('tools missing in actual prompt',set(names)-found)
    targets={h:body(row[f'head_{i}_{h}'],h) for i,h in enumerate(HEADS) if row.get(f'head_{i}_{h}')}
    assert 'function' in targets
    function=targets['function']
    remapped=False
    if function in original:
        old=list(original[function].get('parameters',{}).get('properties',{}))
        new=list(by_name[function].get('function',by_name[function])['parameters'].get('properties',{}))
        assert len(new)<=6,'selected function exceeds six parameter heads'
        values={k:targets.get(f'arg{i+1}') for i,k in enumerate(old)}
        assert all(values[k] is not None for k in old),'missing positional argument head'
        for i,key in enumerate(new):targets[f'arg{i+1}']=values[key]
        # The selected declaration determines which heads are meaningful.
        for i in range(len(new)+1,7):targets.pop(f'arg{i}',None)
        remapped=old!=new
    else:
        assert function=='<|null|>','selected function not declared'
        for i in range(1,7):targets.pop(f'arg{i}',None)
    if '<continue>' in targets.get('content',''):targets.pop('content')
    prefix='<bos>'+''.join('<|turn>'+('model' if r=='assistant' else r)+'\n'+c+'<turn|>\n' for r,c in new_messages)
    prefix+='<|turn>model\n'+THOUGHT
    assert '<|im_start|>' not in prefix and '<|im_end|>' not in prefix,'nested ChatML in native prompt'
    return prefix,targets,{'id':name,'tools':normalized,'selected_function':function,
                           'binding_order':list(by_name[function].get('function',by_name[function])['parameters'].get('properties',{})) if function in by_name else [],
                           'remapped':remapped}

RAW=None
OUT=None
def setup(tokenizer,out,thought=THOUGHT):
    global RAW,OUT,THOUGHT
    THOUGHT=thought
    RAW=Tokenizer.from_file(str(Path(tokenizer)/'tokenizer.json'));RAW.no_padding();RAW.no_truncation();OUT=Path(out)

def worker(task):
    number,rows=task;shard=f'shard-{number:05d}'
    texts=[];entries=[];examples=[];errors=[];bindings=[];heads=Counter();reorders=0
    for name,row in rows:
        try:prefix,targets,signature=normalize_row(row,name)
        except (AssertionError,KeyError,ValueError,TypeError) as error:
            errors.append({'id':name,'reason':str(error)});continue
        reorders+=int(signature['remapped']);bindings.append(signature)
        for head in HEADS:
            if head not in targets:continue
            p=prefix+f'<{head}>';t=targets[head]+f'</{head}><turn|>'
            entries.append((name+':'+head,len(texts),len(texts)+1));texts.extend([p,t])
            if len(examples)<2:examples.append({'id':name+':'+head,'prompt':p,'completion':t,'signature':signature})
    encoded=RAW.encode_batch(texts,add_special_tokens=False) if texts else []
    segments=[];long=[];maxlen=0
    for name,p,t in entries:
        a,b=encoded[p].ids,encoded[t].ids;n=len(a)+len(b);maxlen=max(maxlen,n)
        if n>MAX:long.append({'id':name,'tokens':n,'reason':'complete branch exceeds 8192; excluded without truncation'});continue
        head=name.rsplit(':',1)[1]
        assert a[-1]==RAW.token_to_id(f'<{head}>') and b[-2:]==[RAW.token_to_id(f'</{head}>'),RAW.token_to_id('<turn|>')]
        if len(segments)<2:assert RAW.encode(texts[p]+texts[t],add_special_tokens=False).ids==a+b
        ids=np.asarray(a+b,dtype=np.int32);mask=np.zeros(n,dtype=np.uint8);mask[len(a):]=1
        segments.append((ids,mask,name));heads[head]+=1
    assert segments,'shard has no valid segments'
    free=[];bins=[]
    for i in sorted(range(len(segments)),key=lambda j:-len(segments[j][0])):
        n=len(segments[i][0]);j=bisect.bisect_left(free,(n,-1))
        if j<len(free):remaining,k=free.pop(j);bins[k].append(i);bisect.insort(free,(remaining-n,k))
        else:
            k=len(bins);bins.append([i])
            if n<TARGET:bisect.insort(free,(TARGET-n,k))
    path=OUT/shard;path.mkdir(exist_ok=False)
    arrays=[];masks=[];metadata=[];offset=0;probe=[]
    for group in bins:
        cuts=[0];names=[];supervised=0
        for i in group:
            x,m,name=segments[i];arrays.append(x);masks.append(m);cuts.append(cuts[-1]+len(x));names.append(name);supervised+=int(m.sum())
        metadata.append({'range':[offset,offset+cuts[-1]],'cuts':cuts,'sources':names});offset+=cuts[-1]
        probe.append((cuts[-1],supervised))
    np.save(path/'ids.npy',np.concatenate(arrays));np.save(path/'mask.npy',np.concatenate(masks))
    (path/'packs.json').write_text(json.dumps(metadata))
    with (path/'bindings.jsonl').open('w') as stream:
        for record in bindings:stream.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n')
    receipt={'name':shard,'packs':len(bins),'rows':len(rows),'valid_rows':len(rows)-len(errors),'branches':dict(heads),
             'tokens':offset,'supervised_tokens':sum(p[1] for p in probe),'max_branch_tokens':maxlen,
             'remapped_rows':reorders,'errors':errors,'oversize':long,'probe':probe,
             'sha256':{f:sha(path/f) for f in ['ids.npy','mask.npy','packs.json','bindings.jsonl']}}
    write(path/'RECEIPT.json',receipt)
    return receipt,examples

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--limit',type=int)
    parser.add_argument('--model',required=True,choices=['e2b','e4b','12b','26b_a4b','31b'])
    args=parser.parse_args();started=time.time()
    configs={args.model:json.loads((ROOT/'configs'/f'{args.model}.json').read_text())}
    primary=ROOT/'data'/args.model;primary.mkdir(parents=True,exist_ok=True)
    assert not (primary/'MANIFEST.json').exists(),'data already prepared'
    tok=AutoTokenizer.from_pretrained(configs[args.model]['model_dir'],local_files_only=True)
    base_len=len(tok);assert base_len==262144
    backend=tok.backend_tokenizer.to_str()
    thought_suffix=THOUGHT if args.model in ('12b','26b_a4b','31b') else ''
    for name,cfg in configs.items():
        other=AutoTokenizer.from_pretrained(cfg['model_dir'],local_files_only=True)
        assert other.backend_tokenizer.to_str()==backend,('tokenizer differs; independent encoding required',name)
        if other.chat_template:
            text=other.apply_chat_template([{'role':'system','content':'SYS'},{'role':'user','content':'USR'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            assert text=='<bos><|turn>system\nSYS<turn|>\n<|turn>user\nUSR<turn|>\n<|turn>model\n'+thought_suffix,(name,text)
    tok.add_tokens(list(SPECIAL_TOKENS),special_tokens=True)
    existing=list(getattr(tok,'additional_special_tokens',getattr(tok,'extra_special_tokens',[])))
    tok.add_special_tokens({'additional_special_tokens':list(dict.fromkeys([*map(str,existing),*SPECIAL_TOKENS]))})
    ids={t:tok.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS};assert list(ids.values())==list(range(base_len,base_len+17))
    for t,i in ids.items():assert tok.encode(t,add_special_tokens=False)==[i]
    tok.save_pretrained(primary/'tokenizer')
    receipts=[];examples=[];sources=[];pending=set();buffer=[];shard=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=setup,initargs=(str(primary/'tokenizer'),str(primary),thought_suffix)) as pool:
        def collect():
            nonlocal pending,examples
            done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                rec,ex=future.result();receipts.append(rec)
                if len(examples)<8:examples+=ex
                print(json.dumps({'shard':rec['name'],'rows':rec['rows'],'errors':len(rec['errors']),'branches':sum(rec['branches'].values()),'tokens':rec['tokens'],'seconds':round(time.time()-started)}),flush=True)
        for source,total in SOURCES:
            order=np.random.default_rng(SEED+len(sources)).permutation(total)
            selected=set(map(int,order[:math.ceil(total*.05)]))
            if args.limit:selected=set(sorted(selected)[:args.limit])
            digest=hashlib.sha256()
            with (SOURCE/source).open('rb') as stream:
                for i,line in enumerate(stream):
                    digest.update(line)
                    if i not in selected:continue
                    buffer.append((f'{source}:{i}',json.loads(line)))
                    if len(buffer)==128:
                        pending.add(pool.submit(worker,(shard,buffer)));buffer=[];shard+=1
                        if len(pending)>=args.workers*2:collect()
            assert i+1==total,(source,i+1,total)
            sources.append({'path':str(SOURCE/source),'rows':total,'selected_rows':len(selected),'sha256':digest.hexdigest(),'selected_line_numbers_zero_based':sorted(selected)})
        if buffer:pending.add(pool.submit(worker,(shard,buffer)))
        while pending:collect()
    receipts.sort(key=lambda s:s['name'])
    errors=[e for s in receipts for e in s['errors']];long=[e for s in receipts for e in s['oversize']]
    write(ROOT/'receipts'/f'SOURCE_AUDIT_{args.model}.json',{'selected':sum(s['selected_rows'] for s in sources),'errors':errors,'oversize_branches':long,'remapped_rows':sum(s['remapped_rows'] for s in receipts)})
    # Fail closed if a material protocol mismatch affects the mixture.
    assert len(errors)<sum(s['selected_rows'] for s in sources)*.01,('more than 1% rows quarantined',len(errors))
    probes=[];count=0
    for s in receipts:
        probes.extend((n,sup,count+i) for i,(n,sup) in enumerate(s.pop('probe')));count+=s['packs']
    probe_indices=list(dict.fromkeys([max(probes,key=lambda p:p[0])[2],max(probes,key=lambda p:p[1])[2]]))
    index={'fraction':.05,'seed':SEED,'pack_length':MAX,'target_pack_length':TARGET,'sources':sources,'shards':receipts}
    write(primary/'INDEX.json',index)
    manifest={'valid':True,'fraction':.05,'seed':SEED,'epochs':1,'head_ids':{h:ids[f'<{h}>'] for h in HEADS},'special_token_ids':ids,
              'base_vocab':base_len,'extended_vocab':len(tok),'end_token_id':106,'pack_length':MAX,'target_pack_length':TARGET,
              'selected_rows':sum(s['selected_rows'] for s in sources),'valid_rows':sum(s['valid_rows'] for s in receipts),
              'branches':dict(sum((Counter(s['branches']) for s in receipts),Counter())),
              'packs':count,'real_tokens':sum(s['tokens'] for s in receipts),'supervised_tokens':sum(s['supervised_tokens'] for s in receipts),
              'data_sha256':sha(primary/'INDEX.json'),'tokenizer_sha256':sha(primary/'tokenizer/tokenizer.json'),
              'base_backend_tokenizer_sha256':hashlib.sha256(backend.encode()).hexdigest(),
              'independently_tokenized_for_this_model':True,'max_retained_branch_tokens':max(n for n,_,_ in probes),
              'probe_pack_indices':probe_indices,'thinking':'no-thinking raw rendered template verified',
              'thought_suffix':thought_suffix,
              'signature':'required in required-list order; optional lexicographic; targets remapped from original properties order',
              'no_truncation':True,'excluded_rows':len(errors),'excluded_oversize_branches':len(long),'holdout':0,
              'shuffle':'deterministic global permutation of packed samples mixing all three sources before rank partition',
              'time':time.time()}
    for name,cfg in configs.items():
        directory=ROOT/'data'/name;directory.mkdir(exist_ok=True)
        if directory!=primary:
            for child in primary.iterdir():
                if child.name=='MANIFEST.json':continue
                (directory/child.name).symlink_to(child)
        write(directory/'MANIFEST.json',{**manifest,'model':name,'base_model':cfg['model_dir'],'base_variant':'instruction_tuned'})
        write(ROOT/'receipts'/f'DATA_READY_{name}.json',{**manifest,'model':name})
    write(primary/'prompt_examples.json',examples)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
