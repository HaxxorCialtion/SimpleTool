"""Reuse verified Qwen3.5 tokens, select nested raw rows, filter >8K branches, repack."""
import argparse
import os
import bisect
import hashlib
import json
import math
import shutil
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

ROOT=Path(__file__).resolve().parent
SOURCE=Path(os.environ['SFT_SOURCE_PACKS_DIR'])
PACK=8192
EMPTY_THINK='<think>\n\n</think>\n\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fraction',type=float,required=True);ap.add_argument('--tag',required=True)
    args=ap.parse_args();assert args.fraction in (.005,.05)
    out=ROOT/'data'/args.tag;assert not (out/'READY').exists()
    src=json.loads((SOURCE/'INDEX.json').read_text())
    assert json.loads((SOURCE/'PREFLIGHT.json').read_text())['valid']
    selected={};sources=[]
    for i,s in enumerate(src['sources']):
        lines=set(map(int,np.random.default_rng(20261108+i).permutation(s['rows'])[:math.ceil(s['rows']*args.fraction)]))
        assert lines<=set(s['selected_line_numbers_zero_based'])
        name=Path(s['path']).name;selected[name]=lines
        sources.append({**s,'selected_rows':len(lines),'selected_line_numbers_zero_based':sorted(lines)})
    out.mkdir(parents=True,exist_ok=True);shutil.copytree(SOURCE/'tokenizer',out/'tokenizer',dirs_exist_ok=True)
    tok=AutoTokenizer.from_pretrained(out/'tokenizer',local_files_only=True)
    suffix=np.array(tok.encode(EMPTY_THINK,add_special_tokens=False),dtype=np.int64)
    assistant=tok.encode('<|im_start|>assistant\n',add_special_tokens=False)
    assert assistant+suffix.tolist()==tok.encode('<|im_start|>assistant\n'+EMPTY_THINK,add_special_tokens=False)
    messages=[{'role':'user','content':'protocol check'}]
    rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    assert rendered.endswith('<|im_start|>assistant\n'+EMPTY_THINK)
    assert rendered==tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    protocol_checks=modified=0
    shards=[];excluded=[];retained=defaultdict(set);known=defaultdict(set);maxlen=0
    for s in src['shards']:
        d=SOURCE/s['name'];meta=json.loads((d/'packs.json').read_text());items=[]
        # Validate the original files before using their token IDs and masks.
        for f,expected in s['sha256'].items():assert hashlib.sha256((d/f).read_bytes()).hexdigest()==expected,(s['name'],f)
        ids=np.load(d/'ids.npy',mmap_mode='r');mask=np.load(d/'mask.npy',mmap_mode='r')
        for p in meta:
            start=p['range'][0]
            for a,b,name in zip(p['cuts'],p['cuts'][1:],p['sources']):
                source,line,head=name.rsplit(':',2);row=(source,int(line))
                if int(line) not in selected[source]:continue
                known[s['split']].add(row)
                x=ids[start+a:start+b].copy();m=mask[start+a:start+b].copy()
                k=int(np.flatnonzero(m)[0])-1 # masked head opening token
                assert x[k-len(assistant):k].tolist()==assistant,name
                assert not np.isin(x[:k],suffix[[0,2]]).any(),('unexpected thinking marker',name)
                assert tok.convert_tokens_to_ids('<'+head+'>')==int(x[k])
                y=np.concatenate((x[:k],suffix.astype(x.dtype),x[k:]))
                mm=np.concatenate((m[:k],np.zeros(len(suffix),dtype=m.dtype),m[k:]))
                assert np.array_equal(y[mm.astype(bool)],x[m.astype(bool)])
                assert np.array_equal(y[:k],x[:k]) and np.array_equal(y[k+len(suffix):],x[k:])
                if protocol_checks<128:
                    assert tok.encode(tok.decode(x[:k])+EMPTY_THINK,add_special_tokens=False)==y[:k+len(suffix)].tolist()
                    protocol_checks+=1
                modified+=1
                if len(y)>PACK:
                    excluded.append({'source':name,'tokens':len(y),'reason':'complete non-thinking branch exceeds 8192'});continue
                items.append((y,mm,name))
                maxlen=max(maxlen,len(y));retained[s['split']].add(row)
        if not items:continue
        free=[];bins=[]
        for i in sorted(range(len(items)),key=lambda i:-len(items[i][0])):
            n=len(items[i][0]);j=bisect.bisect_left(free,(n,-1))
            if j<len(free):rem,k=free.pop(j);bins[k].append(i);bisect.insort(free,(rem-n,k))
            else:k=len(bins);bins.append([i]);bisect.insort(free,(PACK-n,k))
        target=out/s['name'];target.mkdir(exist_ok=True);arrays=[];masks=[];metadata=[];offset=0
        counts=Counter()
        for group in bins:
            cuts=[0];names=[]
            for i in group:
                x,m,name=items[i];arrays.append(x);masks.append(m);cuts.append(cuts[-1]+len(x));names.append(name);counts[name.rsplit(':',1)[1]]+=1
            metadata.append({'range':[offset,offset+cuts[-1]],'cuts':cuts,'sources':names});offset+=cuts[-1]
        x=np.concatenate(arrays);m=np.concatenate(masks)
        np.save(target/'ids.npy',x);np.save(target/'mask.npy',m);(target/'packs.json').write_text(json.dumps(metadata))
        receipt={'name':s['name'],'split':s['split'],'packs':len(bins),'rows':len({n.rsplit(':',1)[0] for _,_,n in items}),
                 'tokens':len(x),'supervised_tokens':int(m.sum()),'branches':dict(counts),
                 'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir() if p.name!='RECEIPT.json'}}
        (target/'RECEIPT.json').write_text(json.dumps(receipt,indent=2));shards.append(receipt)
    assert not known['train']&known['val']
    for s in src['shards']:
        for rejected in s.get('excluded_rows',[]):
            source=next(name for name in selected if s['name'].startswith(name))
            if rejected['line'] in selected[source]:excluded.append({'source':f"{source}:{rejected['line']}",**rejected})
    wanted={(s,i) for s,lines in selected.items() for i in lines}
    quarantine={tuple([e['source'].split(':')[0],int(e['source'].split(':')[1])]) for e in excluded if 'tokens' not in e}
    assert wanted==(known['train']|known['val']|quarantine),('row coverage',len(wanted),len(known['train']|known['val']|quarantine))
    result={'fraction':args.fraction,'seed':20261108,'pack_length':PACK,'sources':sources,'shards':shards,
            'splits':{split:{key:sum(s[key] for s in shards if s['split']==split) for key in ['rows','packs','tokens','supervised_tokens']} for split in ['train','val']},
            'original_prompt_preserved_verbatim':True,'prompt_suffix_added':EMPTY_THINK,
            'thinking_protocol':'explicit_empty_think_before_head_v1','thinking_control_token_ids':suffix.tolist(),
            'non_thinking_branches_processed':modified,'prompt_full_retokenization_checks':protocol_checks,
            'targets_unchanged_and_control_masked':True,'chat_template_applied':False,'train_validation_prompt_group_overlap':0,
            'original_tokens_reused_with_verified_suffix_insertion':True,'selection_source':str(SOURCE),'maximum_segment':maxlen,
            'oversize_dropped':sum('tokens' in e for e in excluded),'excluded':excluded,
            'rows_with_all_branches_filtered':len((known['train']|known['val'])-(retained['train']|retained['val'])),
            'source_index_sha256':hashlib.sha256((SOURCE/'INDEX.json').read_bytes()).hexdigest()}
    (out/'INDEX.json').write_text(json.dumps(result,indent=2));(out/'READY').write_text('ok\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','shards']},indent=2),flush=True)

if __name__=='__main__':main()
