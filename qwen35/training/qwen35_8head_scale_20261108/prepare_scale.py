"""Prepare nested per-source subsets with bounded RAM and original raw prompts."""
import argparse, concurrent.futures, hashlib, json, math, os, random, collections, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
HEADS=['content','function']+[f'arg{i}' for i in range(1,7)]
def setup(tokenizer,out):
    global TOK,OUT
    from transformers import AutoTokenizer
    TOK=AutoTokenizer.from_pretrained(tokenizer,local_files_only=True)
    OUT=Path(out)

def worker(task):
    source,chunk,split,rows=task
    name=f'{source}-{chunk:06d}-{split}'
    existing=OUT/name/'RECEIPT.json'
    if existing.exists(): return json.loads(existing.read_text())
    texts=[]; samples=[]; skipped=collections.Counter(); rejected=[]
    for line,row in rows:
        prompt=row.get('prompt') or row.get('prefix') or ''
        assert prompt.endswith('<|im_start|>assistant\n'),(source,line)
        markers=['<think>','</think>','<thinking>','</thinking>']
        if any(m in prompt or any(m in row.get(f'head_{j}_{h}','') for j,h in enumerate(HEADS)) for m in markers):
            rejected.append({'line':line,'reason':'thinking/protocol remnants in raw prompt or target'});continue
        pi=len(texts);texts.append(prompt)
        for j,h in enumerate(HEADS):
            v=row.get(f'head_{j}_{h}','')
            if not v or h=='content' and '<continue>' in v:
                skipped[h]+=1;continue
            assert v.startswith(f'<{h}>') and v.endswith(f'</{h}><|im_end|>'),(source,line,h)
            assert not any(m in v for m in ['<think>','</think>','<thinking>','</thinking>'])
            samples.append((pi,len(texts),f'{source}:{line}:{h}'));texts.append(v)
    encoded=TOK(texts,add_special_tokens=False)['input_ids']
    segs=[]; counts=collections.Counter()
    for pi,ti,segname in samples:
        p,t=encoded[pi],encoded[ti]; n=len(p)+len(t)
        assert n<=16384,('overlong complete segment; no truncation authorized',segname,n)
        ids=np.asarray(p+t,dtype=np.int32)
        mask=np.zeros(n,dtype=np.uint8); mask[len(p)+1:]=1
        assert t[0]==TOK.convert_tokens_to_ids('<'+segname.rsplit(':',1)[1]+'>')
        assert t[-1]==TOK.eos_token_id
        segs.append((ids,mask,segname));counts[segname.rsplit(':',1)[1]]+=1
    # Best-fit packing within a bounded row shard, no source segment is split.
    import bisect
    bins=[]; free=[]
    for i in sorted(range(len(segs)),key=lambda i:-len(segs[i][0])):
        n=len(segs[i][0]); j=bisect.bisect_left(free,(n,-1))
        if j<len(free):
            rem,b=free.pop(j);bins[b].append(i);bisect.insort(free,(rem-n,b))
        else:
            b=len(bins);bins.append([i]);bisect.insort(free,(16384-n,b))
    path=OUT/name;path.mkdir(exist_ok=False)
    idsall=[];maskall=[];meta=[];offset=0
    for b in bins:
        cuts=[0];names=[]
        for i in b:
            ids,mask,nm=segs[i];idsall.append(ids);maskall.append(mask);cuts.append(cuts[-1]+len(ids));names.append(nm)
        meta.append({'range':[offset,offset+cuts[-1]],'cuts':cuts,'sources':names});offset+=cuts[-1]
    ids=np.concatenate(idsall) if idsall else np.array([],dtype=np.int32)
    mask=np.concatenate(maskall) if maskall else np.array([],dtype=np.uint8)
    np.save(path/'ids.npy',ids);np.save(path/'mask.npy',mask)
    (path/'packs.json').write_text(json.dumps(meta))
    receipt={'name':name,'split':split,'packs':len(meta),'rows':len(rows),'branches':dict(counts),
             'tokens':len(ids),'supervised_tokens':int(mask.sum()),'skipped':dict(skipped),'excluded_rows':rejected,
             'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in path.iterdir()}}
    (path/'RECEIPT.json').write_text(json.dumps(receipt,indent=2))
    return receipt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fraction',type=float,required=True);ap.add_argument('--workers',type=int,default=12)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--tokenizer',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();name=f'p{round(100*a.fraction):03d}'
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'READY').exists()
    tokenizer=a.tokenizer
    import shutil
    shutil.copytree(tokenizer,out/'tokenizer',dirs_exist_ok=True)
    sources=[('09_train_ready.jsonl',402137),('09b_train_ready.jsonl',988712),('final_simplelove.jsonl',115110)]
    source_root=a.source_root; receipts=[];sources_receipt=[];buffer={};chunk={};start=time.time()
    os.environ['TOKENIZERS_PARALLELISM']='false'
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,initializer=setup,initargs=(str(tokenizer),str(out))) as pool:
        pending=set()
        def collect():
            nonlocal pending
            done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                rec=f.result();receipts.append(rec)
                print(json.dumps({'shard':rec['name'],'rows':rec['rows'],'tokens':rec['tokens'],'elapsed':round(time.time()-start)}),flush=True)
        for src,n in sources:
            # One deterministic permutation per source, nested across fractions.
            order=np.random.default_rng(20261108+sources.index((src,n))).permutation(n)
            chosen=set(int(i) for i in order[:math.ceil(n*a.fraction)])
            digest=hashlib.sha256(); selected=[];groups={'train':set(),'val':set()}
            with (source_root/src).open('rb') as stream:
                for i,line in enumerate(stream):
                    digest.update(line)
                    if i not in chosen:continue
                    row=json.loads(line);prompt=row.get('prompt') or row.get('prefix') or ''
                    group=hashlib.sha256(prompt.encode()).hexdigest();split='val' if int(group[:8],16)%20==0 else 'train'
                    groups[split].add(group);selected.append(i);key=(src,split)
                    buffer.setdefault(key,[]).append((i,row))
                    if len(buffer[key])>=256:
                        j=chunk.get(key,0);chunk[key]=j+1;pending.add(pool.submit(worker,(src,j,split,buffer.pop(key))))
                        if len(pending)>=a.workers*2:collect()
            assert i+1==n,(src,i+1,n)
            assert not groups['train']&groups['val']
            sources_receipt.append({'path':str(source_root/src),'rows':n,'selected_rows':len(selected),'sha256':digest.hexdigest(),
                                    'selected_line_numbers_zero_based':selected,'group_overlap':0})
        for (src,split),rows in buffer.items():
            pending.add(pool.submit(worker,(src,chunk.get((src,split),0),split,rows)))
        while pending:collect()
    splits={}
    for split in ['train','val']:
        parts=[r for r in receipts if r['split']==split]
        splits[split]={k:sum(r[k] for r in parts) for k in ['rows','packs','tokens','supervised_tokens']}
    receipt={'fraction':a.fraction,'seed':20261108,'sources':sources_receipt,'splits':splits,
             'shards':sorted(receipts,key=lambda s:s['name']),'prompt_preserved_verbatim':True,'chat_template_applied':False,
             'thinking_marker_rows_excluded':sum(sum('thinking' in r['reason'] for r in s.get('excluded_rows',[])) for s in receipts),'oversize_dropped':0,'train_validation_prompt_group_overlap':0}
    (out/'INDEX.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
    (out/'READY').write_text('ok\n');print('READY',name,splits,flush=True)

if __name__=='__main__':main()
