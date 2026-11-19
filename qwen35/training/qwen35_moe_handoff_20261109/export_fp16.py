"""CPU FP16 merge. Independent outputs share unchanged FP16 files by hard link, never symlink."""
import argparse,fcntl,hashlib,json,os,shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from merge_helpers import checkpoint_index,lora_pairs,text_key
ROOT=Path(__file__).resolve().parent

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def dtype_config(x):
    if isinstance(x,dict):
        assert not x.get('quantization_config')
        for k,v in x.items():
            if k in ('dtype','torch_dtype'):x[k]='float16'
            else:dtype_config(v)
    elif isinstance(x,list):
        for v in x:dtype_config(v)

def write_checked(tensors,path):
    assert tensors
    save_file(tensors,str(path),metadata={'format':'pt'})
    with safe_open(str(path),framework='pt') as h:
        assert set(h.keys())==set(tensors)
        for k,t in tensors.items():assert torch.equal(h.get_tensor(k),t),('write mismatch',k)
    return digest(path)

def export(model,stage):
    torch.set_num_threads(4)
    run=ROOT/'runs'/model/stage;adapter=run/'final';out=ROOT/'fp16'/stage/model
    s=json.loads((run/'training_complete.json').read_text());v=json.loads((run/'CHECKPOINT_VALIDATION.json').read_text())
    assert v['valid'] and s['completed_steps']==s['planned_steps'] and s['skipped_steps']==0
    cfg=json.loads((run/'run_config.json').read_text())['config'];base=Path(cfg['MODEL_DIR'])
    identity={str(p.relative_to(run)):digest(p) for p in [run/'training_complete.json',run/'run_config.json',adapter/'adapter_config.json',adapter/'adapter_model.safetensors',adapter/'head_tokens.safetensors']}
    if out.exists():
        accepted=json.loads((out/'EXPORT_VALIDATION.json').read_text());assert accepted['valid'] and accepted['source_identity']==identity
        for name,h in json.loads((out/'MODEL_SHA256.json').read_text()).items():assert digest(out/name)==h
        return
    pairs,scale=lora_pairs(adapter)
    with safe_open(str(adapter/'head_tokens.safetensors'),framework='pt') as h:
        ids=h.get_tensor('ids').long();inputs=h.get_tensor('rows');outputs=h.get_tensor('output_rows');assert h.metadata()['tied_input_output']=='false'
    base_cfg=json.loads((base/'config.json').read_text());hidden=base_cfg.get('text_config',base_cfg)['hidden_size']
    assert inputs.shape==outputs.shape==(17,hidden)
    bmap=checkpoint_index(base);assert 'lm_head.weight' in bmap
    assert set(pairs)<={text_key(k) for k in bmap}
    mutable=sorted(k for k in bmap if text_key(k) in pairs or text_key(k) in ['model.embed_tokens.weight','lm_head.weight'])
    cache=ROOT/'fp16_shared'/model;cache.parent.mkdir(exist_ok=True)
    lock=(cache.parent/f'{model}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    cache_new=not cache.exists();cache_meta=None
    if not cache_new:
        cache_meta=json.loads((cache/'FROZEN_VALIDATION.json').read_text())
        assert cache_meta['valid'] and cache_meta['base']==str(base) and cache_meta['mutable_keys']==mutable
    # Reserve space for LoRA/optimizer checkpoints and other active small-model exports.
    required=sum((base/f).stat().st_size for f in set(bmap.values())) if cache_new else 8*2**30
    free=shutil.disk_usage(ROOT).free
    if free<required+128*2**30:
        (ROOT/'receipts'/f'EXPORT_WAITING_SPACE_{model}_{stage}.json').write_text(json.dumps({'model':model,'stage':stage,'free_bytes':free,'required_bytes':required,'reserve_bytes':128*2**30,'training_blocked':False},indent=2))
        raise SystemExit(75)
    tmp=out.with_name(out.name+'.partial');tmp.mkdir(parents=True,exist_ok=False)
    ctmp=cache.with_name(cache.name+'.partial') if cache_new else cache
    if cache_new:ctmp.mkdir(exist_ok=False)
    fmap={};fhash={};weight_map={};hashes={};merged=incount=outcount=0
    frozen_bytes=mutable_bytes=0
    for number,shard in enumerate(sorted(set(bmap.values())),1):
        fixed={};changed={}
        with safe_open(str(base/shard),framework='pt') as h:
            for k in h.keys():
                mut=k in mutable
                if not mut and not cache_new:continue
                t=h.get_tensor(k);tk=text_key(k)
                if mut:
                    if tk in pairs:
                        p=pairs[tk];assert p['B'].shape[0]==t.shape[0] and p['A'].shape[1]==t.shape[1]
                        t=(t.float()+(p['B'].float()@p['A'].float())*scale).half();merged+=1
                    elif t.is_floating_point():t=t.half()
                    if tk=='model.embed_tokens.weight':t=t.clone();t[ids]=inputs.half();incount+=1
                    if tk=='lm_head.weight':t=t.clone();t[ids]=outputs.half();outcount+=1
                elif t.is_floating_point():t=t.half()
                assert not t.is_floating_point() or (t.dtype==torch.float16 and torch.isfinite(t).all()),('invalid FP16 export',k)
                (changed if mut else fixed)[k]=t.contiguous()
        frozen_bytes+=sum(t.numel()*t.element_size() for t in fixed.values())
        mutable_bytes+=sum(t.numel()*t.element_size() for t in changed.values())
        if fixed:
            name=f'frozen-{number:05d}.safetensors';fhash[name]=write_checked(fixed,ctmp/name)
            fmap.update({k:name for k in fixed})
        if changed:
            name=f'trained-{number:05d}.safetensors';hashes[name]=write_checked(changed,tmp/name)
            weight_map.update({k:name for k in changed})
        del fixed,changed
        print('FP16 merged and reopened:',model,stage,shard,flush=True)
    assert merged==len(pairs) and incount==outcount==1
    if cache_new:
        cache_meta={'valid':True,'base':str(base),'mutable_keys':mutable,'weight_map':fmap,'sha256':fhash,'precision':'float16','tensor_bytes':frozen_bytes,'all_finite':True,'reopened_and_equal':True}
        (ctmp/'FROZEN_VALIDATION.json').write_text(json.dumps(cache_meta,indent=2));ctmp.rename(cache)
    for name,h in cache_meta['sha256'].items():
        os.link(cache/name,tmp/name);hashes[name]=h
        assert (cache/name).stat().st_ino==(tmp/name).stat().st_ino
    weight_map.update(cache_meta['weight_map']);assert set(weight_map)==set(bmap)
    for p in base.iterdir():
        if p.is_file() and p.suffix!='.safetensors' and p.name!='model.safetensors.index.json':shutil.copy2(p,tmp/p.name)
    for p in (adapter/'tokenizer').iterdir():
        if p.is_file():shutil.copy2(p,tmp/p.name)
    index={'metadata':{'total_size':cache_meta['tensor_bytes']+mutable_bytes},'weight_map':weight_map}
    (tmp/'model.safetensors.index.json').write_text(json.dumps(index,indent=2))
    out_cfg=json.loads((tmp/'config.json').read_text());dtype_config(out_cfg);out_cfg['torch_dtype']='float16'
    assert out_cfg['tie_word_embeddings'] is False
    (tmp/'config.json').write_text(json.dumps(out_cfg,indent=2)+'\n')
    tok=AutoTokenizer.from_pretrained(tmp,local_files_only=True)
    tokens=[t for h in ['content','function',*[f'arg{i}' for i in range(1,7)]] for t in (f'<{h}>',f'</{h}>')]+['<|null|>']
    assert [tok.convert_tokens_to_ids(t) for t in tokens]==ids.tolist() and set(ids.tolist())<=set(tok.all_special_ids)
    probe=[{'role':'user','content':'protocol check'}];rendered=tok.apply_chat_template(probe,tokenize=False,add_generation_prompt=True)
    assert rendered.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
    assert rendered==tok.apply_chat_template(probe,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    assert checkpoint_index(tmp)==weight_map and all((tmp/name).is_file() and not (tmp/name).is_symlink() for name in set(weight_map.values()))
    (tmp/'prompt_protocol.json').write_text(json.dumps({'protocol':'explicit_empty_think_before_head_v1','default_enable_thinking':False,'assistant_head_prefix':'<|im_start|>assistant\n<think>\n\n</think>\n\n<HEAD>','raw_chatml_rule':'Preserve system/user context; append empty think before head exactly once; never re-template raw ChatML.'},indent=2))
    for name in ['training_summary.json','CHECKPOINT_VALIDATION.json','run_config.json']:shutil.copy2(run/name,tmp/name)
    accepted={'valid':True,'source_identity':identity,'precision':'FP16 standalone, non-expert training base BF16, adapters FP32','training_expert_storage':cfg.get('EXPERT_DTYPE','bf16'),'frozen_export_base':'original BF16 checkpoint, without baking training-only FP8 expert approximation','shared_frozen_files':'hard links, ordinary files; no symlinks; each export independently movable','merged_lora_weights':merged,'input_token_rows':17,'output_token_rows':17,'hidden_size':hidden,'all_tensors_finite':True,'written_tensors_reopened_and_verified':True,'vision_and_processors_preserved':True,'benchmark_evaluation_performed':False,'thinking_protocol':'explicit_empty_think_before_head_v1','quantized':False,'completed_steps':s['completed_steps']}
    (tmp/'EXPORT_VALIDATION.json').write_text(json.dumps(accepted,indent=2))
    hashes.update({p.name:digest(p) for p in tmp.iterdir() if p.is_file() and p.suffix!='.safetensors'})
    (tmp/'MODEL_SHA256.json').write_text(json.dumps(hashes,indent=2));tmp.rename(out)
    waiting=ROOT/'receipts'/f'EXPORT_WAITING_SPACE_{model}_{stage}.json'
    if waiting.exists():waiting.unlink()
    print('FP16 export complete:',out,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('model',choices=['35b','122b','397b']);ap.add_argument('stage',choices=['p0005','p005']);a=ap.parse_args();export(a.model,a.stage)
