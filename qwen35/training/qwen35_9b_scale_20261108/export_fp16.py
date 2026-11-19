"""Materialize a standalone FP16 VLM, merging text LoRA and distinct token rows."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from merge_helpers import checkpoint_index,lora_pairs,text_key

ROOT=Path(__file__).resolve().parent

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def dtype_config(x):
    if isinstance(x,dict):
        assert not x.get('quantization_config')
        for k,v in x.items():
            if k in ('dtype','torch_dtype'):x[k]='float16'
            else:dtype_config(v)
    elif isinstance(x,list):
        for v in x:dtype_config(v)

def export(stage):
    torch.set_num_threads(4)
    run=ROOT/'runs/9b'/stage;adapter=run/'final';out=ROOT/'fp16'/stage/'9b'
    summary=json.loads((run/'training_complete.json').read_text())
    assert summary['completed_steps']==summary['planned_steps'] and summary['skipped_steps']==0
    validation=json.loads((run/'CHECKPOINT_VALIDATION.json').read_text());assert validation['valid']
    config=json.loads((run/'run_config.json').read_text())['config'];base=Path(config['MODEL_DIR'])
    identity={str(p.relative_to(run)):digest(p) for p in [run/'training_complete.json',run/'run_config.json',
                    adapter/'adapter_config.json',adapter/'adapter_model.safetensors',adapter/'head_tokens.safetensors']}
    if out.exists():
        receipt=json.loads((out/'EXPORT_VALIDATION.json').read_text());assert receipt['source_identity']==identity and receipt['valid']
        for f,h in json.loads((out/'MODEL_SHA256.json').read_text()).items():assert digest(out/f)==h
        print('Existing FP16 export rechecked:',out,flush=True);return
    pairs,scale=lora_pairs(adapter)
    with safe_open(str(adapter/'head_tokens.safetensors'),framework='pt') as h:
        ids=h.get_tensor('ids').long();input_rows=h.get_tensor('rows');output_rows=h.get_tensor('output_rows')
        assert h.metadata()['tied_input_output']=='false'
    assert input_rows.shape==output_rows.shape==(17,4096)
    bmap=checkpoint_index(base);assert 'lm_head.weight' in bmap
    names={text_key(k) for k in bmap};assert set(pairs)<=names
    tmp=out.with_name(out.name+'.partial');tmp.mkdir(parents=True,exist_ok=False)
    merged=inputs=outputs=count=0
    for shard in sorted(set(bmap.values())):
        tensors={}
        with safe_open(str(base/shard),framework='pt') as h:
            for k in h.keys():
                t=h.get_tensor(k);tk=text_key(k)
                if tk in pairs:
                    p=pairs[tk];assert p['B'].shape[0]==t.shape[0] and p['A'].shape[1]==t.shape[1]
                    t=(t.float()+(p['B'].float()@p['A'].float())*scale).half();merged+=1
                elif t.is_floating_point():t=t.half()
                if tk=='model.embed_tokens.weight':
                    t=t.clone();t[ids]=input_rows.half();inputs+=1
                elif tk=='lm_head.weight':
                    t=t.clone();t[ids]=output_rows.half();outputs+=1
                assert not t.is_floating_point() or torch.isfinite(t).all(),('nonfinite export',k)
                tensors[k]=t.contiguous();count+=1
        save_file(tensors,str(tmp/shard),metadata={'format':'pt'})
        with safe_open(str(tmp/shard),framework='pt') as written:
            assert set(written.keys())==set(tensors)
            for k,t in tensors.items():assert torch.equal(written.get_tensor(k),t),('write mismatch',k)
        del tensors
        print('Merged FP16 shard and reopened:',shard,flush=True)
    assert merged==len(pairs) and inputs==outputs==1
    for p in base.iterdir():
        if p.is_file() and p.suffix!='.safetensors':shutil.copy2(p,tmp/p.name)
    for p in (adapter/'tokenizer').iterdir():
        if p.is_file():shutil.copy2(p,tmp/p.name)
    cfg=json.loads((tmp/'config.json').read_text());assert cfg['tie_word_embeddings'] is False
    dtype_config(cfg);cfg['torch_dtype']='float16';cfg['text_config']['tie_word_embeddings']=False
    (tmp/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    tok=AutoTokenizer.from_pretrained(tmp,local_files_only=True)
    probe=[{'role':'user','content':'protocol check'}]
    rendered=tok.apply_chat_template(probe,tokenize=False,add_generation_prompt=True)
    assert rendered.endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')
    assert rendered==tok.apply_chat_template(probe,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    # Raw ChatML callers must also add this block, exactly once, before the head.
    (tmp/'prompt_protocol.json').write_text(json.dumps({
        'protocol':'explicit_empty_think_before_head_v1','default_enable_thinking':False,
        'assistant_head_prefix':'<|im_start|>assistant\n<think>\n\n</think>\n\n<HEAD>',
        'raw_chatml_rule':'Preserve system/user context; append empty think block before head opening, exactly once. Do not re-template raw ChatML.',
        'training_control_tokens_supervised':False,
        'official_source':'https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/chat_template.jinja'
    },ensure_ascii=False,indent=2)+'\n')
    tokens=[t for h in ['content','function',*[f'arg{i}' for i in range(1,7)]] for t in (f'<{h}>',f'</{h}>')]+['<|null|>']
    assert [tok.convert_tokens_to_ids(t) for t in tokens]==ids.tolist()
    assert set(ids.tolist())<=set(tok.all_special_ids)
    assert checkpoint_index(tmp)==bmap and not any(p.is_symlink() for p in tmp.iterdir())
    for name in ['training_summary.json','CHECKPOINT_VALIDATION.json','run_config.json']:shutil.copy2(run/name,tmp/name)
    receipt={'valid':True,'training_complete':True,'completed_steps':summary['completed_steps'],
             'precision':'FP16 standalone weights; trained with BF16 frozen base and FP32 adapters',
             'quantized':False,'source_identity':identity,'tensor_count':count,'merged_lora_weights':merged,
             'input_token_rows':17,'output_token_rows':17,'tied_embeddings':False,
             'all_tensors_finite':True,'written_tensors_reopened_and_verified':True,
             'vision_and_processors_preserved':True,'benchmark_evaluation_performed':False,
             'thinking_protocol':'explicit_empty_think_before_head_v1','default_enable_thinking':False,
             'merge_math':'FP32(base + LoRA delta) -> FP16; independent token rows -> FP16'}
    (tmp/'EXPORT_VALIDATION.json').write_text(json.dumps(receipt,indent=2))
    (tmp/'MODEL_SHA256.json').write_text(json.dumps({p.name:digest(p) for p in sorted(tmp.iterdir()) if p.is_file()},indent=2))
    tmp.rename(out);print('FP16 export complete:',out,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['p0005','p005']);export(ap.parse_args().stage)
