"""Build a bounded, dependency-explicit FP8 eval view; preserve both head row sets.
Frozen expert shards reference immutable official FP8 weights. Updated nonexpert
weights are merged from ORIGINAL BF16 + LoRA, then quantized once into the official
block format. Source checkpoints and public bases are never modified.
"""
import argparse,hashlib,json,os,pathlib,shutil,time,sys

def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--size',choices=['35b','122b','397b'],required=True);ap.add_argument('--pct',choices=['p0005','p005'],required=True);a=ap.parse_args()
 G=pathlib.Path('/inspire/hdd/global_user/shixiaoxin-253107030017/sxx');T=G/'multi-head/rethinking/pretrain_level/final/glm51/sota/hermes/simpletool/final_data/vlm';M=T/'qwen35_moe_handoff_20261109'
 D=pathlib.Path('/inspire/qb-ilm2/project/new-paradigm-of-manipulation/shixiaoxin-253107030017/simpletool_eval_20261110');assert D.is_dir() and D.stat().st_dev!=T.stat().st_dev
 adapter=M/'runs'/a.size/a.pct/'final';base=pathlib.Path('/inspire/hdd/global_public/public_models/Qwen')/{'35b':'Qwen3.6-35B-A3B','122b':'Qwen3.5-122B-A10B','397b':'Qwen3.5-397B-A17B'}[a.size];fp8=base.with_name('Qwen-3.6-35B-A3B-FP8' if a.size=='35b' else base.name+'-FP8');out=D/'eval_views'/f'{a.pct}-{a.size}-fp8'
 if (out/'BUILD_DONE.json').exists():print('DONE',out,flush=True);return
 assert not out.exists(),f'incomplete attempt retained: {out}'
 for name in ['adapter_config.json','adapter_model.safetensors','head_tokens.safetensors','train_config.json','tokenizer/tokenizer.json']:assert (adapter/name).is_file(),adapter/name
 # Completed final is required; do not inspect partially-produced training checkpoints.
 complete=adapter.parent/'training_complete.json';validation=adapter.parent/'CHECKPOINT_VALIDATION.json'
 assert complete.exists() and validation.exists(),(complete,validation)
 print('TRAINING',complete.read_text(),validation.read_text(),flush=True)
 import torch
 from safetensors import safe_open
 from safetensors.torch import save_file
 from transformers import AutoTokenizer
 torch.set_num_threads(8)
 sys.path.insert(0,str(M));from merge_helpers import checkpoint_index,text_key,lora_pairs,fp8_quant,fp8_dequant,selftest
 selftest()
 pairs,scale=lora_pairs(adapter);assert all('.experts.' not in k for k in pairs),'expert LoRA requires another merge path'
 train=json.loads((adapter/'train_config.json').read_text());assert pathlib.Path(train['MODEL_DIR'])==base
 wmap=checkpoint_index(fp8);bmap=checkpoint_index(base);normal={text_key(k):k for k in wmap if text_key(k)};original={text_key(k):k for k in bmap if text_key(k)}
 assert set(pairs)<=set(normal) and set(pairs)<=set(original)
 heads=safe_open(str(adapter/'head_tokens.safetensors'),framework='pt');ids=heads.get_tensor('ids').long();rows=heads.get_tensor('rows');output_rows=heads.get_tensor('output_rows');assert heads.metadata()['tied_input_output']=='false'
 tok=AutoTokenizer.from_pretrained(str(adapter/'tokenizer'),local_files_only=True)
 special=json.loads(heads.metadata()['tokens']);assert ids.tolist()==[tok.convert_tokens_to_ids(t) for t in special] and len(special)==17
 changed={normal[k]:k for k in pairs};rowkeys={normal['model.embed_tokens.weight']:rows,normal['lm_head.weight']:output_rows}
 scales={k:k+'_scale_inv' for k in changed if k+'_scale_inv' in wmap};touched=set(changed)|set(rowkeys)|set(scales.values());rewrite=sorted({wmap[k] for k in touched});allshards=sorted(set(wmap.values()));assert len(rewrite)<=(50 if a.size=="35b" else 4),(len(rewrite),rewrite)
 out.mkdir(parents=True);receipt=dict(status='building',started=time.time(),size=a.size,pct=a.pct,original_base=str(base),official_fp8_base=str(fp8),adapter=str(adapter),source_hashes={n:sha(adapter/n) for n in ['adapter_config.json','adapter_model.safetensors','head_tokens.safetensors','train_config.json','tokenizer/tokenizer.json']},training_complete=json.loads(complete.read_text()),checkpoint_validation=json.loads(validation.read_text()),special_ids=ids.tolist(),input_output_independent=True,lora_pairs=len(pairs),lora_scale=scale,rewrite=rewrite,reference_shards=[f for f in allshards if f not in rewrite],precision='official frozen-expert block FP8; trained nonexperts original BF16 plus LoRA then block FP8; independent FP16/BF16 special input/output rows',dependency='Referenced expert shards are immutable public base symlinks; this is an evaluation view, not a standalone exported checkpoint',rewritten_hashes={},errors=[])
 write(out/'BUILD_STATUS.json',receipt)
 for f in fp8.iterdir():
  if f.is_file() and f.suffix not in ['.safetensors'] and f.name!='model.safetensors.index.json':shutil.copy2(f,out/f.name)
 for f in (adapter/'tokenizer').iterdir():
  if f.is_file():shutil.copy2(f,out/f.name)
 config=json.loads((out/'config.json').read_text());assert config['quantization_config']['weight_block_size']==[128,128];config['tie_word_embeddings']=False
 write(out/'config.json',config);write(out/'prompt_protocol.json',dict(protocol='explicit_empty_think_before_head_v1',assistant_prefix='<think>\n\n</think>\n\n',heads=['content','function']+['arg'+str(i) for i in range(1,7)]))
 for f in allshards:
  if f not in rewrite:os.symlink(fp8/f,out/f)
 bhandles={};seen=set();scaleseen=set();rowdone=set()
 def original_tensor(k):
  key=original[text_key(k)];f=bmap[key]
  if f not in bhandles:bhandles[f]=safe_open(str(base/f),framework='pt')
  return bhandles[f].get_tensor(key)
 invscale={v:k for k,v in scales.items()}
 for fname in rewrite:
  tensors={};pending={}
  with safe_open(str(fp8/fname),framework='pt') as h:
   for k in h.keys():
    if k in scales or k in invscale:
     wk=k if k in scales else invscale[k];sk=scales[wk]
     if wk not in pending:
      q=h.get_tensor(wk);s=h.get_tensor(sk);p=pairs[changed[wk]];w=original_tensor(wk).float()+scale*(p['B'].float()@p['A'].float());assert w.shape==q.shape
      q2,s2=fp8_quant(w,q.dtype,s.dtype);error=float((fp8_dequant(q2,s2)-w).norm()/w.norm());assert error<.06,(wk,error);receipt['errors'].append(dict(key=wk,relative_l2=error));pending[wk]=(q2,s2);seen.add(wk);scaleseen.add(sk)
     tensors[k]=pending[wk][0 if k==wk else 1]
    elif k in changed:
     p=pairs[changed[k]];ref=h.get_tensor(k);w=original_tensor(k).float()+scale*(p['B'].float()@p['A'].float());assert w.shape==ref.shape;tensors[k]=w.to(ref.dtype).contiguous();seen.add(k)
    elif k in rowkeys:
     t=h.get_tensor(k).clone();assert ids.max()<t.shape[0];t[ids]=rowkeys[k].to(t.dtype);assert torch.equal(t[ids],rowkeys[k].to(t.dtype));tensors[k]=t.contiguous();rowdone.add(k)
    else:tensors[k]=h.get_tensor(k)
  assert set(tensors)=={k for k,v in wmap.items() if v==fname},fname
  save_file(tensors,str(out/fname),metadata={'format':'pt'});receipt['rewritten_hashes'][fname]=sha(out/fname);write(out/'BUILD_STATUS.json',receipt);print('SHARD_DONE',a.size,a.pct,fname,flush=True)
  del tensors,pending
 assert seen==set(changed) and scaleseen==set(scales.values()) and rowdone==set(rowkeys)
 # Every shard has the original complete key set, preventing duplicate-key overlays.
 index=json.loads((fp8/'model.safetensors.index.json').read_text());write(out/'model.safetensors.index.json',index)
 receipt.update(status='completed',ended=time.time(),config_sha256=sha(out/'config.json'),tokenizer_sha256=sha(out/'tokenizer.json'),index_sha256=sha(out/'model.safetensors.index.json'))
 write(out/'BUILD_STATUS.json',receipt);write(out/'BUILD_DONE.json',receipt);print('BUILD_DONE',out,flush=True)
if __name__=='__main__':main()
