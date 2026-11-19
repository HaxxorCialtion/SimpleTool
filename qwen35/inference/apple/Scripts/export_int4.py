"""Verified shard-wise text-only Qwen3.5 -> existing Swift affine W4/group64 ABI.
Never runs BF16 inference; preserves raw checkpoints; refuses overwrite.
"""
import argparse,gc,hashlib,json,shutil,time
from pathlib import Path
import mlx.core as mx
mx.set_default_device(mx.cpu)
mx.set_cache_limit(128*1024**2)

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def export(source,out,reference):
 start=time.monotonic();assert not out.exists(),f'Refusing overwrite: {out}'
 assert reference['all_verified'] and Path(reference['local_dir']).resolve()==source.resolve()
 assert shutil.disk_usage(out.parent).free>8*1024**3,'Keep an 8 GiB reserve'
 config=json.loads((source/'config.json').read_text())
 c=config.get('text_config',config)
 assert c['tie_word_embeddings'] and c['linear_key_head_dim']==128 and c['linear_value_head_dim']==128
 out.mkdir();weight_map={};total=0;shards=[];skipped=[]
 norms=('.input_layernorm.weight','.post_attention_layernorm.weight','model.norm.weight','.q_norm.weight','.k_norm.weight')
 for info in reference['files']:
  if not info['path'].endswith('.safetensors'):continue
  src=source/info['path'];assert sha(src)==info['sha256'],'Source changed after download'
  arrays=mx.load(str(src));converted={};count=0
  for key,w in arrays.items():
   if not key.startswith('model.language_model.'):
    skipped.append(key);continue # vision and MTP are not part of this text runtime.
   assert bool(mx.all(mx.isfinite(w)).item()),f'Nonfinite source: {key}'
   name=key.replace('model.language_model.','language_model.model.',1)
   if key.endswith('conv1d.weight'):
    assert w.ndim==3 and w.shape[1]==1 and w.shape[2]==4
    w=w.moveaxis(2,1)
   if any(key.endswith(n) for n in norms):w=w+1.0 # HF offset RMSNorm -> MLX multiplicative RMSNorm
   if w.ndim==2 and key.endswith('.weight'):
    assert w.shape[-1]%64==0
    packed,scale,bias=mx.quantize(w,group_size=64,bits=4)
    converted[name]=packed;converted[name[:-6]+'scales']=scale.astype(mx.float32);converted[name[:-6]+'biases']=bias.astype(mx.float32)
   else:converted[name]=w.astype(mx.float32)
   mx.eval(*[v for k,v in converted.items() if k==name or k in [name[:-6]+'scales',name[:-6]+'biases']]);count+=1
  if not converted:del arrays;gc.collect();mx.clear_cache();continue
  target=out/info['path'];mx.save_safetensors(str(target),converted)
  check=mx.load(str(target));assert set(check)==set(converted)
  assert all(bool(mx.all(check[k]==v).item()) for k,v in converted.items()),'Reload mismatch'
  for k,v in converted.items():
   assert k not in weight_map;weight_map[k]=target.name;total+=v.nbytes
  shards.append({'name':target.name,'bytes':target.stat().st_size,'sha256':sha(target),'source_sha256':info['sha256'],'source_text_tensors':count,'output_tensors':len(converted),'reload_bitwise_equal':True})
  print('EXPORTED',source.name,target.name,target.stat().st_size,flush=True)
  del arrays,converted,check,w,packed,scale,bias;gc.collect();mx.clear_cache()
 config['quantization']={'bits':4,'group_size':64,'mode':'affine'};config['quantization_config']=config['quantization']
 (out/'config.json').write_text(json.dumps(config,indent=2))
 (out/'model.safetensors.index.json').write_text(json.dumps({'metadata':{'total_size':total},'weight_map':weight_map},indent=2))
 for p in source.iterdir():
  if p.is_file() and (p.name.startswith(('tokenizer','special_tokens','added_tokens','chat_template','generation_config')) or p.name in ['README.md','LICENSE']):shutil.copy2(p,out/p.name)
 tok=json.loads((source/'tokenizer.json').read_text());added={a['content']:a['id'] for a in tok['added_tokens']};heads=['speech','expression','inner','command']
 protocol={'heads':heads,'head_token_ids':{h:added['<'+h+'>'] for h in heads},'eos_token_id':added['<|im_end|>'],'source_protocol':'20261027-heads; see Docs/INTEGRATION.md; pass preparedPayload from Heads20261105Prompt','text_only':True}
 (out/'simpletool_config.json').write_text(json.dumps(protocol,indent=2))
 receipt={'source_repo':reference['repo'],'revision':reference['revision'],'source':str(source),'source_untouched':True,'quantization':config['quantization'],'floating_compute':'float32','text_only':True,'omitted_tensor_keys':skipped,'transformations':['model.language_model -> language_model.model','conv1d [C,1,4] -> [C,4,1]','offset RMSNorm +1 before storage','2D weights affine int4/group64','scale/bias and remaining floats FP32'],'shards':shards,'tensor_count':len(weight_map),'tensor_bytes':total,'all_reload_equal':True,'seconds':time.monotonic()-start}
 (out/'conversion_receipt.json').write_text(json.dumps(receipt,indent=2));print('DONE',source.name,total,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--models-root',type=Path,required=True);args=p.parse_args()
 args.models_root.mkdir(parents=True,exist_ok=True)
 for r in json.loads(args.receipt.read_text()):export(Path(r['local_dir']),args.models_root/(r['repo'].split('/')[-1]+'-int4'),r)
