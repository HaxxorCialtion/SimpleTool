"""Frozen Diagnostic-800, native Gemma BF16 base + FP32 training adapters.
No content. No tools executed. Keeps native PLE, softcap and shared layer KV.
"""
import os,sys,json,time,hashlib,argparse,collections,subprocess,platform,socket,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TRAIN=ROOT
BATCHES=['cf-zh','cf-en','hard','multiarg']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def jl(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def write(p,x):
 p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def native_prompt(text,thought):
 suffix='<|im_start|>assistant\n';assert text.endswith(suffix)
 messages=[];position=0
 for m in re.finditer(r'<\|im_start\|>([^\n]+)\n(.*?)<\|im_end\|>\n?',text,re.S):
  assert m.start()==position;role,body=m.groups();assert role in ['system','user','assistant','tool']
  messages.append((role,body));position=m.end()
 assert text[position:]==suffix
 return '<bos>'+''.join('<|turn>'+('model' if r=='assistant' else r)+'\n'+c+'<turn|>\n' for r,c in messages)+'<|turn>model\n'+thought

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--limit',type=int);ap.add_argument('--base',required=True);ap.add_argument('--artifact',type=Path,required=True);ap.add_argument('--datasets',type=Path,default=ROOT.parents[1]/'qwen35/evaluation/diagnostic800/frozen');ap.add_argument('--output',type=Path,default=Path('gemma-evaluation'));a=ap.parse_args()
 a.batch_size = max(a.batch_size, {'e2b': 32, 'e4b': 32, '12b': 16, '26b_a4b': 16, '31b': 8}[a.model])
 import fcntl
 early_out=a.output/(a.model+('-smoke' if a.limit else ''));early_out.mkdir(parents=True,exist_ok=True)
 lock=(early_out/'evaluation.lock').open('w')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  print('Evaluation already running for',a.model,flush=True);return
 if (early_out/'run.json').exists() and read(early_out/'run.json').get('status')=='completed':
  prior=read(early_out/'run.json');assert (early_out/'heads.jsonl').exists() and sha(early_out/'heads.jsonl')==prior['evidence_sha256']
  print('Verified existing completed evaluation for',a.model,flush=True);return
 os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
 import torch,transformers
 from transformers import AutoTokenizer
 sys.path.insert(0,str(TRAIN));import modeling
 torch.set_num_threads(8);torch.manual_seed(1107)
 cfg=read(TRAIN/'configs'/f'{a.model}.json');cfg['model_dir']=a.base;artifact=a.artifact
 assert read(artifact/'COMPLETE.json')['final'] is True
 tok=AutoTokenizer.from_pretrained(artifact/'tokenizer',local_files_only=True,padding_side='left')
 thought='<|channel>thought\n<channel|>' if a.model in ['12b','26b_a4b','31b'] else ''
 template=tok.apply_chat_template([{'role':'system','content':'SYS'},{'role':'user','content':'USR'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 assert template=='<bos><|turn>system\nSYS<turn|>\n<|turn>user\nUSR<turn|>\n<|turn>model\n'+thought,repr(template)
 marker={s:tok.encode(s,add_special_tokens=False) for s in modeling.SPECIAL_TOKENS};assert all(len(v)==1 for v in marker.values())
 dataset={};tasks=[]
 for b in BATCHES:
  d=a.datasets/b;cases=jl(d/'cases.jsonl');prompts=jl(d/'prompts.jsonl');assert len(cases)==len(prompts)==200
  assert [r['id'] for r in cases]==[r['id'] for r in prompts]
  dataset[b]={n:sha(d/n) for n in ['cases.jsonl','prompts.jsonl']}
  for row,p in zip(cases,prompts):
   maximum=max(len(t['function']['parameters']['properties']) for t in row['tools'])
   assert p['heads']==['function']+[f'arg{i}' for i in range(1,maximum+1)]
   for t in row['tools']:
    schema=t['function']['parameters'];assert list(schema['properties'])[:len(schema['required'])]==schema['required']
   common=native_prompt(p['prompt'],thought)
   for head in p['heads']:
    ids=tok.encode(common+f'<{head}>',add_special_tokens=False);assert len(ids)+128<=8192
    tasks.append(dict(batch=b,id=row['id'],head=head,ids=ids,close=marker[f'</{head}>'][0]))
 out=a.output/(a.model+('-smoke' if a.limit else ''));out.mkdir(parents=True,exist_ok=True)
 receipt=dict(status='loading',model=a.model,base=cfg['model_dir'],artifact=str(artifact),completed_training=read(artifact/'COMPLETE.json'),precision='BF16 frozen base; FP32 adapters/special rows; BF16 autocast; no quantization',runtime='native Transformers/PyTorch SDPA; custom original training loader',torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(),hostname=socket.gethostname(),datasets=dataset,marker_ids=marker,temperature=0,seed=1107,max_new_tokens=128,batch_size=a.batch_size,head_policy='function + candidate maximum arity, no content',prompt_adapter='exact frozen bodies preserved; ChatML role shell replaced by trained native Gemma shell; native closed thought only for 12B/26B/31B',source_sha256=sha(__file__),training_modeling_sha256=sha(TRAIN/'modeling.py'),adapter_hashes={p.name:sha(p) for p in artifact.glob('*.safetensors')},tokenizer_sha256=sha(artifact/'tokenizer/tokenizer.json'),started=time.time())
 write(out/'run.json',receipt)
 head_ids={h:marker[f'<{h}>'][0] for h in modeling.HEADS}
 model=modeling.load_adapters(cfg['model_dir'],artifact,tok,head_ids,torch.device('cuda'))
 model.eval();model.requires_grad_(False)
 model.set_attn_implementation('sdpa');model.config.use_cache=True
 forward=model.get_base_model()
 # Test correct training artifact restoration and special-row/PLE restoration.
 from safetensors.torch import load_file
 saved=load_file(artifact/'head_tokens.safetensors',device='cuda')
 rowerr=float((model.get_input_embeddings().final_rows()-saved['rows']).abs().max())
 assert rowerr<1e-5,rowerr
 if 'per_layer_rows' in saved:
  pleerr=float((model.get_decoder().embed_tokens_per_layer.final_rows()-saved['per_layer_rows']).abs().max());assert pleerr<1e-5,pleerr
 else:pleerr=None
 del saved
 stopids=set([tok.eos_token_id])
 for key in ['eos_token_id']:
  v=getattr(model.config,key,None)
  if v is not None:stopids.update(v if isinstance(v,list) else [v])
 stopids.add(tok.convert_tokens_to_ids('<turn|>'))
 pad=tok.pad_token_id if tok.pad_token_id is not None else 0
 def tensors(group):
  L=max(len(t['ids']) for t in group);x=torch.full((len(group),L),pad,device='cuda',dtype=torch.long);mask=torch.zeros_like(x)
  for i,t in enumerate(group):x[i,-len(t['ids']):]=torch.tensor(t['ids'],device='cuda');mask[i,-len(t['ids']):]=1
  pos=(mask.cumsum(-1)-1).clamp(min=0)
  return x,mask,pos
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  # Cache and padding fidelity: two real inputs, two generated steps cached vs full.
  checks=[]
  for task in [tasks[0],tasks[len(tasks)//2]]:
   x,mask,pos=tensors([task]);pre=forward(input_ids=x,attention_mask=mask,position_ids=pos,use_cache=True,logits_to_keep=1)
   assert torch.isfinite(pre.logits).all();token=pre.logits[:,-1].argmax(-1,keepdim=True)
   m=torch.cat([mask,torch.ones_like(token)],-1);p=m.cumsum(-1)-1
   cached=forward(input_ids=token,attention_mask=m,position_ids=p[:,-1:],past_key_values=pre.past_key_values,use_cache=True,logits_to_keep=1)
   full=forward(input_ids=torch.cat([x,token],-1),attention_mask=m,position_ids=p,use_cache=False,logits_to_keep=1)
   same=bool(torch.equal(cached.logits[:,-1].argmax(-1),full.logits[:,-1].argmax(-1)));assert same,'cached/full argmax divergence'
   checks.append(dict(id=task['id'],head=task['head'],cached_full_argmax_equal=same,max_logits_error=float((cached.logits-full.logits).abs().max())))
   del pre,cached,full
  # Padding parity at the first token.
  short,long=tasks[0],max(tasks,key=lambda t:len(t['ids']));group=[short,long]
  x,m,p=tensors(group);bat=forward(input_ids=x,attention_mask=m,position_ids=p,use_cache=False,logits_to_keep=1)
  x,m,p=tensors([short]);single=forward(input_ids=x,attention_mask=m,position_ids=p,use_cache=False,logits_to_keep=1)
  assert int(bat.logits[0,-1].argmax())==int(single.logits[0,-1].argmax()),'left-padding divergence'
  del bat,single
  write(out/'PRECHECK.json',dict(valid=True,cache_checks=checks,left_padding_argmax_equal=True,head_row_error=rowerr,ple_row_error=pleerr))
  done={}
  rawpath=out/'heads.jsonl'
  if rawpath.exists():
   for r in jl(rawpath):
    key=(r['batch'],r['id'],r['head']);assert key not in done;done[key]=r
  selected=tasks[:a.limit] if a.limit else tasks
  pending=sorted([t for t in selected if (t['batch'],t['id'],t['head']) not in done],key=lambda t:len(t['ids']))
  receipt.update(status='generating',head_requests=len(tasks),pending=len(pending));write(out/'run.json',receipt)
  with rawpath.open('a') as raw:
   for offset in range(0,len(pending),a.batch_size):
    group=pending[offset:offset+a.batch_size];x,mask,pos=tensors(group);generated=[[] for _ in group];finished=[False]*len(group);reasons=['max_tokens']*len(group);cache=None
    for step in range(128):
     result=forward(input_ids=x,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True,logits_to_keep=1)
     assert torch.isfinite(result.logits).all(),'nonfinite logits'
     nxt=result.logits[:,-1].argmax(-1);cache=result.past_key_values
     values=nxt.tolist()
     for i,t in enumerate(group):
      if not finished[i]:
       generated[i].append(values[i])
       if values[i]==t['close']:finished[i]=True;reasons[i]='closing_marker'
       elif values[i] in stopids:finished[i]=True;reasons[i]='eos_before_closing_marker'
     if all(finished):break
     x=nxt[:,None];mask=torch.cat([mask,torch.ones_like(x)],-1);pos=(mask.cumsum(-1)-1)[:,-1:]
    for t,ids,reason in zip(group,generated,reasons):
     text=tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False);body=text.split(f'</{t["head"]}>')[0]
     record={k:t[k] for k in ['batch','id','head']};record.update(tokens=ids,output=text,body=body,stop=reason,prompt_tokens=len(t['ids']),prompt_token_sha256=hashlib.sha256(json.dumps(t['ids'],separators=(',',':')).encode()).hexdigest())
     raw.write(json.dumps(record,ensure_ascii=False)+'\n');done[(t['batch'],t['id'],t['head'])]=record
    raw.flush();os.fsync(raw.fileno());del result,cache,x,mask,pos
    receipt.update(done_heads=len(done),updated=time.time());write(out/'run.json',receipt)
    if offset%(a.batch_size*10)==0:print(json.dumps(dict(model=a.model,done=len(done),total=len(selected))),flush=True)
 if a.limit:
  receipt.update(status='smoke_completed',done_heads=len(done));write(out/'run.json',receipt);return
 assert len(done)==len(tasks)
 totals={};stops=collections.Counter(r['stop'] for r in done.values())
 for b in BATCHES:
  dest=out/b;dest.mkdir(exist_ok=True);d=a.datasets/b
  with (dest/'predictions.jsonl').open('w') as f:
   for row in jl(d/'cases.jsonl'):
    heads={t['head']:done[(b,row['id'],t['head'])]['body'] for t in tasks if t['batch']==b and t['id']==row['id']}
    f.write(json.dumps(dict(id=row['id'],heads=heads),ensure_ascii=False)+'\n')
  s=subprocess.run([sys.executable,str(d/'scripts/score.py'),'--cases',str(d/'cases.jsonl'),'--predictions',str(dest/'predictions.jsonl'),'--out',str(dest/'scores.json')],capture_output=True,text=True)
  assert s.returncode==0,s.stderr
  totals[b]=read(dest/'scores.json')['total']
 receipt.update(status='completed',completed=time.time(),totals=totals,stops=dict(stops),peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),evidence_sha256=sha(rawpath));write(out/'run.json',receipt)
 print(json.dumps(dict(model=a.model,status='completed',totals=totals,stops=dict(stops))),flush=True)
if __name__=='__main__':main()
