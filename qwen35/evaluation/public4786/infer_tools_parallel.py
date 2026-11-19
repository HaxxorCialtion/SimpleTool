"""Tool-head-only quality inference; frozen original infer.py is unchanged."""
import argparse,collections,json,os,time,socket
from pathlib import Path
from benchlib import ROOT,sha,score_row,aggregate

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--model',required=True)
 ap.add_argument('--chunk',type=int,default=64);ap.add_argument('--resume',action='store_true');ap.add_argument('--limit',type=int)
 a=ap.parse_args()
 os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
 os.environ.setdefault('TOKENIZERS_PARALLELISM','false');os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD','spawn')
 os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('VLLM_CACHE_ROOT',str(ROOT/'cache'/a.tag))
 import torch,vllm
 from vllm import LLM,SamplingParams
 cases=[json.loads(s) for s in (ROOT/'cases.jsonl').read_text().splitlines()]
 assert json.loads((ROOT/'preflight.json').read_text())['cases_sha256']==sha(ROOT/'cases.jsonl')
 if a.limit:cases=cases[:a.limit]
 cases=[{**c,'heads':[h for h in c['heads'] if h!='content']} for c in cases]
 output=ROOT/'tools-only/runs'/a.tag;output.mkdir(parents=True,exist_ok=True)
 fingerprint=dict(hostname=socket.gethostname(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),head_policy='function plus candidate-max-arity argument heads; no content',tag=a.tag,model=str(Path(a.model).resolve()),config_sha256=sha(Path(a.model)/'config.json'),
  cases_sha256=sha(ROOT/'cases.jsonl'),script_sha256=sha(__file__),adapter_sha256=sha(ROOT/'benchlib.py'),
  chunk=a.chunk,case_count=len(cases),max_tokens=128,context_limit=16384,dtype='float16',
  kv_cache_dtype='auto (FP16 attention; engine default GDN state)',temperature=0,seed=1107,
  max_num_seqs=192,gpu_memory_utilization=.70,enforce_eager=True,prefix_cache=True,
  vllm=vllm.__version__,torch=torch.__version__,gpu=torch.cuda.get_device_name())
 receipt_path=output/'run.json'
 if receipt_path.exists():
  assert a.resume,'existing run requires --resume'
  receipt=json.loads(receipt_path.read_text());assert receipt['config']==fingerprint,'resume config changed'
 else:
  assert not (output/'outputs.jsonl').exists()
  receipt=dict(status='started',config=fingerprint,started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
 raw=[]
 if (output/'outputs.jsonl').exists():raw=[json.loads(s) for s in (output/'outputs.jsonl').read_text().splitlines()]
 assert [r['uid'] for r in raw]==[c['uid'] for c in cases[:len(raw)]],'only prefix-complete chunks resumable'
 pending=cases[len(raw):]
 if pending:
  start=time.perf_counter()
  llm=LLM(model=a.model,dtype='half',tensor_parallel_size=1,max_model_len=16384,
   gpu_memory_utilization=.70,enforce_eager=True,seed=1107,max_num_seqs=192,
   max_num_batched_tokens=8192,enable_prefix_caching=True,trust_remote_code=False,
   limit_mm_per_prompt={'image':0,'video':0})
  receipt['load_seconds']=time.perf_counter()-start
  receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
  with (output/'outputs.jsonl').open('a') as stream:
   for offset in range(0,len(pending),a.chunk):
    chunk=pending[offset:offset+a.chunk];prompts=[];params=[];mapping=[]
    for c in chunk:
     for head in c['heads']:
      prompts.append(c['prompt']+f'<{head}>');mapping.append((c['uid'],head))
      params.append(SamplingParams(temperature=0,seed=1107,max_tokens=128,stop=[f'</{head}>'],
       skip_special_tokens=False,include_stop_str_in_output=True))
    tick=time.perf_counter();results=llm.generate(prompts,params,use_tqdm=False);elapsed=time.perf_counter()-tick
    assert len(results)==len(prompts)
    grouped=collections.defaultdict(list)
    for (uid,head),result in zip(mapping,results):
     o=result.outputs[0];end=f'</{head}>';closed=end in o.text
     stop='closing_marker' if closed else ('max_tokens' if o.finish_reason=='length' else 'eos_before_closing_marker')
     grouped[uid].append(dict(name=head,body=o.text.split(end)[0],output=o.text,tokens=list(o.token_ids),
      stop=stop,finish_reason=o.finish_reason,stop_reason=o.stop_reason))
    for c in chunk:
     record=dict(uid=c['uid'],id=c['id'],dataset=c['dataset'],heads=grouped[c['uid']],batch_seconds=elapsed)
     assert len(record['heads'])==len(c['heads'])
     raw.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    stream.flush();os.fsync(stream.fileno())
    print(json.dumps(dict(tag=a.tag,completed=len(raw),total=len(cases),last_dataset=chunk[-1]['dataset'],batch_seconds=elapsed)),flush=True)
  # Release engine explicitly, only this private model process.
  engine=getattr(llm,'llm_engine',None)
  if engine is not None and hasattr(engine,'engine_core'):engine.engine_core.shutdown()
 scored=[score_row(r,c) for r,c in zip(raw,cases)]
 assert len(raw)==len(cases)
 by_dataset={}
 for name in dict.fromkeys(c['dataset'] for c in cases):by_dataset[name]=aggregate([r for r in scored if r['dataset']==name])
 heads=[h for r in raw for h in r['heads']]
 stats=dict(all=aggregate(scored),datasets=by_dataset,
  stops=dict(collections.Counter(h['stop'] for h in heads)),
  replacement_character_heads=sum('\ufffd' in h['body'] for h in heads),total_heads=len(heads),
  length_limit_by_head=dict(collections.Counter(h['name'] for h in heads if h['stop']=='max_tokens')),
  early_eos_by_head=dict(collections.Counter(h['name'] for h in heads if h['stop']=='eos_before_closing_marker')),
  scored_rows=scored)
 (output/'scores.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
 receipt.update(status='completed',completed_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
  output_sha256=sha(output/'outputs.jsonl'),scores_sha256=sha(output/'scores.json'),actual_case_count=len(raw))
 receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(dict(tag=a.tag,status='completed',all=stats['all'],stops=stats['stops']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
