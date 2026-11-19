import collections,json,sys
from pathlib import Path
from benchlib import ROOT,sha

def main():
 from transformers import AutoTokenizer
 parent=ROOT.parent
 plan=json.loads((parent/'plan.json').read_text())
 tok=AutoTokenizer.from_pretrained(plan['models']['4b']['source'],local_files_only=True)
 assert all(v['tokenizer_sha256']==plan['models']['4b']['tokenizer_sha256'] for v in plan['models'].values())
 cases=[json.loads(s) for s in (ROOT/'cases.jsonl').read_text().splitlines()]
 counts=collections.Counter();lengths=[]
 # Tokenize once per common prompt; all head markers are exactly one added
 # special token in this tokenizer. Check every used marker explicitly.
 for head in ['content','function']+[f'arg{i}' for i in range(1,7)]:
  for token in (f'<{head}>',f'</{head}>'):
   ids=tok.encode(token,add_special_tokens=False)
   assert len(ids)==1 and tok.decode(ids,skip_special_tokens=False)==token,(token,ids)
 for off in range(0,len(cases),128):
  chunk=cases[off:off+128]
  ids=tok([c['prompt'] for c in chunk],add_special_tokens=False)['input_ids']
  for c,tokens in zip(chunk,ids):
   assert c['prompt'].endswith('<|im_start|>assistant\n')
   lengths.append(dict(uid=c['uid'],prompt_tokens=len(tokens)+1))
   counts[c['dataset']]+=1
 assert len(cases)==len({c['uid'] for c in cases})
 maxlen=max(r['prompt_tokens'] for r in lengths)
 assert maxlen+128<=16384,maxlen
 result=dict(status='passed',cases=len(cases),max_prompt_tokens=maxlen,context_limit=16384,
  max_tokens_per_head=128,cases_sha256=sha(ROOT/'cases.jsonl'),tokenizer_sha256=plan['models']['4b']['tokenizer_sha256'],
  dataset_counts=dict(counts),marker_single_token_verified=True,no_context_truncation=True)
 (ROOT/'token-lengths.json').write_text(json.dumps(lengths,indent=2)+'\n')
 (ROOT/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
