"""Native SimpleTool Gemma adapter inference. Prints a proposal; executes no tools."""
import argparse,json
from pathlib import Path

def generate(base,artifact,messages,heads,max_tokens=128):
 import torch
 from transformers import AutoTokenizer
 import modeling
 artifact=Path(artifact)
 tok=AutoTokenizer.from_pretrained(artifact/'tokenizer',local_files_only=True)
 hids={h:tok.convert_tokens_to_ids('<'+h+'>') for h in modeling.HEADS}
 for h in heads:
  assert tok.encode('<'+h+'>',add_special_tokens=False)==[hids[h]]
 model=modeling.load_adapters(base,artifact,tok,hids,torch.device('cuda'))
 model.eval();model.requires_grad_(False);model.set_attn_implementation('sdpa');model.config.use_cache=True
 common=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 outputs={}
 for h in heads:
  ids=torch.tensor([tok.encode(common+'<'+h+'>',add_special_tokens=False)],device='cuda')
  close=tok.convert_tokens_to_ids('</'+h+'>');stop={close,tok.eos_token_id}
  generated=[];past=None
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
   for _ in range(max_tokens):
    result=model(input_ids=ids,past_key_values=past,use_cache=True)
    token=int(result.logits[0,-1].argmax());generated.append(token);past=result.past_key_values
    if token in stop:break
    ids=torch.tensor([[token]],device='cuda')
  raw=tok.decode(generated,skip_special_tokens=False)
  outputs[h]={'raw':raw,'closed':generated[-1]==close,'value':raw.split('</'+h+'>')[0]}
 return {'heads':outputs,'tools_executed':False,'note':'Reference sequential decode; not a parallel speed benchmark.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--artifact',default=str(Path(__file__).resolve().parent));p.add_argument('--messages',type=Path,required=True);p.add_argument('--args',type=int,default=1);p.add_argument('--content',action='store_true');p.add_argument('--max-tokens',type=int,default=128);a=p.parse_args()
 assert 0<=a.args<=6
 heads=(['content'] if a.content else [])+['function']+[f'arg{i}' for i in range(1,max(1,a.args)+1)]
 print(json.dumps(generate(a.base,a.artifact,json.loads(a.messages.read_text()),heads,a.max_tokens),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
