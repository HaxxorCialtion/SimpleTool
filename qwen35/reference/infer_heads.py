"""OpenAI-compatible completion client. Prints proposals; never executes tools."""
import argparse, concurrent.futures, json, os, urllib.request

HEADS=['function']+['arg'+str(i) for i in range(1,7)]
def build_prompt(tools,query,environment=None,history=None):
    for tool in tools:
        schema=tool['function']['parameters'];keys=list(schema.get('properties',{}));required=schema.get('required',[])
        assert keys[:len(required)]==required,'Required arguments must be first, in the same order.'
    system=('You are a model that can do function calling with the following functions\n\n'
        "Available function tools (properties are ordered; arg1..argN follow the selected function's property order; required parameters come first):\n"
        +'\n'.join(json.dumps(t,ensure_ascii=False,separators=(',',':')) for t in tools)
        +"\nChoose the function needed for the user's request. Emit one function call per response. Use <|null|> for omitted optional arguments and arguments that do not belong to the selected function. Do not execute tools or add explanatory text to the function or argument channels.")
    user=json.dumps(dict(environment=environment or {},history=history or [],query=query),ensure_ascii=False)
    return '<|im_start|>system\n'+system+'<|im_end|>\n<|im_start|>user\n'+user+'<|im_end|>\n<|im_start|>assistant\n'

def generate(base,model,common,tools,protocol='empty-think',api_key=None):
    max_args=max(len(t['function']['parameters'].get('properties',{})) for t in tools)
    assert 0<=max_args<=6
    prefix=common+('<think>\n\n</think>\n\n' if protocol=='empty-think' else '')
    def one(head):
        closing='</'+head+'>'
        data=dict(model=model,prompt=prefix+'<'+head+'>',temperature=0,seed=1107,max_tokens=128,
                  stop=[closing],skip_special_tokens=False,include_stop_str_in_output=True)
        headers={'Content-Type':'application/json'}
        if api_key:headers['Authorization']='Bearer '+api_key
        req=urllib.request.Request(base.rstrip('/')+'/completions',data=json.dumps(data).encode(),headers=headers)
        with urllib.request.urlopen(req,timeout=180) as r:response=json.load(r)
        choice=response['choices'][0];text=choice['text'];closed=closing in text
        return head,dict(raw=text,value=text.split(closing)[0],closed=closed,finish_reason=choice['finish_reason'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,max_args)+1) as pool:
        outputs=dict(pool.map(one,HEADS[:max(1,max_args)+1]))
    if not all(x['closed'] for x in outputs.values()):return dict(valid=False,reason='unclosed_head',heads=outputs)
    name=outputs['function']['value'];schemas={t['function']['name']:t['function']['parameters'] for t in tools}
    if name not in schemas:return dict(valid=False,reason='unknown_function',heads=outputs)
    args={}
    for i,(key,spec) in enumerate(schemas[name].get('properties',{}).items(),1):
        value=outputs['arg'+str(i)]['value']
        if value=='<|null|>':continue
        # String channels are raw text; all other types must be valid JSON.
        args[key]=value if spec.get('type')=='string' else json.loads(value)
    return dict(valid=True,name=name,arguments=args,heads=outputs)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base-url',default='http://127.0.0.1:8000/v1');p.add_argument('--model',required=True)
    p.add_argument('--protocol',choices=['direct','empty-think'],default='empty-think');a=p.parse_args()
    tools=[dict(type='function',function=dict(name='set_brightness',description='Set display brightness.',parameters=dict(type='object',properties=dict(percent=dict(type='integer',minimum=0,maximum=100)),required=['percent'],additionalProperties=False)))]
    prompt=build_prompt(tools,'Set screen brightness to 35 percent.')
    print(json.dumps(generate(a.base_url,a.model,prompt,tools,a.protocol,os.environ.get('OPENAI_API_KEY')),ensure_ascii=False,indent=2))
