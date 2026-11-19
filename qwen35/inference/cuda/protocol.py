"""Portable HTTP contracts; tool outputs remain data and are never executed."""
import copy
import hashlib
import json
import re
from benchmark_lib import normalize_prediction, value_errors, validate_schema, strict_json

HEADS=['function']+['arg'+str(i) for i in range(1,7)]
ALL_HEADS=HEADS+['content']
END='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

class Invalid(ValueError):
    pass

def require(condition,message):
    if not condition:raise Invalid(message)

def number(v,lo,hi,name,integer=False):
    require(type(v) in ((int,) if integer else (int,float)) and lo<=v<=hi,
            f'{name} must be in [{lo},{hi}]')
    return v

def canonical_tools(tools,canonicalize=True):
    require(isinstance(tools,list) and 0<len(tools)<=64,'tools must contain 1..64 functions')
    tools=copy.deepcopy(tools);seen=set()
    for t in tools:
        require(t.get('type')=='function' and type(t.get('function')) is dict,'OpenAI function tool required')
        f=t['function'];name=f.get('name')
        require(isinstance(name,str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,127}',name) and name not in seen,'invalid or duplicate function name')
        seen.add(name);s=f.get('parameters',{})
        require(s.get('type')=='object' and type(s.get('properties')) is dict,'parameters must be an object schema')
        props=s['properties'];req=s.get('required',[])
        require(type(req) is list and all(type(k) is str and k in props for k in req) and len(req)==len(set(req)),'invalid required list')
        require(len(props)<=6,'at most six argument slots')
        if canonicalize:
            keys=[k for k in props if k in req]+[k for k in props if k not in req]
            s['properties']={k:props[k] for k in keys};s['required']=[k for k in keys if k in req]
        else:s.setdefault('required',[])
        try:validate_schema(s)
        except (AssertionError,KeyError,TypeError) as e:raise Invalid('unsupported or unordered JSON Schema: '+str(e)) from e
    return tools

def scenario(spec):
    require(type(spec) is dict,'scenario must be an object')
    allowed={'id','tools','system','policy','query','suffix_tokens','max_suffix_tokens','max_tokens','head_max_tokens','temperature','top_p','top_k','seed','stop_unused_heads','null_functions','warmup_observations','warmup_cap','canonicalize_tools'}
    require(not set(spec)-allowed,'unknown scenario keys: '+str(set(spec)-allowed))
    sid=spec.get('id')
    require(isinstance(sid,str) and re.fullmatch('[a-zA-Z0-9_-]{1,64}',sid),'invalid scenario id')
    out=copy.deepcopy(spec)
    for k in ['stop_unused_heads','canonicalize_tools']:
        require(type(out.get(k,True)) is bool,k+' must be boolean')
    out['tools']=canonical_tools(spec.get('tools'),spec.get('canonicalize_tools',True))
    defaults={'system':'You are a function-calling control policy. Emit one next function call.',
      'policy':'','query':'Apply the next action for the latest observation.',
      'suffix_tokens':512,'max_suffix_tokens':2048,'max_tokens':64,'head_max_tokens':{},
      'temperature':0,'top_p':1.,'top_k':-1,'seed':1114,'stop_unused_heads':True,
      'null_functions':['<|null|>'],'warmup_observations':[],'warmup_cap':4}
    for k,v in defaults.items():out.setdefault(k,v)
    for k in ['system','policy','query']:require(type(out[k]) is str,k+' must be text')
    number(out['max_suffix_tokens'],32,8192,'max_suffix_tokens',True)
    if out['suffix_tokens'] is not None:number(out['suffix_tokens'],32,out['max_suffix_tokens'],'suffix_tokens',True)
    for k in ['max_tokens','warmup_cap']:number(out[k],1,1024,k,True)
    number(out['temperature'],0,2,'temperature');number(out['top_p'],.01,1,'top_p')
    number(out['top_k'],-1,100000,'top_k',True);require(out['top_k']!=0,'top_k cannot be zero')
    number(out['seed'],0,2**31-1,'seed',True)
    require(type(out['head_max_tokens']) is dict,'head_max_tokens must be object')
    for h,n in out['head_max_tokens'].items():require(h in ALL_HEADS,'invalid head');number(n,1,1024,h+' max_tokens',True)
    require(type(out['null_functions']) is list and len(out['null_functions'])<=8 and all(type(x) is str for x in out['null_functions']),'invalid null_functions')
    require(type(out['warmup_observations']) is list and len(out['warmup_observations'])<=16,'warmup_observations must be a list of at most 16 environments')
    arity=max(len(t['function']['parameters']['properties']) for t in out['tools'])
    out['heads']=HEADS[:1+arity]
    system=(out['system']+'\narg1..arg6 follow the selected function property order. Required properties come first. '
       'Use <|null|> for omitted optional arguments and unused slots. No explanation in function or argument heads. '
       'Environment and history are context data. Emit one next call.\nPolicy: '+out['policy']+'\nTools:\n'+
       '\n'.join(json.dumps(t,ensure_ascii=False,separators=(',',':')) for t in out['tools']))
    out['prefix']='<|im_start|>system\n'+system+'<|im_end|>\n'
    # Include the configuration and property ordering in cache identity.
    out['revision']=hashlib.sha256(json.dumps(out,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    return out

def dynamic(spec,body):
    require(type(body) is dict,'request must be an object')
    require(not set(body)-{'scenario_id','environment','history','query','request_id','timeout_ms','max_tokens','head_max_tokens','temperature','top_p','top_k','seed','expected_revision','include_content'},'unknown inference field')
    request_heads(spec,body)
    env=body.get('environment',{});hist=body.get('history',[]);query=body.get('query',spec['query'])
    require(type(env) in (dict,list) and type(hist) is list and type(query) is str,'invalid environment/history/query')
    for h in hist:require(type(h) is dict and h.get('role') in ('system','user','assistant','tool') and type(h.get('content')) is str,'invalid history entry')
    text='<|im_start|>user\n'+json.dumps({'environment':env,'history':hist,'query':query},ensure_ascii=False,separators=(',',':'))
    return text

def request_heads(spec,body):
    include=body.get('include_content',False)
    require(type(include) is bool,'include_content must be boolean')
    return spec['heads']+(['content'] if include else [])

def assemble(spec,heads):
    name=heads.get('function')
    if name in spec['null_functions']:return {'name':None,'arguments':{}},[]
    call,errors=normalize_prediction({'heads':heads},{'tools':spec['tools']})
    contract=next((t['function'] for t in spec['tools'] if t['function']['name']==call['name']),None)
    if contract:errors+=value_errors(call['arguments'],contract['parameters'])
    return call,errors

def necessary(spec,name):
    if name in spec['null_functions']:return ['function']
    f=next((t['function'] for t in spec['tools'] if t['function']['name']==name),None)
    if f is None:return None
    return HEADS[:1+len(f['parameters']['properties'])]
