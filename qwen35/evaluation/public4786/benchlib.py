"""Frozen benchmark adapter: schema order, raw prompts, local accepted-value scoring."""
import collections,copy,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'upstream'))
from simpletool_eval import rules
from simpletool_eval.strict import strict_call
from simpletool_eval.conversion import parse_arg_value

SYSTEM_PREFIX="You are a model that can do function calling with the following functions\n\nAvailable function tools (properties are ordered; arg1..argN follow the selected function's property order; required parameters come first):\n"
SYSTEM_SUFFIX="\nChoose the function needed for the user's request. Emit one function call per response. Use <|null|> for omitted optional arguments and arguments that do not belong to the selected function. Do not execute tools or add explanatory text to the function or argument channels."

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def normalize_schema(schema,changes,path='parameters'):
    schema=copy.deepcopy(schema)
    # Preserve source type names: dict/float/tuple/any and unusual spellings are
    # benchmark content. Only property order and required order are adapted.
    props=schema.get('properties')
    if isinstance(props,dict):
        required=schema.get('required',[])
        assert isinstance(required,list) and len(required)==len(set(required)),(path,required)
        assert set(required)<=set(props),(path,required,list(props))
        keys=[k for k in props if k in required]+[k for k in props if k not in required]
        ordered_required=[k for k in keys if k in required]
        if keys!=list(props) or ordered_required!=required:
            changes.append(dict(path=path,original_order=list(props),new_order=keys,original_required=required,new_required=ordered_required))
        schema['properties']={k:normalize_schema(props[k],changes,path+'.'+k) for k in keys}
        if 'required' in schema:schema['required']=ordered_required
    if isinstance(schema.get('items'),dict):schema['items']=normalize_schema(schema['items'],changes,path+'.items')
    return schema

def normalize_tools(functions):
    tools=[];changes=[]
    for i,tool in enumerate(functions):
        f=copy.deepcopy(tool.get('function',tool))
        assert isinstance(f.get('name'),str) and isinstance(f.get('parameters'),dict)
        f['parameters']=normalize_schema(f['parameters'],changes,str(i)+'.'+f['name'])
        tools.append(dict(type='function',function=f))
    assert len({t['function']['name'] for t in tools})==len(tools),'duplicate candidate tool names'
    return tools,changes

def build_prompt(messages,tools):
    # Match preceding Qwen3.5 Mobile Actions system style. No reference values
    # are passed to this function. Embedded source environment/history retained.
    system='\n'.join(str(m['content']) for m in messages if m['role']=='system')
    text=(system+'\n' if system else '')+SYSTEM_PREFIX
    text+='\n'.join(json.dumps(t,ensure_ascii=False,separators=(',',':')) for t in tools)
    prompt='<|im_start|>system\n'+text+SYSTEM_SUFFIX+'<|im_end|>\n'
    for m in messages:
        if m['role']=='system':continue
        assert m['role'] in ('user','assistant','tool')
        prompt+='<|im_start|>'+m['role']+'\n'+str(m['content'])+'<|im_end|>\n'
    return prompt+'<|im_start|>assistant\n'

def aggregate(rows):
    metrics=('function_match','required_arguments_match','typed_required_arguments_match','full_local_match','all_heads_closed','tool_heads_closed')
    return dict(n=len(rows),**{k:dict(count=sum(bool(r[k]) for r in rows),fraction=sum(bool(r[k]) for r in rows)/len(rows) if rows else None) for k in metrics})

def typed_value(value,schema):
    kind=str(schema.get('type','any')).lower().rstrip('.')
    if value is None:return False
    if kind in ('string','str'):return isinstance(value,str)
    if kind in ('integer','int'):return type(value) is int
    if kind in ('float','number'):return type(value) in (int,float) and math.isfinite(value)
    if kind in ('boolean','bool'):return type(value) is bool
    if kind in ('array','list','tuple'):return isinstance(value,list)
    if kind in ('dict','object'):return isinstance(value,dict)
    return True

def score_row(raw,case):
    heads={h['name']:h for h in raw['heads']}
    name=heads['function']['body'].strip()
    tools={t['function']['name']:t['function'] for t in case['tools']}
    chosen=tools.get(name)
    arguments={};type_errors=[]
    if chosen:
        for i,(key,schema) in enumerate(chosen['parameters'].get('properties',{}).items(),1):
            body=heads.get(f'arg{i}',{}).get('body','')
            value=parse_arg_value(body)
            if value is not None:arguments[key]=value
            if value is not None and not typed_value(value,schema):type_errors.append(key)
    target=case['ground_truth'][0]
    reference_name=next(iter(target))
    accepted=target[reference_name]
    reference_tool=tools[reference_name]
    required=reference_tool['parameters'].get('required',[])
    function_match=(name==reference_name)
    parameter_matches={k:rules.check_param_match(arguments.get(k),accepted[k] if isinstance(accepted[k],list) else [accepted[k]]) for k in required}
    required_match=function_match and all(parameter_matches.values())
    typed_required=required_match and all(k not in type_errors and k in arguments for k in required)
    call=dict(name=name,arguments=arguments)
    return dict(uid=case['uid'],id=case['id'],dataset=case['dataset'],reference_name=reference_name,
        predicted_name=name,reference_accepted_arguments=accepted,predicted_arguments=arguments,
        required_argument_names=required,required_parameter_matches=parameter_matches,
        type_error_keys=type_errors,function_match=function_match,
        required_arguments_match=required_match,typed_required_arguments_match=typed_required,
        full_local_match=strict_call(call,target),
        all_heads_closed=all(h['stop']=='closing_marker' for h in raw['heads']),
        tool_heads_closed=all(h['stop']=='closing_marker' for h in raw['heads'] if h['name']!='content'))
