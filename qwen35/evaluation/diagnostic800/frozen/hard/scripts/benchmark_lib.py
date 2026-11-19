"""Small, dependency-free dataset validator and typed literal scorer.

This validates ONLY the JSON Schema subset used in this benchmark, not arbitrary
JSON Schema. No network, model loading or tool execution.
"""
import copy
import json
import math
import re

INPUT_KEYS = ('tools', 'environment', 'history', 'query')
SCHEMA_KEYS = {'type','description','properties','required','additionalProperties',
               'items','enum','minimum','maximum','minItems','maxItems'}

def strict_json(text):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise ValueError('duplicate JSON key '+k)
            out[k]=v
        return out
    def nonfinite(value): raise ValueError('nonfinite JSON '+value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)

def typed_equal(a,b):
    # JSON has number, not a semantically meaningful 10 vs 10.0 distinction.
    if type(a) in (int,float) and type(b) in (int,float):
        return math.isfinite(a) and math.isfinite(b) and a==b
    if type(a) is not type(b): return False
    if isinstance(a,dict): return a.keys()==b.keys() and all(typed_equal(v,b[k]) for k,v in a.items())
    if isinstance(a,list): return len(a)==len(b) and all(typed_equal(x,y) for x,y in zip(a,b))
    return a==b

def esc(s): return str(s).replace('~','~0').replace('/','~1')

def diffs(a,b,path=''):
    if type(a) is not type(b): return [path]
    if isinstance(a,dict):
        return [p for k in dict.fromkeys([*a,*b]) for p in
                ([path+'/'+esc(k)] if k not in a or k not in b else diffs(a[k],b[k],path+'/'+esc(k)))]
    if isinstance(a,list):
        if len(a)!=len(b): return [path]
        return [p for i,(x,y) in enumerate(zip(a,b)) for p in diffs(x,y,path+'/'+str(i))]
    return [] if a==b else [path]

def pointer(obj,path):
    for k in path.strip('/').split('/'):
        k=k.replace('~1','/').replace('~0','~')
        obj=obj[int(k)] if isinstance(obj,list) else obj[k]
    return obj

def validate_schema(schema,path='schema'):
    assert set(schema)<=SCHEMA_KEYS,(path,'unsupported schema key',set(schema)-SCHEMA_KEYS)
    kind=schema['type']; assert kind in ('object','array','string','integer','number','boolean','null'),path
    if kind=='object':
        props=schema['properties']; req=schema['required']
        assert isinstance(props,dict) and isinstance(req,list),path
        assert len(req)==len(set(req)) and set(req)<=props.keys(),path
        assert list(props)[:len(req)]==req,(path,'required must come first in property order')
        for k,v in props.items():validate_schema(v,path+'/'+k)
    elif kind=='array':validate_schema(schema['items'],path+'/items')

def value_errors(value,schema,path=''):
    kind=schema['type']
    ok={'object':lambda:type(value) is dict,'array':lambda:type(value) is list,
        'string':lambda:type(value) is str,'integer':lambda:type(value) is int,
        'number':lambda:type(value) in (int,float) and math.isfinite(value),
        'boolean':lambda:type(value) is bool,'null':lambda:value is None}[kind]()
    if not ok:return [path+': expected '+kind]
    errs=[]
    if 'enum' in schema and not any(typed_equal(value,v) for v in schema['enum']):errs.append(path+': enum')
    if kind=='object':
        props=schema['properties']
        errs += [path+'/'+esc(k)+': missing required' for k in schema['required'] if k not in value]
        errs += [path+'/'+esc(k)+': unknown key' for k in value if k not in props]
        for k in value:
            if k in props:errs+=value_errors(value[k],props[k],path+'/'+esc(k))
    if kind=='array':
        for i,v in enumerate(value):errs+=value_errors(v,schema['items'],path+'/'+str(i))
        for key,check in [('minItems',lambda x:len(value)<x),('maxItems',lambda x:len(value)>x)]:
            if key in schema and check(schema[key]):errs.append(path+': '+key)
    if kind in ('integer','number'):
        if 'minimum' in schema and value<schema['minimum']:errs.append(path+': minimum')
        if 'maximum' in schema and value>schema['maximum']:errs.append(path+': maximum')
    return errs

def input_object(row):return {k:row[k] for k in INPUT_KEYS}

def render_prompt(row):
    """Serialize ONLY model input; labels/tags/rationale never enter prompt."""
    tools='\n'.join(json.dumps(t,ensure_ascii=False,separators=(',',':')) for t in row['tools'])
    system=('You are a model that can do function calling with the following functions\n\n'
        'Available function tools (properties are ordered; arg1..argN follow the selected '
        "function's property order; required parameters come first):\n"+tools+'\n'
        "Choose the function needed for the user's request. Emit one function call per response. "
        'Use <|null|> for omitted optional arguments and arguments that do not belong to the selected function. '
        'Do not execute tools or add explanatory text to the function or argument channels.\n'
        'Environment and supplied history are context data. Answer only the current query with one next call. '
        'Strings and identifiers are exact; arrays retain their stated order. Dates, units and selection rules '
        'come from the provided context and tool contracts. '
        'Date strings use YYYY-MM-DD; time-only strings use HH:MM. Full timestamps use '
        'YYYY-MM-DDTHH:MM:SS followed by Z for UTC or the explicit offset required by the tool '
        '(e.g. +08:00), without fractional seconds.')
    # JSON context envelope prevents supplied observations being confused with live chat roles.
    user=json.dumps({k:row[k] for k in ('environment','history','query')},ensure_ascii=False,separators=(',',':'))
    return '<|im_start|>system\n'+system+'<|im_end|>\n<|im_start|>user\n'+user+'<|im_end|>\n<|im_start|>assistant\n'

def candidate_schemas(row):
    return {t['function']['name']:t['function']['parameters'] for t in row['tools']}

def validate_rows(rows):
    assert len(rows)==200,len(rows)
    ids=set(); groups={}; prompts=set()
    for row in rows:
        rid=row['id']; assert rid not in ids,rid; ids.add(rid)
        assert row['family'] in ('environment','history','binding','logic','transformation'),rid
        assert row['language'] in ('zh','en'),rid
        assert row['tags'] and len(set(row['tags']))==len(row['tags']),rid
        assert all(re.fullmatch('[a-z][a-z0-9_]*',x) for x in row['tags']),rid
        assert row['objective'] and row['rationale'] and row['query'],rid
        assert type(row['environment']) is dict and type(row['history']) is list,rid
        assert all(set(h)=={'role','content'} and h['role'] in ('user','assistant','tool') and type(h['content']) is str for h in row['history']),rid
        names=[]
        for t in row['tools']:
            assert t['type']=='function',rid
            f=t['function'];names.append(f['name']);s=f['parameters']
            assert s['type']=='object' and len(s['properties'])<=6,rid
            validate_schema(s,rid+'/'+f['name'])
        assert len(set(names))==len(names),rid
        expected=row['expected'];assert set(expected)=={'name','arguments'} and expected['name'] in names,rid
        errors=value_errors(expected['arguments'],candidate_schemas(row)[expected['name']])
        assert not errors,(rid,errors)
        p=render_prompt(row);assert p not in prompts,('duplicate prompt',rid);prompts.add(p)
        # The renderer ignores all non-input fields, even when a label is malicious.
        trap=copy.deepcopy(row)
        for k in ('expected','objective','tags','rationale','contrast'):trap[k]='METADATA_LEAK_SENTINEL'
        assert render_prompt(trap)==p,rid
        groups.setdefault(row['pair_id'],[]).append(row)
    assert len(groups)==100,len(groups)
    for pid,pair in groups.items():
        assert len(pair)==2 and {r['variant'] for r in pair}=={'a','b'},pid
        a,b=sorted(pair,key=lambda r:r['variant']);contrast=a['contrast']
        assert contrast==b['contrast'],pid
        assert a['family']==b['family'] and a['tags']==b['tags'],pid
        assert diffs(input_object(a),input_object(b))==[contrast['path']],(pid,'not exactly one changed leaf')
        assert contrast['source']==contrast['path'].split('/')[1],pid
        change=diffs(a['expected'],b['expected'])
        assert sorted(change)==sorted(contrast['expected_changed_paths']),(pid,change,contrast)
        assert contrast['effect'] in ('change','invariant'),pid
        assert bool(change)==(contrast['effect']=='change'),pid
    return groups

def normalize_prediction(pred,row):
    """Accept structured calls or explicitly serialized function/arg heads.

    Head strings are already decoded bodies with start/stop markers removed.
    Strings are raw text; numeric/bool/object/array values must be strict JSON.
    Optional omission uses exact <|null|>. Extra heads beyond selected arity ignored.
    """
    if 'heads' not in pred:
        return dict(name=pred.get('name'),arguments=pred.get('arguments')),[]
    heads=pred['heads'];assert isinstance(heads,dict),'heads must be object'
    name=heads.get('function');schemas=candidate_schemas(row)
    if type(name) is not str or name not in schemas:return dict(name=name,arguments={}),['unknown function']
    args={};errors=[];schema=schemas[name]
    for i,(key,s) in enumerate(schema['properties'].items(),1):
        head='arg'+str(i)
        if head not in heads:
            if key in schema['required']:errors.append('missing '+head)
            continue
        text=heads[head]
        if type(text) is not str:errors.append('head body must be string: '+head);continue
        if text=='<|null|>' and key not in schema['required']:continue
        if s['type']=='string':value=text
        else:
            try:value=strict_json(text)
            except (ValueError,TypeError):errors.append('invalid JSON: '+head);continue
        args[key]=value
    return dict(name=name,arguments=args),errors

def score_case(pred,row):
    try:call,errors=normalize_prediction(pred,row)
    except (ValueError,TypeError,AssertionError) as e:call=dict(name=None,arguments=None);errors=[str(e)]
    schemas=candidate_schemas(row);name=call['name'];fn=name==row['expected']['name']
    if type(name) is str and name in schemas:errors+=value_errors(call['arguments'],schemas[name])
    else:errors.append('function not in candidate set')
    exp=row['expected'];required=candidate_schemas(row)[exp['name']]['required']
    args=call['arguments'] if isinstance(call['arguments'],dict) else {}
    required_ok=fn and all(k in args and typed_equal(args[k],exp['arguments'][k]) for k in required)
    return dict(id=row['id'],name_correct=fn,schema_valid=not errors,
        required_correct=required_ok and not errors,exact_correct=typed_equal(call,exp) and not errors,
        predicted=call,errors=errors)
