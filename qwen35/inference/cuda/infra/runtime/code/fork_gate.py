"""Opt-in scheduler gate for a native SimpleTool request group.
All requests enter the ordinary waiting queue; cancellation/counting remain native.
No tensor/state alias bypass: normal prefix hashes and managers decide reuse.
"""
import os,time
from collections import deque

def meta(request):
    p=request.sampling_params
    return (p.extra_args or {}).get('simpletool_fork') if p is not None else None

def trace(s,event,request,**kw):
    if os.environ.get('SL_FORK_TRACE')!='1':return
    if not hasattr(s,'_sl_fork_events'):s._sl_fork_events=deque(maxlen=10000)
    s._sl_fork_events.append(dict(event=event,at_perf=time.perf_counter(),request_id=request.request_id,**kw))

def on_add(s,request):
    m=meta(request)
    if not m:return
    if request.resumable or request.mm_features or s.connector is not None:
        raise ValueError('SimpleTool fork experiment supports native text-only, non-streaming, no connector')
    group=m['group'];role=m['role'];n=m['N']
    if not isinstance(group,str) or role not in ('anchor','branch') or not isinstance(n,int) or n<1:
        raise ValueError('Invalid SimpleTool fork metadata')
    if len(request.prompt_token_ids)!=n+1:raise ValueError('Fork expects common N plus one marker')
    groups=getattr(s,'_sl_fork_groups',None)
    if groups is None:groups=s._sl_fork_groups={}
    if role=='anchor':
        if group in groups:raise ValueError('Duplicate live fork group')
        groups[group]=request.request_id
    else:
        anchor=s.requests.get(groups.get(group))
        if anchor is None:raise ValueError('Submit group anchor before its siblings, before stepping engine')
        if request.prompt_token_ids[:n]!=anchor.prompt_token_ids[:n]:raise ValueError('Group common prefix differs')
    trace(s,'enqueue_'+role,request,group=group,N=n)

def blocked(s,request):
    m=meta(request)
    if not m or m['role']!='branch':return False
    anchor=s.requests.get(getattr(s,'_sl_fork_groups',{}).get(m['group']))
    if anchor is None:return False # Anchor completed/aborted; native cache lookup or cold fallback.
    confirmed=anchor.num_computed_tokens-max(anchor.num_output_placeholders,anchor.num_in_flight_tokens)
    ready=confirmed>=anchor.num_prompt_tokens
    trace(s,'branch_ready' if ready else 'branch_hold',request,group=m['group'],confirmed=confirmed)
    return not ready

def before_free(s,request):
    m=meta(request)
    if m and m['role']=='anchor':
        getattr(s,'_sl_fork_groups',{}).pop(m['group'],None)
        getattr(s,'_sl_fork_published',set()).discard(m['group'])
        trace(s,'anchor_free',request,group=m['group'])

def after_update(s):
    for group,rid in getattr(s,'_sl_fork_groups',{}).items():
        r=s.requests.get(rid)
        if r is None:continue
        confirmed=r.num_computed_tokens-max(r.num_output_placeholders,r.num_in_flight_tokens)
        published=getattr(s,'_sl_fork_published',None)
        if published is None:published=s._sl_fork_published=set()
        if confirmed>=r.num_prompt_tokens and group not in published:
            published.add(group)
            trace(s,'anchor_confirmed',r,group=group,confirmed=confirmed)
