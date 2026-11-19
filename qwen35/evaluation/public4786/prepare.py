"""Freeze all 18 QZ SimpleTool source datasets under current owner's protocol."""
import collections,json,re,sys
from pathlib import Path
from benchlib import ROOT,sha,normalize_tools,build_prompt
from simpletool_eval.data import load_pair

def main():
    data=ROOT/'upstream/data'; manifest=json.loads((data/'manifest.json').read_text())
    expected=json.loads((ROOT/'source-receipt.json').read_text())
    for rel,digest in expected['files'].items():assert sha(ROOT/'upstream'/rel)==digest,rel
    cases=[];exclusions=[];changes=[];summary={};warnings=[]
    for dataset,info in manifest['datasets'].items():
        rows,gt,_=load_pair(data,info['filename'])
        mobile_bad=set()
        if dataset=='mobile_actions':
            groups=collections.Counter(r['id'].rsplit('_',1)[0] for r in rows)
            mobile_bad={g for g,n in groups.items() if n>1}
            assert len(groups)==961 and len(mobile_bad)==321
        counts=collections.Counter(input=len(rows))
        for row in rows:
            uid=dataset+'/'+row['id'];reason=None
            if row['id'] not in gt:reason='missing_or_invalid_ground_truth'
            elif dataset=='mobile_actions' and row['id'].rsplit('_',1)[0] in mobile_bad:reason='entire_multi_call_mobile_task'
            elif len(gt[row['id']])!=1:reason='not_exactly_one_reference_call'
            elif any(len(f.get('function',f).get('parameters',{}).get('properties',{}))>6 for f in row['function']):reason='any_candidate_exceeds_six_parameters'
            if reason:
                exclusions.append(dict(uid=uid,reason=reason));counts[reason]+=1;continue
            messages=row['question'][0]
            if len(row['question'])!=1 or any(m.get('history') for m in messages):raise ValueError(('unexpected multi-turn',uid))
            tools,ordering=normalize_tools(row['function'])
            target=gt[row['id']][0]
            assert len(target)==1,(uid,target)
            name=next(iter(target));tool=next((t['function'] for t in tools if t['function']['name']==name),None)
            if tool is None:reason='reference_function_absent_from_candidates'
            else:
                required=tool['parameters'].get('required',[])
                if set(required)-set(target[name]):reason='missing_required_parameter_annotation'
            if reason:
                exclusions.append(dict(uid=uid,reason=reason));counts[reason]+=1;continue
            prompt=build_prompt(messages,tools)
            maxarity=max(len(t['function']['parameters'].get('properties',{})) for t in tools)
            heads=['content','function']+[f'arg{i}' for i in range(1,maxarity+1)]
            case=dict(uid=uid,id=row['id'],dataset=dataset,tools=tools,messages=messages,prompt=prompt,heads=heads,
                ground_truth=gt[row['id']],original_input_sha256=sha(data/info['filename']),ordering_changed=bool(ordering))
            cases.append(case);counts['included']+=1
            if ordering:changes.append(dict(uid=uid,changes=ordering));counts['order_adapted']+=1
        assert counts['input']==counts['included']+sum(v for k,v in counts.items() if k not in ('input','included','order_adapted'))
        summary[dataset]=dict(counts)
    # Source Mobile Actions is the same underlying 640 single-task subset but
    # preserves its original environment/history text and original type names.
    assert summary['mobile_actions']['included']==640
    with (ROOT/'cases.jsonl').open('w') as f:
        for c in cases:f.write(json.dumps(c,ensure_ascii=False)+'\n')
    for fn,value in [('exclusions.json',exclusions),('order-adaptations.json',changes)]:
        (ROOT/fn).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    result=dict(status='prepared',source_commit=expected['commit'],cases_sha256=sha(ROOT/'cases.jsonl'),
        source_manifest_sha256=sha(data/'manifest.json'),datasets=summary,total_input=sum(x['input'] for x in summary.values()),
        total_included=len(cases),total_excluded=len(exclusions),total_heads=sum(len(c['heads']) for c in cases),
        policy=dict(single_call_only=True,mobile_entire_multi_task_excluded=True,required_first=True,
            candidate_order_preserved=True,required_list_follows_properties=True,source_type_names_preserved=True,
            source_environment_and_history_preserved=True,max_candidate_parameters=6,gt_never_in_prompt=True,
            primary_metric='exact function + required parameters via frozen upstream local acceptable-value matcher',
            full_metric='strict_call: all annotated parameters, reject extras; local matcher, not official BFCL',
            heads='content,function,arg1..argN; N=max arity of all candidate tools, independent of reference'),warnings=warnings)
    (ROOT/'preparation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
