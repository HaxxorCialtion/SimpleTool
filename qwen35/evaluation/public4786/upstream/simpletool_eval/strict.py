"""Strict, auditable scorer for the recovered SimpleTool outputs.

This intentionally differs from the historical local scorer: it requires the
predicted call count and rejects extra argument names. It still uses the local
GT acceptable-value lists and six-parameter exclusion contract.
"""
import argparse, json
from pathlib import Path
from .conversion import convert_raw_heads_to_fc
from .data import load_pair, read_jsonl, selected_names, sha256
from . import rules


def exact_value(pred, accepted):
    vals=accepted if isinstance(accepted,list) else [accepted]
    return any(rules.values_equal(pred,v) for v in vals if v != '')


def strict_call(pred, gt_call):
    if not isinstance(pred,dict) or 'name' not in pred: return False
    name=pred.get('name'); args=pred.get('arguments',{})
    if name not in gt_call: return False
    params=gt_call[name]
    if set(args)-set(params): return False
    for key,accepted in params.items():
        vals=accepted if isinstance(accepted,list) else [accepted]
        optional='' in vals
        if key not in args:
            if not optional: return False
        elif not exact_value(args[key],vals): return False
    return True


def strict_prediction(pred, gt, *, history_present=False):
    # Mobile Actions conversion creates one row per sequential call. The first
    # row has empty history and targets gt[0]; later rows have prior calls in
    # history and target the final GT call. SimpleTool emits one call per row.
    if len(gt)>1:
        target = gt[-1] if history_present else gt[0]
    else:
        target = gt[0] if gt else {}
    if len(pred)!=1 or not target: return False
    return strict_call(pred[0],target)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--data',type=Path,default=Path('data')); p.add_argument('--output',type=Path,default=Path('reports/strict')); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    all_results={}
    for name in selected_names():
        ds=Path(name).stem; rows,gt,info=load_pair(a.data,name); path=a.predictions/f'{ds}.jsonl'
        if not path.exists(): continue
        preds,_=read_jsonl(path); inp={r['id']:r for r in rows}; s={'input':len(rows),'predicted':len(preds),'scored':0,'filtered':0,'missing_gt':0,'correct':0,'function_correct':0,'missing_prediction_ids':sorted(set(inp)-{p['id'] for p in preds}),'multi_call_gt':sum(len(v)>1 for v in gt.values()),'strict_fail_reasons':{'call_count_or_shape':0,'wrong_function_or_args':0}}
        pred_ids=[p['id'] for p in preds]
        if len(pred_ids) != len(set(pred_ids)):
            raise ValueError(f'{path}: duplicate prediction IDs')
        unknown=set(pred_ids)-set(inp)
        if unknown:
            raise ValueError(f'{path}: unknown prediction IDs: {sorted(unknown)[:5]}')
        if set(pred_ids) != set(inp):
            raise ValueError(f'{path}: incomplete predictions; missing {len(set(inp)-set(pred_ids))} IDs')
        details=[]
        for item in preds:
            iid=item['id']; d={'id':iid};
            if iid not in gt: s['missing_gt']+=1; d['excluded']='missing_ground_truth'
            elif rules.has_tool_exceeding_max_params(inp[iid]): s['filtered']+=1; d['excluded']='tool_exceeds_six_parameters'
            else:
                pred=item.get('result') or convert_raw_heads_to_fc(item.get('raw_heads',{}),inp[iid].get('function',[])); s['scored']+=1; history_present=bool(inp[iid].get('question') and inp[iid]['question'][0] and inp[iid]['question'][0][0].get('history'))
                ok=strict_prediction(pred,gt[iid],history_present=history_present); s['correct']+=ok
                target_gt=(gt[iid][-1] if history_present and len(gt[iid])>1 else gt[iid][0]) if gt[iid] else {}
                if len(pred)==1 and all(isinstance(pp,dict) and pp.get('name') in target_gt for pp in pred): s['function_correct']+=1
                if not ok:
                    if len(pred)!=len(gt[iid]): s['strict_fail_reasons']['call_count_or_shape']+=1
                    else: s['strict_fail_reasons']['wrong_function_or_args']+=1
                d.update(prediction=pred,strict_correct=ok)
            details.append(d)
        s['accuracy']=s['correct']/s['scored'] if s['scored'] else 0; s['function_accuracy']=s['function_correct']/s['scored'] if s['scored'] else 0
        all_results[ds]=s; (a.output/f'{ds}.details.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in details))
    groups={'BFCL Non-Live':['BFCL_v3_simple','BFCL_v3_multiple'],'BFCL Live':['BFCL_v3_live_simple','BFCL_v3_live_multiple'],'BFCL Exec':['BFCL_v3_exec_simple_dataset','BFCL_v3_exec_multiple_dataset'],'Mobile Actions':['mobile_actions'],'Others':[n for n in all_results if not n.startswith('BFCL') and n!='mobile_actions']}
    agg={}
    for g,ns in groups.items():
        present=[all_results[n] for n in ns if n in all_results]; total=sum(x['scored'] for x in present); agg[g]={'accuracy':sum(x['correct'] for x in present)/total if total else 0,'function_accuracy':sum(x['function_correct'] for x in present)/total if total else 0,'scored':total}
    report={'scorer':'strict local SimpleTool scorer; exact call count and argument keys; not official BFCL execution/AST','datasets':all_results,'categories':agg,'data_manifest_sha256':sha256(a.data/'manifest.json')}; (a.output/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    lines=['# Strict SimpleTool accuracy audit','','This is the primary conservative local score. It requires exact call count and rejects extra predicted argument names. It does not execute official BFCL programs.','','| Category | Scored | Function | Overall |','| --- | ---: | ---: | ---: |']
    for g,v in agg.items(): lines.append(f'| {g} | {v["scored"]} | {v["function_accuracy"]:.2%} | {v["accuracy"]:.2%} |')
    lines += ['', '## Per dataset','', '| Dataset | Scored | Correct | Function | Overall |', '| --- | ---: | ---: | ---: | ---: |']
    for n,s in all_results.items(): lines.append(f'| {n} | {s["scored"]} | {s["correct"]} | {s["function_accuracy"]:.2%} | {s["accuracy"]:.2%} |')
    (a.output/'summary.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
if __name__=='__main__': main()
