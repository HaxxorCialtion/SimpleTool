"""Audits for optimistic scoring choices in the recovered evaluator."""
import argparse, json
from pathlib import Path
from .conversion import convert_raw_heads_to_fc
from .data import load_pair, read_jsonl, selected_names
from . import rules


def strict_decision(pred, gt_list):
    out={'func_correct':False,'overall_correct':False}
    if not pred:
        return out
    # Require exactly one parsed call for these single-call datasets.
    if len(pred) != 1:
        return out
    call=pred[0]; name=call.get('name',''); args=call.get('arguments',{})
    for gt_item in gt_list:
        if not gt_item: continue
        gt_name=list(gt_item)[0]; gt_params=gt_item[gt_name]
        if name != gt_name: continue
        out['func_correct']=True
        # Legacy scorer only checks GT keys. Strict scorer also rejects hallucinated keys.
        if set(args) - set(gt_params):
            continue
        if all(rules.check_param_match(args.get(k), v if isinstance(v,list) else [v]) for k,v in gt_params.items()):
            out['overall_correct']=True; break
    return out


def main():
    p=argparse.ArgumentParser(); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--data',type=Path,default=Path('data')); p.add_argument('--output',type=Path,default=Path('reports/scoring_audit.json')); a=p.parse_args()
    report={'warning':'This compares scoring policies on the same predictions; it is not a new model run.','datasets':{}}
    for name in selected_names():
        ds=Path(name).stem; rows,gt,_=load_pair(a.data,name); path=a.predictions/f'{ds}.jsonl'
        if not path.exists(): path=a.predictions/f'simpletool_RT-Qwen3-4B_{ds}.jsonl'
        if not path.exists(): continue
        preds,_=read_jsonl(path); inp={r['id']:r for r in rows}; x={'predictions':len(preds),'scored':0,'legacy_func':0,'legacy_overall':0,'strict_func':0,'strict_overall':0,'extra_arg_predictions':0,'multi_call_predictions':0,'filtered':0,'missing_gt':0}
        for item in preds:
            iid=item['id']
            if iid not in gt: x['missing_gt']+=1; continue
            if rules.has_tool_exceeding_max_params(inp[iid]): x['filtered']+=1; continue
            pred=item.get('result') or convert_raw_heads_to_fc(item.get('raw_heads',{}),inp[iid].get('function',[])); old=rules.evaluate_single_prediction(pred,gt[iid]); new=strict_decision(pred,gt[iid]); x['scored']+=1
            x['legacy_func']+=old['func_correct']; x['legacy_overall']+=old['overall_correct']; x['strict_func']+=new['func_correct']; x['strict_overall']+=new['overall_correct']
            if pred and len(pred)>1: x['multi_call_predictions']+=1
            if pred and set(pred[0].get('arguments',{})) - set(list(gt[iid][0].values())[0] if gt[iid] else {}): x['extra_arg_predictions']+=1
        for k in ['legacy_func','legacy_overall','strict_func','strict_overall']:
            x[k+'_acc']=x[k]/x['scored'] if x['scored'] else 0
        x['strict_minus_legacy_overall']=x['strict_overall_acc']-x['legacy_overall_acc']; report['datasets'][ds]=x
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
