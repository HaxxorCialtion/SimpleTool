"""Score outputs; no model execution. See README for prediction formats."""
import argparse
import collections
import json
from pathlib import Path
from benchmark_lib import score_case, strict_json, typed_equal

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cases',type=Path,default=Path(__file__).resolve().parents[1]/'cases.jsonl')
    ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    rows=[strict_json(x) for x in a.cases.read_text().splitlines() if x.strip()]
    by_id={r['id']:r for r in rows};preds={}
    for line in a.predictions.read_text().splitlines():
        if not line.strip():continue
        p=strict_json(line);rid=p['id']
        if rid not in by_id or rid in preds:raise ValueError('unknown or duplicate ID '+str(rid))
        preds[rid]=p
    scored=[score_case(preds.get(r['id'],{}),r) for r in rows]
    smap={r['id']:s for r,s in zip(rows,scored)};pairs=collections.defaultdict(list)
    for r in rows:pairs[r['pair_id']].append(r)
    pair_scores=[]
    for pid,p in pairs.items():
        x,y=sorted(p,key=lambda r:r['variant']);sx,sy=smap[x['id']],smap[y['id']]
        both=sx['exact_correct'] and sy['exact_correct'];changed=not typed_equal(sx['predicted'],sy['predicted'])
        valid=sx['schema_valid'] and sy['schema_valid']
        pair_scores.append(dict(pair_id=pid,effect=x['contrast']['effect'],both_correct=both,
            response_changed=changed,both_schema_valid=valid,
            correct_change=both if x['contrast']['effect']=='change' else None,
            correct_invariance=both if x['contrast']['effect']=='invariant' else None))
    def aggregate(rs):
        return dict(n=len(rs),**{k:sum(x[k] for x in rs)/len(rs) if rs else None for k in
            ('name_correct','schema_valid','required_correct','exact_correct')})
    summary=dict(total=aggregate(scored),missing_prediction_ids=sorted(set(by_id)-set(preds)),
        family={f:aggregate([smap[r['id']] for r in rows if r['family']==f]) for f in sorted({r['family'] for r in rows})},
        tags={t:aggregate([smap[r['id']] for r in rows if t in r['tags']]) for t in sorted({t for r in rows for t in r['tags']})},
        pair_joint_accuracy=sum(x['both_correct'] for x in pair_scores)/len(pair_scores),
        pair_groups={effect:dict(n=len(ps),joint_accuracy=sum(x['both_correct'] for x in ps)/len(ps))
            for effect in ('change','invariant') if (ps:=[x for x in pair_scores if x['effect']==effect])},
        cases=scored,pairs=pair_scores)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('cases','pairs','tags')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
