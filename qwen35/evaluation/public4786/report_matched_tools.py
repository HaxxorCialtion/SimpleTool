"""Verify all twelve tools-only runs and report paired quality comparisons."""
import collections,csv,json,tarfile
from pathlib import Path
from benchlib import ROOT,sha,score_row,aggregate
from report import GROUPS,METRICS

def main():
 folder=ROOT/'tools-only';dest=folder/'matched-report';dest.mkdir(exist_ok=True)
 cases=[json.loads(s) for s in (ROOT/'cases.jsonl').read_text().splitlines()]
 case_hash=sha(ROOT/'cases.jsonl');runs={};scored_by={};checks={}
 overlap=set(json.loads((ROOT/'training-overlap.json').read_text())['normalized_train_overlap_uids'])
 plan=json.loads((folder/'parallel-plan.json').read_text())
 jobs={j['tag']:j for lane in plan['lanes'].values() for j in lane['jobs']}
 for size in ('0p8b','2b','4b'):
  for generation in ('old','new'):
   for precision in ('fp16','int4'):
    tag=f'{generation}-{precision}-{size}';path=folder/'runs'/tag
    receipt=json.loads((path/'run.json').read_text());config=receipt['config']
    assert receipt['status']=='completed',tag
    assert config['cases_sha256']==case_hash and config['adapter_sha256']==sha(ROOT/'benchlib.py'),tag
    script='infer_tools_only.py' if tag=='old-fp16-4b' else 'infer_tools_parallel.py'
    assert config['script_sha256']==sha(ROOT/script),tag
    if tag in jobs:
     assert config['model']==jobs[tag]['model'] and config['config_sha256']==jobs[tag]['config_sha256'],tag
    assert receipt['output_sha256']==sha(path/'outputs.jsonl') and receipt['scores_sha256']==sha(path/'scores.json'),tag
    raw=[json.loads(s) for s in (path/'outputs.jsonl').read_text().splitlines()]
    assert [r['uid'] for r in raw]==[c['uid'] for c in cases],tag
    for r,c in zip(raw,cases):assert [h['name'] for h in r['heads']]==[h for h in c['heads'] if h!='content'],(tag,r['uid'])
    scored=[score_row(r,c) for r,c in zip(raw,cases)];saved=json.loads((path/'scores.json').read_text())
    assert saved['scored_rows']==scored and saved['all']==aggregate(scored),tag
    datasets={n:aggregate([r for r in scored if r['dataset']==n]) for n in dict.fromkeys(c['dataset'] for c in cases)}
    assert datasets==saved['datasets'],tag
    heads=[h for r in raw for h in r['heads']];stops=dict(collections.Counter(h['stop'] for h in heads))
    assert stops==saved['stops'] and len(heads)==saved['total_heads']==21465,tag
    groups={g:aggregate([r for r in scored if r['dataset'] in names]) for g,names in GROUPS.items()}
    five={}
    for metric in METRICS:
     ratio=lambda n:datasets[n][metric]['fraction']
     bfcl=sum(ratio(n) for n in datasets if n.startswith('BFCL'))/6
     sn=[f'sealtools_test_{i}tools' for i in range(1,7)]
     seal_micro=sum(datasets[n][metric]['count'] for n in sn)/sum(datasets[n]['n'] for n in sn)
     families=dict(BFCL=bfcl,SealTools=(seal_micro+ratio('sealtools_test_in_domain')+ratio('sealtools_test_out_domain'))/3,MobileActions=ratio('mobile_actions'),OpenFunctions=ratio('openfunctions_v1_test'),ToolAlpaca=(ratio('toolalpaca_test_real_converted')+ratio('toolalpaca_test_simulated_converted'))/2)
     five[metric]=dict(families=families,macro_average=sum(families.values())/5)
    runs[tag]=dict(all=saved['all'],datasets=datasets,groups=groups,stops=stops,receipt=receipt,five_family_macro=five,normalized_user_nonoverlap=aggregate([r for r in scored if r['uid'] not in overlap]))
    scored_by[tag]=scored
    checks[tag]=dict(rows=len(raw),heads=len(heads),rescore_passed=True,no_content=True,output_sha256=receipt['output_sha256'],scores_sha256=receipt['scores_sha256'])
 def paired(old,new):
  a,b=scored_by[old],scored_by[new]
  return {m:dict(before=sum(r[m] for r in a),after=sum(r[m] for r in b),fixed_ids=[x['uid'] for x,y in zip(a,b) if not x[m] and y[m]],regressed_ids=[x['uid'] for x,y in zip(a,b) if x[m] and not y[m]]) for m in METRICS}
 training={f'{p}-{s}':paired(f'old-{p}-{s}',f'new-{p}-{s}') for s in ('0p8b','2b','4b') for p in ('fp16','int4')}
 quant={f'{g}-{s}':paired(f'{g}-fp16-{s}',f'{g}-int4-{s}') for s in ('0p8b','2b','4b') for g in ('old','new')}
 result=dict(status='completed',case_count=len(cases),cases_sha256=case_hash,content_generated=False,runs=runs,training_pairs=training,quantization_pairs=quant)
 (dest/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 (dest/'verification.json').write_text(json.dumps(dict(status='passed',run_count=len(runs),total_rows=sum(c['rows'] for c in checks.values()),total_heads=sum(c['heads'] for c in checks.values()),runs=checks),indent=2)+'\n')
 rows=[]
 for tag,r in runs.items():
  for dataset,v in r['datasets'].items():rows.append(dict(run=tag,dataset=dataset,n=v['n'],**{m:v[m]['count'] for m in METRICS}))
 with (dest/'quality.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 def fmt(tag,metric='required_arguments_match',group=None):
  v=runs[tag]['all'] if group is None else runs[tag]['groups'][group]
  return f'{v[metric]["count"]}/{v["n"]} ({v[metric]["fraction"]:.2%})'
 lines=['# SimpleTool Qwen3.5：0.5% 与 5% 批次同口径工具头评测','',
  '状态：12/12 完成，全部原始输出重新评分校验通过。日期：2026-11-09。','',
  '## 口径','',
  '固定 4,786 行、18 份来源数据；每组 21,465 条分支。只生成 function 与候选工具最大参数数需要的 arg1..argN，不生成 content。排除多调用请求；候选工具可以有多个。必填参数前置，同时同步 properties 与 required 顺序，arg 槽位按所选函数属性顺序解释。候选顺序保持不变，分支数不读取 GT。','',
  '主指标：函数名精确匹配且全部必填参数符合项目冻结的 acceptable-value 匹配器；可选参数及未用槽位不影响主指标。另报函数选择、顶层类型约束必填、全部 GT 参数匹配。不是官方 BFCL AST、执行或语义分数；没有执行任何预测工具。所有失败保留在分母内。','',
  'vLLM 0.31.0，Torch 2.14.1+cu130，三张 RTX 4090 48 GiB，各模型独占一张卡、TP=1。温度0、seed1107，每批64行、每头上限128 tokens，FP16激活/attention KV、GDN状态用引擎默认，eager、prefix cache 开启，max_num_seqs=192。INT4 为 group64 affine RTN 转 AWQ 布局并由 Marlin 执行，未做校准 AWQ；embedding、输出层、视觉层和部分小算子仍浮点。此处不是 iPhone KV8 配置，也不评测速度上限。','',
  '## 全行加权结果','',
  '| 模型 | 精度 | 旧 0.5% 必填正确 | 新 5% 必填正确 | 变化 pp | 旧函数正确 | 新函数正确 | 修正 / 退步 |',
  '|---|---|---:|---:|---:|---:|---:|---:|']
 for s in ('0p8b','2b','4b'):
  for p in ('fp16','int4'):
   old=f'old-{p}-{s}';new=f'new-{p}-{s}';pair=training[f'{p}-{s}']['required_arguments_match']
   delta=(pair['after']-pair['before'])/len(cases)*100
   lines.append(f'| {s} | {p} | {fmt(old)} | {fmt(new)} | {delta:+.2f} | {fmt(old,"function_match")} | {fmt(new,"function_match")} | {len(pair["fixed_ids"])} / {len(pair["regressed_ids"])} |')
 lines+=['','## 分来源类别：必填正确','','| 类别 | 模型 | 精度 | 旧 0.5% | 新 5% |','|---|---|---|---:|---:|']
 for g in GROUPS:
  for s in ('0p8b','2b','4b'):
   for p in ('fp16','int4'):lines.append(f'| {g} | {s} | {p} | {fmt(f"old-{p}-{s}",group=g)} | {fmt(f"new-{p}-{s}",group=g)} |')
 lines+=['','## 其他指标与停止情况','','| 运行 | 带类型必填 | 完整 GT 参数 | 正常闭合 | 提前 EOS | 长度上限 |','|---|---:|---:|---:|---:|---:|']
 for tag,r in runs.items():lines.append(f'| {tag} | {fmt(tag,"typed_required_arguments_match")} | {fmt(tag,"full_local_match")} | {r["stops"].get("closing_marker",0)} | {r["stops"].get("eos_before_closing_marker",0)} | {r["stops"].get("max_tokens",0)} |')
 lines+=['','## 审计与限制','',
  '全行汇总是行数加权；Seal-Tools 同一问题有不同候选数量变体，不能视为独立问题，也不等同论文宏平均。summary.json 另外保留按复现项目五大来源家族构造的宏平均，仍受当前过滤与评分口径影响。','',
  '新 5% 的 75,299 条选中原始训练/验证数据仅做用户文本完全一致和空白大小写归一化检查，找到两条训练重合：BFCL_v3_simple/simple_9、simple_310。剔除敏感性结果在 summary.json；旧 0.5% 未做同等训练审计，亦未做语义或预训练去重。','',
  '双卡容器最初误加载 CUDA12.9 兼容库，启动前失败；failed-driver-attempt/ 保留日志。重试通过 LD_PRELOAD 显式加载宿主595.71.05驱动库；推理代码、模型与冻结计划未变。三个主机系统时间可能不同，不用跨节点时间戳计算时延。运行回执记录主机及 GPU UUID（此前旧4B FP16回执除外）。','',
  '来源快照：HaxxorCialtion/SimpleTool，commit ea920f4983579ded7f3349d673b11f970884366c。cases.jsonl SHA256: '+case_hash,'',
  '旧模型：qwen35_8head_lora_20261107/fp16/{0p8b,2b,4b}；新模型：qwen35_8head_scale_20261108/fp16/p005/{0p8b,2b,4b}。各个实际模型绝对路径及config摘要见回执。原先含content的六组结果和旧4B单独报告保留，不被本报告覆盖。','',
  'runs/<tag>/ 保存逐头原始输出、逐行评分及运行回执；matched-report/summary.json 保存成对修正/退步 UID、按18份数据分数、量化对照与重合敏感性结果。','']
 (dest/'REPORT.md').write_text('\n'.join(lines))
 print(json.dumps({k:v['all'] for k,v in runs.items()},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
