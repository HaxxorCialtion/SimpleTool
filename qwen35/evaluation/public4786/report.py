"""Re-score immutable outputs, verify coverage, and publish quality-only report."""
import argparse,collections,csv,json,tarfile
from pathlib import Path
from benchlib import ROOT,sha,score_row,aggregate
CLOUD='/inspire/hdd/global_user/shixiaoxin-253107030017/sxx/multi-head/rethinking/pretrain_level/final/glm51/sota/hermes/simpletool/final_data/vlm/mobile_actions_scale_p005_codex_20261109/benchmark'
GROUPS={
 'BFCL Non-Live':['BFCL_v3_simple','BFCL_v3_multiple'],
 'BFCL Live':['BFCL_v3_live_simple','BFCL_v3_live_multiple'],
 'BFCL Exec (local)':['BFCL_v3_exec_simple_dataset','BFCL_v3_exec_multiple_dataset'],
 'Seal-Tools':[f'sealtools_test_{i}tools' for i in range(1,7)]+['sealtools_test_in_domain','sealtools_test_out_domain'],
 'OpenFunctions':['openfunctions_v1_test'],
 'ToolAlpaca':['toolalpaca_test_real_converted','toolalpaca_test_simulated_converted'],
 'Mobile Actions':['mobile_actions']}
LOCAL='/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/artifacts/mobile_actions_scale_p005_20261109/benchmark'
METRICS=('function_match','required_arguments_match','typed_required_arguments_match','full_local_match')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');a=ap.parse_args()
 cases=[json.loads(s) for s in (ROOT/'cases.jsonl').read_text().splitlines()]
 prep=json.loads((ROOT/'preparation.json').read_text());assert sha(ROOT/'cases.jsonl')==prep['cases_sha256']
 runs={};examples={};audit={};paired_cache={}
 overlap=json.loads((ROOT/'training-overlap.json').read_text()) if (ROOT/'training-overlap.json').exists() else None
 overlap_uids=set(overlap['normalized_train_overlap_uids']) if overlap else set()
 for size in ('0p8b','2b','4b'):
  for precision in ('fp16','int4'):
   tag=f'{precision}-{size}';out=ROOT/'runs'/tag
   if not (out/'run.json').exists():
    if a.partial:continue
    raise ValueError('missing run '+tag)
   receipt=json.loads((out/'run.json').read_text())
   if receipt['status']!='completed':
    if a.partial:continue
    raise ValueError('incomplete run '+tag)
   assert receipt['config']['cases_sha256']==sha(ROOT/'cases.jsonl'),tag
   assert sha(out/'outputs.jsonl')==receipt['output_sha256'],tag
   assert sha(out/'scores.json')==receipt['scores_sha256'],tag
   raw=[json.loads(s) for s in (out/'outputs.jsonl').read_text().splitlines()]
   assert [r['uid'] for r in raw]==[c['uid'] for c in cases],tag
   for r,c in zip(raw,cases):assert [h['name'] for h in r['heads']]==c['heads'],r['uid']
   scored=[score_row(r,c) for r,c in zip(raw,cases)]
   saved=json.loads((out/'scores.json').read_text())
   assert saved['scored_rows']==scored and saved['all']==aggregate(scored),tag
   ds={n:aggregate([r for r in scored if r['dataset']==n]) for n in prep['datasets']}
   assert saved['datasets']==ds,tag
   groups={g:aggregate([r for r in scored if r['dataset'] in names]) for g,names in GROUPS.items()}
   # Match published reproduction's equal-weight five-family construction,
   # but adapted rows/prompts/primary metric mean scores aren't paper scores.
   five={}
   for metric in METRICS:
    ratio=lambda n:ds[n][metric]['fraction']
    bfcl=sum(ratio(n) for n in ds if n.startswith('BFCL'))/6
    sn=[f'sealtools_test_{i}tools' for i in range(1,7)]
    seal_micro=sum(ds[n][metric]['count'] for n in sn)/sum(ds[n]['n'] for n in sn)
    seal=(seal_micro+ratio('sealtools_test_in_domain')+ratio('sealtools_test_out_domain'))/3
    families=dict(BFCL=bfcl,SealTools=seal,MobileActions=ratio('mobile_actions'),OpenFunctions=ratio('openfunctions_v1_test'),ToolAlpaca=(ratio('toolalpaca_test_real_converted')+ratio('toolalpaca_test_simulated_converted'))/2)
    five[metric]=dict(families=families,macro_average=sum(families.values())/5)
   run={k:saved[k] for k in ('all','stops','replacement_character_heads','total_heads','length_limit_by_head','early_eos_by_head')}
   run.update(datasets=ds,groups=groups,five_family_macro=five,receipt=receipt,
    normalized_user_nonoverlap=aggregate([r for r in scored if r['uid'] not in overlap_uids]),
    normalized_user_nonoverlap_bfcl_simple=aggregate([r for r in scored if r['dataset']=='BFCL_v3_simple' and r['uid'] not in overlap_uids]))
   runs[tag]=run
   paired_cache[tag]=[{**{k:s[k] for k in METRICS},'uid':s['uid'],'content_closed':next(h['stop']=='closing_marker' for h in r['heads'] if h['name']=='content')} for s,r in zip(scored,raw)]
   by_uid={r['uid']:r for r in raw};by_case={c['uid']:c for c in cases}
   samples={}
   for name in prep['datasets']:
    wrong=[s for s in scored if s['dataset']==name and not s['required_arguments_match']]
    samples[name]=[dict(scored=s,prompt=by_case[s['uid']]['prompt'],heads=by_uid[s['uid']]['heads']) for s in wrong[:4]]
   examples[tag]=samples
   audit[tag]=dict(rows=len(raw),heads=sum(len(r['heads']) for r in raw),output_sha256=sha(out/'outputs.jsonl'),rescore_passed=True)
 assert a.partial or len(runs)==6
 pairs={}
 for size in ('0p8b','2b','4b'):
  f=paired_cache.get('fp16-'+size);q=paired_cache.get('int4-'+size)
  if f is None or q is None:continue
  pairs[size]={}
  for metric in METRICS+('content_closed',):
   pairs[size][metric]=dict(fp16_correct=sum(r[metric] for r in f),int4_correct=sum(r[metric] for r in q),
    int4_fixed_ids=[a['uid'] for a,b in zip(f,q) if not a[metric] and b[metric]],
    int4_regressed_ids=[a['uid'] for a,b in zip(f,q) if a[metric] and not b[metric]])
 summary=dict(quantization_pairs=pairs,status='partial' if len(runs)<6 else 'completed',cases_sha256=sha(ROOT/'cases.jsonl'),preparation=prep,runs=runs)
 (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 (ROOT/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
 (ROOT/'verification.json').write_text(json.dumps(dict(status=summary['status'],runs=audit,all_case_ids_verified=True,raw_outputs_rescored=True),indent=2)+'\n')
 rows=[]
 for tag,r in runs.items():
  for dataset,v in r['datasets'].items():
   rows.append(dict(run=tag,dataset=dataset,n=v['n'],**{k:v[k]['count'] for k in METRICS}))
 if rows:
  with (ROOT/'quality.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 def cell(tag,group,metric='required_arguments_match'):
  if tag not in runs:return '待完成'
  g=runs[tag]['groups'][group];m=g[metric]
  return f'{m["count"]}/{g["n"]} ({m["fraction"]:.2%})'
 def fmt(tag,metric):
  if tag not in runs:return '待完成'
  r=runs[tag]['all'];return f'{r[metric]["count"]}/{r["n"]} ({r[metric]["fraction"]:.2%})'
 lines=['# SimpleTool Qwen3.5 八头 5% 批次：完整基准子集评测','',f'状态：{summary["status"]}；完成 {len(runs)}/6 组。日期：2026-11-09。RTX 4090 48 GiB，vLLM 0.31.0。','',
 '## 数据来源与过滤','',
 '从 QZ 上作者的 SimpleTool 复现发布目录复制并下载数据快照，未刷新为上游不同版本。源仓库 `HaxxorCialtion/SimpleTool`，commit `'+json.loads((ROOT/'source-receipt.json').read_text())['commit']+'`。原始数据、possible_answer、manifest、来源声明、源解析器和测试均保留于 upstream/。','',
 f'原始 {prep["total_input"]} 行，保留 {prep["total_included"]} 行，过滤 {prep["total_excluded"]} 行；每个模型实际生成 {prep["total_heads"]} 个独立分支。所有样本在推理前冻结；无预测失败后删行。','',
 '| 数据集 | 原始 | 保留 | 排除原因 |','|---|---:|---:|---|']
 for n,v in prep['datasets'].items():
  reason='；'.join(f'{k}: {value}' for k,value in v.items() if k not in ('input','included','order_adapted')) or '—'
  lines.append(f'| {n} | {v["input"]} | {v["included"]} | {reason} |')
 lines += ['',
 'Mobile Actions 原有 961 个任务转换成 1283 行。本轮遵循用户要求，整个排除 321 个多调用任务（643 行），仅保留 640 个独立单调用任务；没有用参考历史把多调用任务变成额外测试行。BFCL 的 multiple 类别表示多个候选工具，并非必须调用多个工具，因此保留其单调用样本。','',
 '超过六个 property 的任一候选工具会导致整行排除，沿用项目容量约束。缺失/无效 GT、GT 函数未出现在候选工具、GT 缺少 schema 必填值的行也预先排除，具体 ID 与原因见 exclusions.json。没有手动修正错配 ID。','',
 'Seal-Tools 的多个候选工具数版本重复使用一批 query；它们是受控条件变体，不是独立的额外问题。总行数准确率不能当成独立样本泛化估计。','',
 '## Prompt、解码和评分','',
 '- 工具候选顺序保留；每个对象的必填 property 按原相对顺序前移，可选 property 随后；required 列表改为相同必填顺序，原始数据未改。嵌套对象也做相同顺序对齐，变更见 order-adaptations.json。','- 保留源 schema 类型名（包括 dict、float、tuple、any 及原有非标准拼写），不替换描述、默认值、枚举或目标值。','- 使用上一轮 Qwen3.5 评测的简洁 system 风格，原始 user 中 environment/history 原样保留；不再套 chat template，不添加思维链，不向 prompt 注入 GT。与发布项目旧 v1 prompt 存在差异，不能直接比较其论文/历史分数。','- 分支为 content、function、arg1..argN；N 是所有候选工具的最大参数数，不查看参考函数。最多八分支；不按 GT 决定解码 head。各分支遇到自己的结束 marker 终止；null 是内容，不是 stop。','- 主指标：精确函数名 + schema.required 参数，值按冻结项目本地 acceptable-value 规则比较。可选参数与未用槽位不影响主指标。该本地规则容忍字符串大小写/空白及部分数值转换，并非逐字精确、官方 AST 或真实执行判定。','- 同时报告 typed_required_arguments_match，对所选函数必填参数做顶层类型检查（如 bool 不作 integer）；此项不是递归 JSON Schema 验证。full_local_match 调用源 strict_call 检查所有 GT 参数并拒绝额外声明参数。','- 源本地 matcher 对嵌套对象可接受值、数组标注等的处理能力有限；原始标注未重写。错误不能一概解释成真实工具执行失败。','- 函数名只允许单个候选工具的精确名字，不把多函数输出拆开挑一个正确结果。','',
 '## 必填参数正确率（包含函数名）','',
 '| 类别 | 0.8B FP16 | 0.8B INT4 | 2B FP16 | 2B INT4 | 4B FP16 | 4B INT4 |','|---|---:|---:|---:|---:|---:|---:|']
 tags=[f'{precision}-{size}' for size in ('0p8b','2b','4b') for precision in ('fp16','int4')]
 for group in GROUPS:lines.append('| '+group+' | '+' | '.join(cell(t,group) for t in tags)+' |')
 lines+=['','各类别表格按本轮保留行数 micro 汇总；不等同于论文五类宏平均。','',
 '## 函数选择准确率','',
 '| 类别 | 0.8B FP16 | 0.8B INT4 | 2B FP16 | 2B INT4 | 4B FP16 | 4B INT4 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for group in GROUPS:lines.append('| '+group+' | '+' | '.join(cell(t,group,'function_match') for t in tags)+' |')
 lines += ['','## 全行汇总与完整输出诊断','','| 运行 | 函数名 | 必填参数 | 带顶层类型检查的必填参数 | 完整 GT 参数 | 正常闭合分支 | 长度上限 | 提前 EOS | U+FFFD 分支 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for tag,r in runs.items():
  lines.append('| '+tag+' | '+' | '.join(fmt(tag,m) for m in METRICS)+f' | {r["stops"].get("closing_marker",0)}/{r["total_heads"]} | {r["stops"].get("max_tokens",0)} | {r["stops"].get("eos_before_closing_marker",0)} | {r["replacement_character_heads"]} |')
 lines += ['','五类宏平均另保存在 summary.json 的 five_family_macro：BFCL 六份数据平均；Seal-Tools 为 1–6 工具合并 micro、in-domain、out-domain 三者平均；ToolAlpaca 两份平均；再与 OpenFunctions、Mobile Actions 做五类等权平均。零分组保留。由于过滤、prompt、评分与发布版不同，这仍不是原论文分数。','',
 '## 精度变化的通道诊断','','工具参数匹配接近不等于所有通道无损。4B INT4 的 content 提前 EOS 为 1690 个，而 FP16 为 57 个；工具分支两者全部闭合。逐模型/指标 FP16→INT4 的修正与退化 ID 另存 summary.json 的 quantization_pairs，包括 content_closed。其他尺寸须看本轮实际结果，不能推广该 4B 观察。','','## 训练文本重合与敏感性结果','','已检查 INDEX.json 指定的 75,299 条原始选中数据（按训练 split hash 规则：train 71,524，val 3,775），按最终 user 文本比较。两条 BFCL Simple 文本与 train 完全相同，也在大小写/空白归一化比较中重合：simple_9、simple_310。主分母保留；去掉这两条的结果见 summary.json 中 normalized_user_nonoverlap 与 normalized_user_nonoverlap_bfcl_simple。该检查包含隔离行，4B 只训练至 2353 步，不证明语义、改写或预训练数据无重合。无训练权限或重新训练操作。','','## 运行与量化','',
 '本轮用新 0.8B/2B 的 2474 步和新 4B 的 2353 步保存点。FP16 原始导出不变；INT4 复用上一轮 group64 affine RTN 权重，AWQ 布局 / Marlin 执行，没有 activation calibration。主要语言线性层 INT4，其余部分浮点；A16、attention KV16，GDN state 引擎默认。不是 iPhone MLX/KV8 验证。','',
 '跨请求和跨 head 连续批处理；每次提交 64 个样本，max_num_seqs=192，eager，prefix cache 开启，max_model_len=16384，max_num_batched_tokens=8192，显存比例 0.70。温度 0，seed=1107，每分支上限 128 tokens。输入最长 4039 tokens，未截断 prompt。所有 marker 已检查为实际 tokenizer 的独立特殊 token。','',
 '达到长度上限、提前 EOS 和内容错误都保留在固定分母内；指标不强制正常闭合，闭合单独报告。本轮不计单请求速度，不执行任何模型生成工具；没有向 Mac 下载模型权重。','',
 '源测试 12 项与新增协议测试 5 项，以及 token/数据 preflight 均通过。初次 Mac 和云端测试从错误目录启动，遇到 import path 错误；修正工作目录/PYTHONPATH 后同一套源测试全部通过，未修改源测试。原始预测重新评分并逐条核对保存分数；各运行覆盖全部固定 UID，哈希与回执一致。','',
 '## OpenFunctions 错误案例说明','','4B INT4 的 OpenFunctions 低分包含多种情况：Chicago, IL 与 Chicago, Illinois、Los Angeles, CA 与 Los Angeles, California、Check the oven. 与 Check the oven 都被源 matcher 判为不同。另有输入只写 tonight 或 specific date range，但 GT 指定具体日期；预订餐厅例子输入仅 Italian restaurant，但 GT 要求 Italiano’s。还有真正缺失参数或日期值不同的输出。见 openfunctions-error-probe.json；没有修改参考值或人工改分。这说明本地匹配分数不能直接作为真实工具执行成功率。','','## GPU 利用率的解释','','本客户端每次同步 generate 64 个问题的分支，整批完成后解析/写盘再提交下一批，并非全量异步队列持续补充。短分支陆续结束时，批尾只有少数长分支，下一批尚未入队。模型切换/加载、tokenization、写盘及 eager 执行的 kernel 提交间隙也可能降低占空比；尚未逐低谷做 profiler 归因。nvidia-smi utilization.gpu 是采样窗口内 GPU 忙碌时间比例，不能直接当作峰值 FLOPS 达成率。日志确认该 FP16 dtype 使用 Triton GDN decode，未启用要求 BF16 的 fused CUDA GDN 路径。此次不调整这些参数，以保持质量对照一致；本轮耗时/利用率不是持续饱和吞吐上限。','','## 路径与复现','', '云端工作目录：','', '```text',CLOUD,'```','','本地小型数据/脚本/结果目录：','','```text',LOCAL,'```','',
 '文件：cases.jsonl（完整实际 prompt/工具/GT）、preparation.json、exclusions.json、order-adaptations.json、source-receipt.json、preflight.json、quality.csv、summary.json、examples.json、verification.json、runs/<tag>/{outputs.jsonl,scores.json,run.json}。','',
 '```bash','python prepare.py','python preflight.py','bash run_suite.sh','python report.py','```','',
 'examples.json 保留每数据集若干真实错误的完整 prompt、全部分支和参考 acceptable values。未做人工语义改分。', '']
 (ROOT/'REPORT.md').write_text('\n'.join(lines))
 if not a.partial:
  archive=ROOT/'benchmark-evidence-no-weights.tar.gz'
  with tarfile.open(archive,'w:gz',dereference=False) as tar:
   for p in sorted(ROOT.rglob('*')):
    rel=p.relative_to(ROOT)
    if not p.is_file() or p.is_symlink() or 'cache' in rel.parts or '__pycache__' in rel.parts:continue
    if p.suffix not in ('.json','.jsonl','.py','.sh','.md','.log','.csv','.txt'):continue
    tar.add(p,arcname=str(rel),recursive=False)
  (ROOT/'archive-sha256.txt').write_text(sha(archive)+'  '+archive.name+'\n')
 print(json.dumps(dict(status=summary['status'],completed=list(runs),rows=len(cases)),ensure_ascii=False))
if __name__=='__main__':main()
