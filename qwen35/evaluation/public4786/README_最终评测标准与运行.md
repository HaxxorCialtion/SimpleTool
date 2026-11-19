# SimpleTool 公开 benchmark 最终评测标准与脚本

交接日期：2026-11-14。这里交付的是11月9日最终完成12组模型对照的冻结数据、原始推理/评分脚本与协议；本次没有启动GPU、重跑模型或修改原始评分。新增check_frozen.py、rescore_run.py仅方便迁移校验和单组重新评分。

## 1. 采用哪一套标准

**本文件描述的最终4786行、tools-only标准用于这次交接。** `upstream/AI_AGENT_EVAL_PROTOCOL.md`与`upstream/AGENTS.md`是较早公开Qwen3复现的历史协议，完整保留作来源追溯。有两项与后来用户要求不同：

- 历史Mobile Actions允许参考history拆行；**本轮排除整个多调用任务，只保留640个单调用任务**，不借参考前序调用增加测试行。
- 历史要求保留源properties原序；**本轮先按源相对顺序把必填参数前移、可选参数后置，再将required列表同步为该必填顺序**。arg1…argN按归一化后的properties插入顺序映射。不是按字母排序，也不是按原required列表重排。

两者是不同评测版本，不得将历史upstream strict CLI跑出的结果与本轮主指标混称同一分数。代码以根目录benchlib.py、prepare.py、infer_tools_parallel.py为准。

## 2. 固定数据与分母

来源：`HaxxorCialtion/SimpleTool`，commit `ea920f4983579ded7f3349d673b11f970884366c` 的作者本地数据快照。不是本次刷新上游最新版。源数据和possible_answer在upstream/data，来源声明和许可文本保留，不能假定混合后的全部数据统一使用一种许可。

原始5647行，预先排除861行，保留4786行，来自18份数据文件：

|家族/类别|最终行数|
|---|---:|
|BFCL Non-Live：simple + multiple|599|
|BFCL Live：live_simple + live_multiple|1128|
|BFCL Exec：exec_simple + exec_multiple|149|
|Seal-Tools：1–6候选工具、in/out-domain|2024|
|OpenFunctions|109|
|ToolAlpaca：real + simulated|137|
|Mobile Actions：单调用任务|640|
|合计|4786|

Mobile Actions原961个任务转换为1283行；排除321个多调用任务（643行），保留640行。BFCL的multiple指多个候选工具，不等于一次必须调用多个工具，符合单调用条件的样本仍保留。

任意候选工具超过6个顶层property，整行排除，即使GT选中的工具更短。另排除无效/缺失GT、目标函数不在候选中、参考答案缺少schema必填值、非恰好一个参考调用。所有排除在推理前冻结，见exclusions.json。预测错、漏参数、EOS、长度截断不能事后删掉。

冻结cases.jsonl SHA-256：

```text
80694c46b01c6e1241010edaa9b2a94db20cbe642877a11655c3a301ebd57b6a
```

## 3. Prompt与位置参数协议

工具候选顺序、参数名、描述、类型、默认值、enum、目标值不改。包括dict/float/tuple/any等非标准类型拼写，也原样保留。只改变必填/可选property与required顺序，嵌套schema同样处理。改动清单在order-adaptations.json。

使用benchlib.build_prompt构造原始ChatML，保留原system及user中的environment/history文本。冻结prompt以 `<|im_start|>assistant\n` 结束；各分支分别追加 `<function>`、`<arg1>`…，不再套一次chat template，**这版没有额外空think块或CoT**。不要将后来的H200 demo prompt直接替换进来后仍称严格复现。

仅推理function与arg1…argN；N等于所有候选工具最大property数量（最多6），不运行content，也不读取GT决定head数量。一次输出一个调用。所有head共享同一问题输入，但独立解码；每个head遇到自己的 `</head>` 停止，`<|null|>`是内容不是stop。

最终选择函数确定后，只解释该函数property范围内的arg；多出的head无论内容都忽略。例如函数只有两个property，arg3…arg6不影响任何调用评分。候选可选参数仍占位置，只是不一定影响主指标。

cases.jsonl含ground_truth只为离线评分，推理脚本实际送给引擎的是case.prompt+head marker，不传整个case对象。必须保留这一隔离。

## 4. 指标：主指标只要求必要参数

|字段|判据|
|---|---|
|function_match|函数名去首尾空白后精确等于参考函数；不得拆多个函数挑正确的|
|required_arguments_match（主指标）|函数名正确，且所选函数全部schema.required参数符合冻结acceptable-value匹配器|
|typed_required_arguments_match|主指标通过，并额外要求必填值存在、符合顶层类型；bool不可算integer|
|full_local_match|strict_call检查全部GT标注参数，并拒绝GT没有的预测参数；GT空字符串有其可省略语义|
|all_heads_closed/tool_heads_closed|结束标记闭合情况，独立报告|

可选参数、未使用参数槽位不影响主指标。没有必填参数的函数只要函数名正确，主指标即可通过。`required_arguments_match`本身没有完整JSON Schema递归类型校验，存在源匹配器允许bool/number转换的情况，所以类型指标必须另列。

源acceptable-value matcher容忍部分字符串大小写/空白和数值转换；**不是统一逐字比较，也不是语义LLM裁判**。以upstream/simpletool_eval/rules.py的实现为准；数组/嵌套对象及源标注存在局限。不要自动改GT、重写城市名、增补日期或用另一个模型修复预测来抬分。

重要：历史主指标没有强制head正常闭合；闭合单独统计。新agent若要加“闭合才算对”，必须另列新指标、保留旧主指标，不能悄悄改变历史分数。

BFCL Exec没有跑官方AST/execution checker，所有这些数字都应标“本地acceptable-value匹配”；没有执行任何工具。若要官方榜单结果，应另立官方协议评测，不将本包分数冒充官方分数。

## 5. 推理设置与适用模型

历史：vLLM0.31.0、Torch2.14.1+cu130，RTX4090 48GiB，单模型独占GPU、TP1、FP16计算与attention KV、GDN状态引擎默认；eager、prefix cache开启、每批64题、max_num_seqs192、max_num_batched_tokens8192、context16384、温度0、seed1107、每头128输出token。图像/视频关闭。质量测试允许跨问题并行，不把耗时当速度基准。

已完成的12组为Qwen3.5八头0.8B/2B/4B × 0.5%/5%训练版本 × FP16/INT4。INT4为group64 affine RTN、AWQ文件布局、Marlin执行，未做activation calibration，不能称校准AWQ或iPhone MLX/KV8结果。

**infer_tools_parallel.py是原始历史脚本，硬编码dtype=half、TP=1、context等配置。** 适合原环境复现，不是任何大模型通用启动器。新agent运行35B/122B/397B或新版vLLM前应在独立文件里适配dtype、TP与模型后端并记录差异，尤其397B先前FP8/FP16出现乱码，不得照抄half。保留数据/解析/评分，可更换推理transport；这会是新runtime实验，不覆盖原结果。

上下文、tokenizer及prompt契约必须按实际目标模型检查。新训练版本若要求空think，另建prompt版本、记录哈希与对照，不能静默修改cases.jsonl。不能只因为启动成功就认定相同协议。

## 6. 接手运行命令

在包根目录执行，Python不带GPU也能跑校验和评分；推理才需要目标模型、Torch/vLLM及可用CUDA环境。本包不下载模型、不分配GPU。

```bash
cd /absolute/path/SimpleTool_Public_Benchmark_Handoff
python3 check_frozen.py
python3 -m unittest test_adapter -v
(cd upstream && python3 -m unittest discover -s tests -v)
```

给实际待测模型做tokenizer检查：

```bash
python3 check_frozen.py --model /absolute/path/to/model
```

检查head开始/结束标记均为独立token且可往返decode，并验证全部输入不会被截断。历史preflight.json只是旧tokenizer的回执，不替代新模型检查。原preflight.py依赖父目录旧plan.json，不要照抄执行；用新check_frozen.py。

原环境5题冒烟（头数可能由前5题决定，不等于全数据或完整八头验收）：

```bash
CUDA_VISIBLE_DEVICES=0 python3 infer_tools_parallel.py \
  --tag YOUR_MODEL_SMOKE --model /absolute/path/to/model --limit 5 --chunk 5
```

原环境正式评测，使用另一个全新tag：

```bash
CUDA_VISIBLE_DEVICES=0 python3 infer_tools_parallel.py \
  --tag YOUR_MODEL_FULL --model /absolute/path/to/model --chunk 64
```

冒烟后不要对同一tag用不同limit/resume硬接正式评测。--resume仅接受完全相同fingerprint及连续完整前缀输出。模糊失败先检查run.json及已保存outputs，勿盲目重发。

输出在tools-only/runs/TAG/：outputs.jsonl保留逐head原文/token/停止原因；scores.json保留逐题指标与分数据集汇总；run.json记录模型路径、config/data/script/adapter哈希、环境、状态、覆盖率和输出哈希。保留这些文件且不覆盖。

单独重新评分，不启动模型：

```bash
python3 rescore_run.py \
  --outputs tools-only/runs/YOUR_MODEL_FULL/outputs.jsonl \
  --output tools-only/runs/YOUR_MODEL_FULL/rescored-new.json
```

5题冒烟重评分需显式加--allow-prefix。不能称为全量分数。上游strict CLI读的是历史逐dataset预测格式，不能直接喂tools-only/outputs.jsonl。新rescore_run.py复用原benchlib.score_row，不改变匹配规则。

report_matched_tools.py只用于原12组完整对照，依赖旧parallel-plan与旧run目录；通用新模型用scores.json/rescore_run.py汇总，不要把旧tag替换成新模型后伪造历史对照。

## 7. 汇总与质量审计

必须给每份数据的input/filtered/scored分母、函数选择、必填匹配、带类型必填、full_local与停止情况。样本错误不删除；模型/量化/prompt/tokenizer/runtime分开记录。主全行分数是micro汇总，Seal-Tools同query不同候选数量不是独立问题，不能当独立泛化估计。

若对比论文五大来源宏平均：BFCL六份分数等权；Seal-Tools先合并1–6候选micro，再与in-domain/out-domain三者平均；ToolAlpaca两份平均；再将BFCL/SealTools/ToolAlpaca/MobileActions/OpenFunctions五家族等权。不能把BFCL Live/Non-Live/Exec当三份后又与另外四家族混成七类宏平均。

训练文本审计：新5%批次查到BFCL simple_9、simple_310两条训练文本重合；历史主分母保留，另报排除敏感性结果。旧0.5%没有同等审计，未做语义或预训练去重，不能声称完全无污染。

## 8. 文件与云端位置

根目录保存原始脚本、冻结cases、排除/顺序变更/预检；upstream保存源数据、参考答案、匹配器与测试；historical-reference保存最终12组报告/逐类表/校验及汇总，不包含大批模型原始输出。没有权重或凭据。

原本地目录：

```text
/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/artifacts/mobile_actions_scale_p005_20261109/benchmark
```

历史云端目录（本轮未联网核查是否迁移）：

```text
/inspire/hdd/global_user/shixiaoxin-253107030017/sxx/multi-head/rethinking/pretrain_level/final/glm51/sota/hermes/simpletool/final_data/vlm/mobile_actions_scale_p005_codex_20261109/benchmark
```

解压到新目录直接按相对路径使用。不要覆盖旧评测、训练目录或正在服务的H200进程。本包内任何历史命令都是参考资料，GPU资源和运行方式由用户给接手agent的新任务决定。
