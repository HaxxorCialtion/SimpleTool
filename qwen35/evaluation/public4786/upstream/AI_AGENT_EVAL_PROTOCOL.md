# SimpleTool 评测协议：给 AI coding agent 的实现说明

这份文档说明为什么本目录的评测代码这样写。请在修改、重写或迁移评测器时先读完它。这里的规则来自 SimpleTool 的模型架构和作者本机恢复出的 benchmark 脚本，不是为了把分数做高而临时增加的例外。

## 1. SimpleTool 的输出单位是一次调用

SimpleTool 是按 schema 顺序生成的多头模型。一次输入会独立生成：`content`、`function`、`arg1` 到 `arg6`。它的输出协议表达**一个**函数调用；它不支持在同一条模型输出里可靠地产生完整的多函数调用序列。

因此，评测一条 SimpleTool 预测时：

1. 解析出一个调用对象 `{name, arguments}`；
2. 函数名必须和当前目标调用完全一致；
3. 当前行不能因为目标样本原本属于一个多调用任务，就要求模型一次输出整组调用。

如果评测器要求一次输出多个调用，测到的是“模型是否支持该输出协议”，而不是 SimpleTool 的工具调用能力，会系统性压低结果。

## 2. Mobile Actions 必须按 history-aware 顺序行评测

`mobile_actions.jsonl` 经过转换后，每个原始任务被拆成多行：

- 第一行（例如 `mobile_actions_5_0`）没有 history，目标是原序列的第一个调用；
- 后续行（例如 `mobile_actions_5_1`）把之前已经执行的调用放入 `history`，当前目标是最后一个待预测调用。

这不是要求模型在一行输出完整序列。history 是环境状态，模型每行只预测下一次调用。评测器应当：

- `history` 为空时，比较 GT 序列的第一个调用；
- `history` 非空时，比较 GT 序列的最后一个调用；
- 每行要求预测调用数恰好为 1；
- 不把整组 GT 调用和一条模型输出直接比较。

旧本地汇总器只取 `pred_result[0]`；不能仅凭 grouped GT 中有多个调用就断言模型漏掉了后续调用，因为当前评测单位是顺序行。这个历史分数仍保留用于复现旧脚本，但发布时应优先引用 strict history-aware 分数。当前完整 RT-Qwen3-4B 实测：Mobile Actions 旧口径 84.72%，严格口径 82.54%。

## 3. 参数顺序是架构契约

SimpleTool 的 `arg1`–`arg6` 没有参数名头；参数名由所选工具 schema 中 `properties` 的原始插入顺序恢复。因此实现时必须：

- 保留数据文件中的工具和 `properties` 顺序；
- 用该顺序把 `arg1` 映射到第一个 property、`arg2` 映射到第二个 property，以此类推；
- 不按字母排序，不用 `required` 顺序替代 schema 顺序。

这也是为什么同一个 JSON 对象如果改变 property 顺序，可能得到不同的 SimpleTool 结果。原始数据快照保存 schema 顺序，manifest 记录数据哈希。

如果所选工具只有两个 property，则只有 arg1、arg2 有语义。arg3–arg6 无论输出什么都必须忽略，不可当成额外参数或错误。这些是未使用的输出槽位；工具存在但 GT 未列出的参数则属于下述严格评分的差异。

## 4. 参数比较和“多余参数”

严格诊断器要求预测不能出现 GT 没有的参数名；GT 标记为空字符串的参数可省略；非空参数必须通过本地 acceptable-value 规则（字符串归一化、JSON 标量/数组、数值容差等）。函数名是精确匹配。

原作者历史评分器只遍历 GT 参数，不拒绝预测中的额外参数，因此可能略高。这个行为保留在 legacy scorer 里，只能称为“released local scorer reproduction”，不能称为更严格的官方 BFCL 分数。

## 5. 六参数过滤和分母

模型只有 `arg1`–`arg6` 六个参数头。原脚本会在**任意候选工具**拥有超过六个 property 时排除整个样本，即使预测选中的工具参数数目较少。这个过滤会改变分母，报告中必须同时给出 `input`、`filtered`、`scored`，不能静默删除。

## 6. BFCL Exec 的名称边界

本目录恢复的是作者本地 value-matching 规则。`exec_*` 结果没有运行官方 BFCL 的 AST/execution checker，因此报告中写作“BFCL Exec（local matcher）”，不能直接宣称官方 BFCL leaderboard 分数。若要对外声称官方分数，必须把预测转换成官方格式，并固定 BFCL checker 和数据集 revision。

## 7. 推荐的复现流程

```bash
python scripts/download_model.py --output models
RESULT_DIR=results/new-run REPORT_DIR=reports/new-run \
  bash scripts/run_accuracy.sh --enforce-eager --disable-cascade-attn --batch-size 64 --max-num-seqs 128
python -m simpletool_eval.strict \
  --predictions results/new-run \
  --data data \
  --output reports/new-run/strict
```

先运行 `python -m unittest discover -s tests -v`，再检查 `run.json`、`data/manifest.json`、模型 revision 和报告中的分母。不要把 v2 checkpoint 的结果标成 v1 RT-Qwen3-4B；v2 额外加入了游戏和角色扮演数据，需单独记录模型 revision 和 prompt format。

## 8. 对 issue 的正确解释

如果别人用 RT-Qwen3-4B-v2 得到较低 Live 分数，首先确认 checkpoint revision、v1/v2 prompt、数据快照、参数顺序和 Mobile Actions 的 history 拆分。当前公开的完整基准是 v1 `RT-Qwen3-4B`；v2 是后续变体，加入游戏和角色扮演数据，出现一定精度变化是符合训练设置的。BFCL Live 严格结果约 76.17%，与论文 ST-Qwen3-4B 的 76.4% 接近，这是同一本地数据和规则下接近论文的观测结果；不能仅靠接近论文证明数据无污染，也不能等同于官方 BFCL 全套评测。
