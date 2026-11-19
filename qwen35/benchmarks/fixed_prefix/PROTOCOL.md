# SimpleTool H200 固定公共前缀、变化环境输入测试

日期：2026-11-13。最终数值以 `REPORT.md`、`RESULTS.json`、`evidence/` 中的逐请求记录为准。

## 测试边界

所有速度测试都不生成 content。模型训练有八个分支，但本次最多使用七个工具分支：`function,arg1,arg2,arg3,arg4,arg5,arg6`。分支是同一个模型、同一个 LM projection 的不同序列，并非七个不同模型或七个独立输出层。解码使用真实 batch forward。

三组场景是可复现的控制策略输入和函数输出案例；没有连接机械臂、游戏或数字人引擎，没有实际执行工具。它们验证变化观测下的推理和工具调用，不构成物理闭环验收。

本轮是一个逻辑请求接一个逻辑请求，同请求内各 head 并行。没有测多用户饱和吞吐，也不包含 HTTP、网络传输、tokenizer、环境步进、工具执行、冷加载和 CUDA graph 捕获时间。

## 输入与对照

纯 infra：公共 P=512 tokens，变化 S=512 tokens，总长 1024。每个重复使用不同后缀 token 序列，B∈{1,2,4,7}，每个 head 强制生成 T∈{16,32} tokens，每格 20 个不同输入。关闭自然结束仅用于这组定长核测试，不能用它证明实际工具生成长度。

每个输入有两种对照：唯一 cache_salt 的跨请求冷缓存，命中必须为 0；固定 namespace 中已计算并保留的公共边界，命中必须为 512，新增必须为 512。两种对照都先完成公共 prefill，再启动各工具头，因此同请求内跨头公共部分均复用。每个 head 还需处理自己的一个起始 marker，这不算在“新增公共 tokens”里。

真实场景：机械臂、游戏 NPC、数字人各有 60 个不同观测。system、策略和 tools 固定，environment 与观测 ID、坐标、事件、参数持续变化；预期调用不注入模型输入。每个场景有独立、固定的公共前缀，长度由真实 tokenizer 决定。变化后缀补到 512 tokens，补齐使用显式标注为无作用的诊断标点，不是自然观测本来就有 512 tokens。原始文本、token IDs 和 padding 数可审计。

三个场景交错运行，三个公共边界同时保留。计时前的预热输入、正式 180 条输入、其后 600 条持续变化输入相互分离。稳态测试不重复两道题十五轮；600 条额外输入各自有不同观测 ID。

## 缓存与调度

1. 先对每个固定 system/tools 前缀做一次不生成 token 的 prefill，等待 CUDA 完成并发布公共末端状态，保留其缓存块引用。
2. 新观测到来时，以该前缀状态继续计算变化后缀。Qwen3.5/3.6 为混合 full attention 与 GDN 架构，复用对象包括 full-attention KV、GDN recurrent state 和 convolution state。仅保存普通 KV 不足以正确恢复。
3. 新的完整公共状态物化后，才提交所需工具头；各头公共 KV 只读共享、递归状态独立可写。禁止不同 head 共享可变 GDN state。
4. 工具头同时进入原生 vLLM decode batch，marker、终止 token 都从实际 tokenizer 得到。各头遇到自己的闭合 token 即独立停止。
5. function 完成后，运行时从它实际选择的函数 schema 解析参数数目，等待这些必要 head。该时刻记为 tools-ready。测试没有读取 gold label 来决定哪些头应该结束。多余 head 的物理排空耗时另记 wall_ms，本轮没有实现读到函数后立刻撤销多余头。

`prefix_match_unit=1` 是缓存哈希/匹配单位，不代表 GPU KV page 只有一个 token；实际 page 大小和 GDN 对齐由引擎决定。私有 overlay 支持发布精确末端边界，无须把边界凑到 528 等物理块大小。提前单独计算 system/tools 是本轮取得跨题缓存命中的必要步骤。

物理布局还需区分完整页与末页：完整公共 KV 页共享，写入 marker/增量 token 的非对齐末页由原生 copy-on-write 复制为各 head 的私有页。缓存命中计数表示逻辑 prefill 计算复用，不代表零拷贝、全部 prefix 字节永远只有一份，或每个 KV 元素只从 DRAM 读取一次。GDN/conv 状态也必须复制为独立可写状态。这些拷贝开销计入 TTFT/E2E。`shared-kv-20261110/shared_attention.py` 的专用 Triton attention 内核不属于本轮激活路径。

## 图与状态保护

公共 prefill 使用有界、形状专门化的 CUDA graph，原生 decoder 使用 FULL_DECODE_ONLY，覆盖 batch 1…8。公共图仅允许在预热期间捕获，计时期间禁止新捕获。真实 tokens、positions、slot mapping、attention metadata 每次重新写入稳定输入 buffer；没有把旧输入输出当新结果复用。

捕获时会备份并恢复涉及的 GDN/conv 状态，避免 warmup/capture 对递归状态产生额外更新；多卡捕获使用原生分布式 graph context，注册 all-reduce buffer。每个 rank 都保留图计数、fallback 计数、allocated/reserved memory 和 cache pool 快照。

图数量、eager fallback、显存池和长运行首尾速度均从收据核对。没有证据时不把变慢归因于“缓存满了”。高并发调度、长上下文、视觉、多租户淘汰和 checkpoint service 化不在本次验收范围。

## 计时定义

- `prime_ms`：本次公共输入提交，到状态完成、发布可供分支复用的主机端耗时，含调度与发布开销。已命中的前缀不重复计算。不是单纯 GPU kernel 时间。
- `first_ms`：同一请求起点到某 head 的第一个输出 token 的时间，包含 prime。
- `tpot_host_ms`：某 head `(last_ms-first_ms)/(实际输出 token 数-1)`；实际输出包含自己的结束 token。
- `tools_ready_ms`：从请求起点到 function 和其 schema 所需参数 head 全部结束、完成参数组装的时间；对应最慢必要 head 的关键路径，附带主机轮询与组装开销。
- `wall_ms`：本批所有实际启动 head 都结束的耗时，可能晚于 tools-ready。二者分别交付。
- 由 `1000/tools_ready_ms` 计算的 Hz 只表示本测试单请求推理预算，不是整个真实控制系统已经达到该闭环频率。

原始 arrivals 保留每个 head 每次返回的 token 数和时间，不能用所有 head 总 token 数除 wall 来替代单头 TPOT。

## 正确性与质量

每个模型先对 12 个真实输入分别比较批量分头、整段独立冷重算、相同切分独立单头重算，以及“固定公共前缀→变化后缀→marker”的缓存批量与独立单头路径。正式“固定前缀→变化后缀”缓存路径的批量/单头对照严格逐 token 一致才进入计时；无固定前缀时 batch/单头的差异另行审计。整段冷 prefill 与切分 prefill 是另一个数值执行路径，差异作为独立审计项保留，不能宣称全路径无损或普遍逐 token 一致。分词器、结束标记、完整输出和失败尝试保留。逐 token 对照检验实现一致性；180 条调用对 gold 的评分检验任务质量，两种验收不能互相代替。

工具属性顺序与 required 顺序一致，所有必填参数靠前。评分按实际函数 schema 映射 arg1…，不考核不存在的尾部参数。此批所有声明参数均为必填。数据 gold 经过同一评分器回环检查。

结果中同时报告函数名正确率、完整必要参数正确率、结束 token 覆盖和有效 tools-ready 覆盖；无效函数或缺失参数不伪装成成功决策。速度分布与正确调用子集分布分别保留。

## 路径与重放

本机：`/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/experiments/simpletool_h200_speed_20261113/scenario_v2/`

云端：`/inspire/qb-ilm2/project/new-paradigm-of-manipulation/shixiaoxin-253107030017/simpletool_h200_speed_20261113/scenario_v2/`

场景生成：`make_scenarios.py`；实际引擎驱动：`bench_scenarios.py`；启动配置：`launch_scenarios.sh`；批次配置：`controller.py`；派生记录：`DRIVER_PROVENANCE.json`。原始引擎位于父目录的 `payload/`，云端对应父目录文件及其 `runtime/`，不修改共享 vLLM 安装。

模型权重路径、量化配置、vLLM/Torch/CUDA 版本、TP 数、异步调度选择均记录在各运行 `config.json`。4B/9B INT4 是云端 AWQ/Marlin 路径，不能用作 iPhone MLX 或 KV8 测量；大模型 FP8 与小模型 INT4 也不是同精度的参数 scaling 对照。
