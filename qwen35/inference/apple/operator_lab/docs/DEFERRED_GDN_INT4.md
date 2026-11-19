# int4 解码：GDN 延迟写回与 Swift 移植

2026-10-08，独立工程。4B/2B 都加载现有 compact affine int4/group64 权重，不重新量化，不运行新的 FP16 语言推理。计算、卷积历史及 GDN 状态仍为 FP32。

本阶段把完整 Python/Swift 四通道解码接到一种新的状态表示。**已观察到真实请求的 MLX 分配峰值下降；新增时延差没有通过 10% 噪声门槛。** 配置显式启用，默认仍是逐步生成 dense GDN 状态。Mac Release、完整 iOS 库和 90 个独立 iOS shader 变体编译通过，手机没有执行。

## 状态表示与数值契约

普通 recurrence 每步读取矩阵并写出下一份矩阵。新表示保留不可变 base，另存每个分支最近几步的 key、gate 与 delta，到第 2/4/8/16 步才重建 dense base。重放严格依次执行原来的 FP32 乘法、加法及归约，不把更新改写成一个代数低秩求和，也没有改用半精度状态。

首次 base 是公共 prefill 的单份 B1 矩阵，分支历史独立；写回后 base 含各分支的独立矩阵。batch 缩小时，公共 B1 base 保留，卷积与历史按存活通道过滤；owned base 按通道过滤。KV prefix 始终单份。所有更新都是 MLX 函数式输出，不修改借出的 Metal buffer。

`simpletool/deferred_gdn.py`、`simpletool/deferred_model.py` 是 Python 实验入口。Swift 对应 `DeferredGDN.swift` 与 `StaticDecode.swift` 的可变宽度状态展开；GDN 状态由两个数组变为五个数组，full attention 仍使用两个可写后缀数组，公共 prefix 由闭包引用。2B 的三 token 完整标签仍逐位置推进，末位置才计算词表。

源码保留最初的空 history 地址空间编译失败：`artifacts/deferred_gdn_first_empty_history_failed.py` 和对应日志。修复使用 `auto` 推导 history 指针地址空间；小 Dv 增加越界 guard。未改动共享 MLX 环境或 Zcode 文件。

## 正确性证据

| 检查 | 覆盖 | 结果 |
|---|---|---|
| Python recurrence | 五种 head/value 形状，B1–4，间隔1/2/4/8/16，共享/owned seed，每步输出和重建矩阵 | 3080/3080 逐位相同；200份 seed 哈希不变 |
| Python 重排与 gate 边界 | B4→B3→B2→B1，重排，gate/beta 为0/1/极小值 | 280/280 逐位相同 |
| Python 完整模型 | 两模型×四种间隔×五条真实指令；另加16间隔/8步调度 | 50/50 case；单路和四路共400次通道 token 对拍 |
| Swift 自然停止 | 两模型，间隔1/2/4/8/16，基线和16间隔的8步调度，五条指令 | 70/70 case，560次 token 与 ByteLevel 文本对拍 |
| Swift 完整缓存 | B1–4，间隔2/4/8/16，1/8步调度，19 token，精确suffix容量 | 64/64 case；卷积、重建矩阵、KV后缀逐位相同，source 哈希不变 |
| 最终 Swift binary | 基线/16间隔各模型；另测16间隔+both投影+8步调度 | 6/6 case，48次 token/文本对拍 |
| 单请求内存探针 | Python/Swift各12个独立子进程，实际自然停止 | 全部 gold token 对齐、prefix哈希不变 |

较早的非内存 Swift 验证报告中 `prefix_unchanged` 是未做 hash 检查的占位值，不能用于哈希验收。最终 binary 在该模式返回 null，另有 `prefix_hash_checked=false`；本阶段前缀不变的证据来自独立完整缓存和内存探针。原始报告保留。

对拍使用同一 int4 checkpoint 的 serial gold，不是重新量化质量验收。2B 的部分通道仍到达64 token 上限；不能把对拍通过解释成输出质量或闭合率已经合格。

## 单请求内存

每项为一个独立进程，只加载一个模型、执行一个 B4 请求。prefill 后清理本进程 allocator cache 并重置峰值。指标是 **MLX active allocation peak**，不包括整个 App 的 physical footprint，也没有 iPhone 内存结论。cache limit 为256MiB软限。

Swift Mac Release 的实测：

| 模型/输入packet | 逐步写回 MiB | 间隔4 MiB | 间隔16 MiB | 间隔16减少 MiB |
|---|---:|---:|---:|---:|
| 4B case0 | 2691.7 | 2586.4 | 2397.9 | 293.8 |
| 4B case2 | 2691.2 | 2585.8 | 2383.6 | 307.6 |
| 2B case0 | 1188.7 | 1191.7 | 1142.0 | 46.7 |
| 2B case4 | 1188.7 | 1191.7 | 1144.2 | 44.5 |

两条4B输入的总峰值约降低11%；2B的间隔4反而略增，不能假设每种间隔都降低峰值。Python 4B两条输入的间隔16分别减少308.1/321.6MiB。请求停止时间、batch收缩、prefill长度及重建点会影响峰值，不能将短请求结果直接外推到长上下文或所有多轮。

## 时延与取舍

统一比较同一模型对象/权重、一次共同 prefill、完整通道标签、每头强制96 token、忽略停止，十个单路/四路 runner 在同进程轮换五轮。首次路径编译单独记录。4B做两个独立进程，2B一个；外部负载未锁定。该benchmark峰值包含十个 runner，不能作为上表的单请求内存。

Python 两批4B：基线B4为2.033/2.024s，间隔2为1.998/1.997s，间隔4为2.011/1.957s，间隔16为2.113/2.116s。2B间隔2约0.7%差异、间隔16约慢5%。较小间隔无足够证据确认新速度收益；16间隔为内存候选，会增加 history 重放计算及编译形状。

Swift Release 实测四路时延：

| 批次 | 基线 s | 间隔2 s | 间隔4 s | 间隔8 s | 间隔16 s |
|---|---:|---:|---:|---:|---:|
| 4B第一批 | 2.080 | 1.969 | 1.973 | 2.006 | 2.129 |
| 4B独立复测 | 2.037 | 1.968 | 1.997 | 2.032 | 2.130 |
| 2B | 1.031 | 1.015 | 1.023 | 1.045 | 1.087 |

间隔2的4B时延差为5.3%/3.4%，间隔16为慢2.4%/4.6%；2B间隔16慢5.4%。所有新增差异低于10%门槛，效率观测仍约63%–66%。没有将这条路线晋升为速度优化默认。原始数据见 `reports/swift_deferred_{4b,4b_repeat,2b}_ab_20261008.json`；汇总包含每个间隔的相对时延和噪声门槛判断。不能称为达到硬件理论极限或98%效率。

## 显式运行

```bash
cd /Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine
ST_SWIFT_EXECUTION=compiled ST_SWIFT_DEFERRED_GDN=16 \
  artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI \
  artifacts/4b_int4_compact reports/swift_suite_4b_0_20261008.packet.json \
  reports/my_deferred_validation.json

# 实际B4单请求内存探针
ST_SWIFT_BENCHMARK=deferredMemory ST_SWIFT_DEFERRED_GDN=16 \
  artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI \
  artifacts/4b_int4_compact reports/swift_suite_4b_0_20261008.packet.json \
  reports/my_deferred_memory.json

# 各路径完整缓存对拍 / 五轮公平时延对照
# 分别设置 ST_SWIFT_BENCHMARK=deferredState / deferred

/Users/cialtionshi/mlx_env/bin/python run_deferred_swift_suite.py --phase benchmark
```

`ST_SWIFT_DEFERRED_GDN` 接受1/2/4/8/16，默认1。自然停止验证要求 compiled 模式。`ST_SWIFT_DECODE_CHUNK=8` 可显式叠加；GPU块内终止后的多算token丢弃，存活分支在边界缩小。Python仍为独立实验类，没有接入默认HTTP、图像或会话缓存接口；导出 canonical GDN 状态需要 materialize。

本阶段快照：`artifacts/deferred_gdn_snapshot_20261008/`。含最终 Mac CLI、两份资源bundle、源码归档；权重、源码、binary、metallib与归档哈希写入 `reports/deferred_gdn_summary_20261008.json`。Swift port前的源代码另存 `artifacts/deferred_before_swift_port_20261008.zip`。

下一步仍围绕int4投影、公共状态读取的dispatch局部性、重放寄存器/ALU成本及分支收缩调度。Swift原生prompt编码、图像、多轮完整状态提交、iPhone实机/耗电、量化质量和llama.cpp严格对拍门槛仍待完成。
