# int4 kernel 管理：固定 history、动态 phase 与融合更新

2026-10-08，继 `DEFERRED_GDN_INT4.md`。继续使用同一2B/4B compact int4 checkpoint，未重新量化，未新增FP16语言推理。浮点计算与GDN状态保持FP32，默认运行配置未变。

本阶段针对两种成本：history长短改变引起的不同编译输入形状，以及每层三个独立history更新算子。它已接入完整Python和Swift模型，不只是孤立kernel。手机尚未执行。

## 执行与所有权

旧history形状是 `[B,P,...]`，P每步变化。新history容量固定为间隔减一，有效长度由运行时int32 phase决定。代码只有defer和write两种步函数，phase不再是Metal模板常量。首次base仍是公共B1状态，分支history独立；write步产生owned B状态。write后保留旧history，phase归零，未来槽位被覆写前不能读。

`fixed` 使用三个函数式slice update；`fused` 在GDN Metal kernel中同时输出更新后的key、gate、delta history，跳过三个更新调用。key每个key head只由一个value head的dv0四个lane写一次，gate仅由dv0/local0写，delta由各dv/local0写。输出与输入是不同的MLX数组，没有向借出的buffer写入。

动态有效长度并不允许动态分配整块matrix。host按已知used位置选择write/defer，write函数才申请dense矩阵。Swift直接使用host的used计数；Python读取调用方创建的used标量，不读中间预测。完整性能计时包含这些host工作。

多步自回归在write边界切分GPU块，非write段使用动态phase的同一个块函数。中间prediction留在GPU，末prediction仍待消费；停止后丢弃多算token，存活通道在块边界过滤。base为B1时保留公共base，owned base及独立history按存活通道重排。

## 验证

| 对象 | 覆盖 | 结果 |
|---|---|---|
| 两个Python kernel版本 | 五种head/value形状，B1–4，间隔2/4/8/16，共享/owned seed，连续多次write；未使用槽位填NaN；gate边界及B4→B1重排 | 6320项y/重建matrix逐位相同，seed哈希不变 |
| Python真实指令 | 两模型×两实现×四间隔×1/8步×五指令，单路及B4 | 160 case，1280次通道token对拍 |
| Python完整缓存 | 两模型/两实现，B1–4，各间隔；1/8步，间隔2/16另含2/4步，19 token/精确容量 | 192 case，卷积、重建matrix及KV逐位相同，source哈希不变 |
| Swift真实指令 | 固定/融合全部四间隔与1/8步；原dynamic间隔1/16回归；两模型五指令，单路及B4 | 200 case，1600次token与ByteLevel文本对拍 |
| Swift完整缓存 | 两模型/两实现，各间隔，B1–4，1/8步，19 token/精确容量 | 128 case，完整缓存逐位相同，source哈希不变 |
| SDK/build | Mac Release、完整iOS库、独立shader | 两库构建通过，120个iOS shader变体编译通过 |

自然停止对拍不是量化质量验收，2B部分通道仍达到64 token上限。只有独立state和memory probe记录source hash；普通Swift验证的 `prefix_hash_checked=false` 和 `prefix_unchanged=null` 不作为哈希证据。

首个动态phase实现误把MLX零维标量按指针下标读取，Metal编译失败。已按本地Cmlx接口修正为scalar reference读取。失败源码及两份失败日志保留为 `fixed_history_*first_scalar_failed*`。没有修改共享MLX环境。

## 编译输入组合与性能条件

在固定96-token完整模型中，16间隔的动态history实际观察到B1有16种、B4有32种输入形状组合。固定history只有B1一种/B4两种state形状；计入write/defer调用模式后为B1两种/B4四种。**这不是内部compiler缓存条数或GPU pipeline计数。**

性能对照共十四个runner，同一套权重与公共prefix，完整标签、每头96 token、忽略停止。五轮以3个位置轮换次序；Python/Swift分别做两次独立4B进程和一次2B进程。首轮路径依次预热，不能当作公平冷GPU测试。benchmark峰值包含全部runner，不是部署RAM。

Swift完整96-token四路暖时延：

| 批次 | dense基线 s | 旧dynamic16 s | fixed16 s | fused16 s |
|---|---:|---:|---:|---:|
| 4B第一批 | 2.005 | 2.133 | 2.072 | 2.037 |
| 4B独立复测 | 2.003 | 2.130 | 2.067 | 2.038 |
| 2B | 1.027 | 1.073 | 1.076 | 1.071 |

fused16相对旧dynamic16的4B差异为约4.5%/4.4%；Python独立两批为4.0%/4.2%。都低于10%门槛。相对dense基线，新16间隔仍略慢；4间隔没有稳定胜出。四头效率的变化可能来自单路变慢，必须同时阅读single与batch4时延。没有新增可确认的持续模型吞吐突破。

## 独立请求与系统内存

每个probe子进程只加载一个int4模型、执行两次自然B4请求，decode前清理自身allocator cache并重置MLX峰值。每个runtime五组输入/重复×三个history policy，另补相同输入的dense基线，共40个子进程/80个请求。4B case0有两个独立重复，其余输入一个重复。

parent仅查询自己创建的PID，每20ms读取Darwin `proc_pid_rusage`/`RUSAGE_INFO_V4`；布局按本地SDK `sys/resource.h` 核验，296字节。记录sampled footprint与退出前观察到的kernel lifetime-max footprint。它们包括加载、prefill、编译、hash检查和两次请求，不能与decode重置后的MLX active峰值混为一谈；采样可能扰动probe时延。Python额外保留自身 `task_vm_info` 的阶段快照。

first request表示新进程的首次请求，GPU、磁盘缓存和同机负载未锁定。Swift每个请求重新prefill一次，Python两个请求复用同一prefilled source；比较需在同一runtime/policy实验内进行。上述均不是iPhone App物理内存或耗电验收。

Swift实际请求（计入每请求的compiled context、标签推进、自然停止和batch收缩；下表为decode窗口，不含prefill）：

| 4B输入/进程 | dense首次 s | dynamic16首次 s | fixed16首次 s | fused16首次 s | dense/fused16 MLX峰值 MiB |
|---|---:|---:|---:|---:|---:|
| case0第一次 | 0.359 | 0.454 | 0.324 | 0.317 | 2691.7 / 2416.0 |
| case0独立复测 | 0.334 | 0.443 | 0.322 | 0.315 | 2691.7 / 2416.0 |
| case2 | 0.388 | 0.546 | 0.393 | 0.389 | 2691.2 / 2415.3 |

相对旧dynamic16，新固定/融合表示的Swift真实请求观测少约27%–30%，同进程第二请求也保持这个差异：Swift每请求重建compiled context，旧history多种输入形状的trace/编译开销并未完全摊销。**相对dense基线，没有稳定超过10%的新时延收益**：case0两次差异不同，case2基本相当。Python主要改善新进程首次请求，重复请求时差很小。不能把旧实验线的改善写成模型GPU吞吐提高30%。

新fused16在三个4B probe中，相对dense的MLX峰值减少约276MiB（约10%）；但比旧dynamic16增加约18–32MiB。2B新表示比dense慢约7%，MLX峰值少约31MiB。默认未改变，不是每个模型都值得采用相同policy。

Mac kernel lifetime-max footprint 的4B case0两次独立观测为dense3136/3053MiB、dynamic16 3065/3063MiB、fused16 2990/2988MiB。case2为dense3092MiB、fused16 2980MiB。这是全进程/两请求/退出前观测，不是decode-only或iPhone峰值；dense两个进程本身有约83MiB差异，不能把减少量当作固定收益。2B case0的fused16 footprint略高于dense，case4更低。

全部原始时延、每请求gold/hash、footprint轨迹与对应范围见 `reports/fixed_history_summary_20261008.json`。

## 入口与快照

```bash
cd /Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine
ST_SWIFT_EXECUTION=compiled ST_SWIFT_DEFERRED_GDN=16 ST_SWIFT_GDN_HISTORY=fused \
  artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI \
  artifacts/4b_int4_compact reports/swift_suite_4b_0_20261008.packet.json \
  reports/my_fused_history.json

# 可选 ST_SWIFT_GDN_HISTORY=fixed；默认dynamic。
# 多步显式设置 ST_SWIFT_DECODE_CHUNK=8。
# ST_SWIFT_BENCHMARK=history / historyState / historyProbe 对应三种实验。
/Users/cialtionshi/mlx_env/bin/python run_fixed_history_experiments.py --phase benchmark
/Users/cialtionshi/mlx_env/bin/python run_fixed_history_experiments.py --phase probes
/Users/cialtionshi/mlx_env/bin/python run_fixed_history_experiments.py --phase baseline_probes
```

Python入口 `CompiledFixedHistoryStep(model,interval=16,fused=True)` 需先 `prepare_deferred_model(model)`。状态导出要用materialize和相应phase；未接入默认HTTP/图像/多轮会话，不能把五数组history缓存直接作为原两数组GDN缓存提交。

源码、最终CLI及资源bundle保存在 `artifacts/fixed_history_snapshot_20261008/`，含int4 checkpoint、源码、binary、shader及archive哈希的汇总为 `reports/fixed_history_summary_20261008.json`。此前的Swift/deferred快照保留。模型/缓存源文件在验证后没有改变；最终只调整了benchmark轮换次序，最后的完整benchmark仍逐轮检查token。

手机实机、原生prompt编码、图像、多轮会话提交、量化质量、llama.cpp严格对拍和98%效率门槛仍未完成。下一步继续针对公共base访问局部性、矩阵投影/重放ALU与量化投影kernel的真实瓶颈，而不从减少算子数量直接推断提速。
