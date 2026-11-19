# Swift int4 固定后缀与整步编译：2026-10-08

后续投影kernel已接入并完成验证，见 [SWIFT_PROJECTIONS_INT4.md](SWIFT_PROJECTIONS_INT4.md)。下文保留本阶段原始结果；“尚未接入streaming/fused”描述为本阶段快照。

继续完整 Swift 路径的执行调度优化。4B/2B 均保持 affine int4/group64 权重和 FP32 浮点计算；没有新增 FP16 模型推理。完整原生手机产品及98%目标仍未验收。

## 具体改动

1. Fork 后 suffix K/V 采用固定 capacity `[B,Hkv,capacity,D]`，通过 pinned MLX-C 的 dynamic slice update 写入当前槽位。attention 用动态 used mask 排除所有未写位置。prefix K/V 仍物理单份，不复制到四个 suffix buffer。
2. 通过显式 tensor 状态实现整步编译：输入 token、used 标量及每层 conv/GDN 或 suffix K/V，输出贪心 token 与所有新状态。RoPE 从 prefix length + used 计算，每个分支显式广播 offset，保留已验证的旧版本 RoPE 绕过。
3. compiled wrapper 不保存初始 GDN 状态的额外快照，首步只读 B1 seed，后续为独立 batch 状态。停止时仍过滤各分支矩阵和 KV，形状改变后编译器选择相应 specialization。
4. 2B 完整三 token 标签的前两个位置只更新缓存，不执行 vocabulary projection。最后标签位置才产生第一个预测。compiled marker graph 也省去无用的最终输出计算。
5. prefill 只评估发布的缓存，不强制评估被丢弃的 final hidden；依赖这些缓存的全部算子仍执行。最后一层只用于缓存之外的输出/MLP/norm 因 lazy dependency 不再强制执行。跨 chunk 状态已额外实测。
6. 每步贪心预测和状态在一个 evaluation 调用中完成，复用编译函数；固定计数 benchmark 每轮 reset 状态，保持同一个 immutable prefix 和权重。

这仍是**函数式 MLX buffer 所有权**。固定容量意味着稳定形状，不等于已证明每步零分配或原地写同一个 MTLBuffer；未对借出的 MLX Metal buffer 做写入。原生 owned GDN in-place kernel 尚未进入完整模型。

## 验证

Release 三条执行线：`dynamic`、`staticCache`、`compiled`。每个模型5条真实指令，每条4个通道；serial 和 B4 均与 matching int4 Python serial gold 比较。自然 EOS/关闭标签停止，上限64 token。

| 执行线 | 2B serial / B4 | 4B serial / B4 | 解码文本 |
|---|---:|---:|---:|
| dynamic | 各20/20 | 各20/20 | 全部相等 |
| staticCache | 各20/20 | 各20/20 | 全部相等 |
| compiled | 各20/20 | 各20/20 | 全部相等 |

总计30条 case、240次通道比较通过，停止后出现B4/B3/B2/B1。上限处的 max_tokens 不视为自然闭合质量通过。仅生成 command 数据，不执行。

- 固定 KV 更新回归16/16：raw/compiled × B1..4 × capacity3/17，检查每个槽位直到最后位置，以及原始输入保持不变。
- RoPE 独立12/12回归保持通过。
- 4B 501-token真实长前缀，跨256+245两次chunk；staticCache/compiled 的 serial 和 B4 各4/4对gold一致。
- 完整 Release Mac 与 iOS arm64 库编译通过。iOS 未签名、未运行在手机。

原始记录：

- `reports/swift_static_release_suite_20261008.json` / `.log`，每条 case 有独立日志与结果。
- `reports/swift_static_tail_final_regression_20261008.json`、`swift_static_rope_regression_20261008.json`。
- `reports/swift_static_long_packet_20261008.json` 及 `swift_static_long_{staticCache,compiled}_20261008.json`。
- `reports/swift_static_stateonly_{release,ios}_build_20261008.log`。
- 汇总/源码与二进制 SHA：`reports/swift_static_summary_20261008.json`；由只读 `make_swift_static_report.py` 生成。

## 热态比较

同进程同模型/权重，prefill一次，单头speech及实际四种标签；固定96 token/头，忽略停止，温热完整路径后，六种组合轮换执行5轮。计时包含完整标签状态推进和decode，排除prefill、reset及首次完整预热。无需重新下载或量化。

| 模型/批次 | dynamic单头 | dynamic四头 | compiled单头 | compiled四头 | 四头时延降低 | compiled单/四头效率 |
|---|---:|---:|---:|---:|---:|---:|
| 4B 第1独立进程 | 1.527s | 2.365s | 1.284s | 1.988s | 15.9% | 64.6% |
| 4B 第2独立进程 | 1.534s | 2.364s | 1.285s | 1.985s | 16.0% | 64.7% |
| 2B 一批 | 0.846s | 1.293s | 0.657s | 1.016s | 21.4% | 64.6% |

所有热态及首次路径的fixed-count tokens与dynamic reference相等。4B compiled四头约193 aggregate token/s，2B约378。2B尚未独立复测。

单独固定capacity四头约2.333s/1.283s，仅约1%的差异，低于既有10%噪声门槛；不单独声称提速。编译方案同时改善单头和四头，所以未提升四头相对于单头的效率，更未接近98%。不可把当前Swift结果与此前不同kernels/runtime的Python结果直接比较。

首批binary仍强制评估prefill final hidden，复测binary只评估cache；这次差异在计时区间之外，Release三路径对拍在最终binary上完成。首批binary SHA另外保存在 `reports/swift_static_first_binary_20261008.json`。外部竞争负载与热状态未受控，两个5轮进程支持本条件下的观测，不是硬件永久性能保证。

`first_run_s` 在同一进程按顺序测量，后测路径已享有前测路径的Metal specialization缓存，不能用作公平冷启动比较。六个runner和共同seed同时存在，MLX peak约3.54GB也不能当作单路径部署或phone RAM。

## 下一步依据

仅按safetensors header统计，4B MLP张量约1.274GB，占language payload约53.8%；tied embedding/projection约357.6MB。完整Swift仍用原有row-reuse投影，尚未接入Python中最新streaming QMV与gate/up+SwiGLU单dispatch。后续应继续针对这些大矩阵的register占用、shared输入、dispatch数量及一次处理多步的host/GPU同步尝试。

4B FP32 GDN matrix每分支50.33MB，B4矩阵每步至少涉及约402.65MB读写；加上language payload的条件估算约2.772GB/步。这个估算排除conv、KV、activation、cache命中与实际重复读取，**不是实测DRAM流量或硬件上限**。因此不能仅用量化权重字节数宣布已接近带宽极限。完整state与phone内存还需单路径测量。

原生prompt编码、图像/视觉位置、完整多轮增量缓存、owned状态原地更新、手机实机/功耗和llama.cpp既有门槛仍未通过。iPhone当前不可连接，仍有独立工程工作可继续。

## 运行及实现入口

```sh
# 在项目目录，使用已编译的Release binary
ST_SWIFT_EXECUTION=compiled artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI artifacts/4b_int4_compact reports/swift_4b_natural_packet_20261008.json reports/swift_compiled_result.json
ST_SWIFT_BENCHMARK=1 artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI artifacts/4b_int4_compact reports/swift_4b_packet_20261008.json reports/swift_benchmark.json
artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI --static-regression
/Users/cialtionshi/mlx_env/bin/python check_swift_static.py
```

SDK入口：`Int4Model.validatePacket(_:execution:)` 和 `benchmarkPacket(_:tokens:rounds:)`；模型、状态与编译封装分别在 `Model.swift`、`StaticDecode.swift`、`BenchDecode.swift`。底层调用对照固定版本 [MLX Swift compile源代码](https://github.com/ml-explore/mlx-swift/blob/0.31.1/Source/MLX/Transforms%2BCompile.swift) 及 [MLX-C dynamic slice update](https://github.com/ml-explore/mlx-c/blob/0726ca922fc902c4c61ef9c27d94132be418e945/mlx/c/ops.h)，本项目没有修改这些依赖。
