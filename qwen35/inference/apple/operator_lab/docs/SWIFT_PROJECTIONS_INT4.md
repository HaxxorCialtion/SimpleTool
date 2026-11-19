# Swift int4 投影 kernel：2026-10-08

本项目 streaming QMV、gate/up/SwiGLU 融合 kernel 已接入完整 Swift 2B/4B 模型。权重为 affine int4/group64，浮点计算及 GDN 状态仍为 FP32，没有新增 FP16 语言模型推理。数值验证通过；时延差低于既有 10% 噪声门槛，三种新配置保留显式候选，默认 `baseline` 不变。完整 iPhone 产品与 98% 效率目标尚未验收。

## 实现

- Streaming QMV 每次保留四个输入值，降低寄存器工作集，同时复用多分支的 packed 权重行。选择 M2..4、K%512==0、N>=256；M4 的 N>=65536 或 K>4096 保留 row-reuse，M1 不变。
- gate/up 投影和 SwiGLU 合为一个 custom dispatch，直接生成激活结果，避免两个投影结果及中间调度。权重仍分开保存，不拼接、不展开 dense。4B N9216/K2560 的 M2/M4 用 rows1/threads256，M3 用 rows4/threads128；其他形状用 rows4/threads128。M1、大 prefill 保留原路径。
- kernel 对象持久复用，`ProjectionPolicy` 不可变。benchmark 四策略引用同一批 immutable MLXArray 权重，运行时断言对象一致。`stream/fused/both` 可搭配 dynamic/compiled；短 prefill 尾 chunk 同样可能进入新 kernel。

源码：`SimpleToolSwift/Sources/SimpleToolSwift/{Model,BenchDecode,ProjectionRegression}.swift` 和 `Kernels/{stream_qmv,fused_swiglu}.metal.txt`。body 与已验证 Python 源码逐字一致，来源及 SHA 在 `Kernels/provenance.json`。接口依据固定版本 [MLX Swift custom kernel 源码](https://github.com/ml-explore/mlx-swift/blob/0.31.1/Source/MLXFast/MLXFastKernel.swift)，未修改 vendor 或共享环境。

## 验证

| 检查 | 结果 |
| --- | --- |
| 融合算子 | 288/288 对保存的独立 Python 两次投影加 SwiGLU gold 逐位相等 |
| Streaming 算子 | 216/216 对每行独立 compact QMV gold 逐位相等 |
| 真实 4B 权重 | 上述含 144 项真实第一层参数案例 |
| 原始 fixtures | 64 个 raw 文件 SHA 与原始记录一致 |
| 完整模型 | 两模型 × 三策略 × 两执行线 × 五条真实指令，60/60 case 通过 |
| 通道输出 | 每模型/策略/执行组合 serial、B4 各20/20 token与文本对 matching int4 Python serial gold 相等 |
| 前缀边界 | 4B 259（256+3）、501（256+245）token，both 的 dynamic/compiled 单路与 B4 均通过 |
| 构建 | Mac Release CLI、iOS arm64 库通过；未在手机执行 |

指令使用完整 tokenizer 标签、EOS/关闭标签独立停止、batch 缩小，上限64 token/通道；边界包上限32。对拍证明同权重实现一致，不代表量化质量或全部指令闭合。command 仅作数据比较，没有执行。

首次 regression CLI 编译因 template 类型错误失败，改用 `KernelTemplateArg` 后通过，失败日志保留。全部模型对拍与 benchmark 使用最终同一个 Release binary。

## 完整模型时延

M4 Pro，同进程同权重、prefill一次；四策略各single/B4共八个runner。每条路径完整预热一次，轮换五轮，固定96 token/头。计时含标签推进和decode，不含prefill/reset/首次预热。每轮完整token都与baseline相等。效率为各策略自己的 `T_single/T_B4`，吞吐加速比为四倍。

| 模型/进程 | baseline B4 | stream B4 | fused B4 | both B4 | both 时延降低 | both 效率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B 第一批 | 1.994 s | 1.948 s | 1.948 s | 1.925 s | 3.50% | 66.80% |
| 4B 独立复测 | 1.994 s | 1.941 s | 1.942 s | 1.914 s | 4.01% | 66.92% |
| 2B 一批 | 1.019 s | 1.010 s | 1.012 s | 1.010 s | 0.89% | 65.19% |

两批4B均观测到小差异，但低于噪声门槛，不宣布可靠新性能收益。2B未独立复测。外部负载和热状态未受控，M4 Pro调参不能直接当作iPhone最优配置。八个runner同时保有状态，MLX峰值不是部署RAM；未测实际DRAM流量，不能断言距硬件理论上限多少。

## 证据与复现

汇总 `reports/swift_projection_summary_20261008.json` 保存源码、资源、Release binary、iOS module SHA。原始结果：

- `reports/swift_projection_tensor_regression_20261008.json`、`swift_projection_fixture_audit_20261008.json`。
- `reports/swift_projection_{dynamic,compiled}_suite_20261008.json`；每个case另有结果与日志。
- `reports/swift_projection_{boundary,long}_{dynamic,compiled}_20261008.json`。
- `reports/swift_projection_4b_ab_20261008.json`、`swift_projection_4b_ab_repeat_20261008.json`、`swift_projection_2b_ab_20261008.json`。
- `reports/swift_projection_regression_build_fixed_20261008.log`、`swift_projection_ios_build_20261008.log`。

项目根目录运行：

```sh
ST_SWIFT_PROJECTIONS=both ST_SWIFT_EXECUTION=compiled artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI artifacts/4b_int4_compact reports/swift_4b_natural_packet_20261008.json reports/swift_projection_result.json
ST_SWIFT_BENCHMARK=projections artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI artifacts/4b_int4_compact reports/swift_4b_packet_20261008.json reports/swift_projection_benchmark.json
artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI --projection-regression artifacts/native_swiglu_fixture
/Users/cialtionshi/mlx_env/bin/python check_swift_projections.py --execution compiled
/Users/cialtionshi/mlx_env/bin/python check_swift_projections.py --execution dynamic
/Users/cialtionshi/mlx_env/bin/python make_swift_projection_report.py
```

最后一条只读取保存结果、校验SHA及重建汇总。SDK使用 `Int4Model(directory:projectionPolicy:)`。

前一阶段固定suffix KV与整步编译收益仍见 [SWIFT_STATIC_INT4.md](SWIFT_STATIC_INT4.md)。原生prompt编码、Swift图像与完整多轮、owned GDN原地更新接入、iPhone实机/功耗、量化质量及llama.cpp一致性门槛仍待完成。本轮没有改变这些验收状态。
