# Swift 完整 int4 数值路径：2026-10-08

后续固定缓存/整步编译及Release热态比较已追加在 [SWIFT_STATIC_INT4.md](SWIFT_STATIC_INT4.md)。下文保留首次数值路径验收，末尾相应TODO已部分推进，当前状态以后续报告为准。

本轮将独立实现从 Python 和孤立 Metal 算子推进到可加载实际 2B/4B int4 权重的 Swift 完整语言模型库。Mac 实际生成已对拍，iOS 库已编译；手机运行、完整产品部署和速度收益尚未验收。未新增 FP16 语言模型推理。

## 已实现

- 直接读取 `artifacts/{2b,4b}_int4_compact` safetensors，affine int4/group64 保持打包，tied embedding 与 vocabulary projection 共用一套张量。
- scale/bias 使用此前验证无损的窄存储；所有浮点计算、激活及 GDN 状态保持 FP32。配置中的历史 `dtype: float16` 不决定运行精度。
- 完整 dense Qwen3.5：embedding、GDN、卷积、full attention、partial RoPE、RMSNorm、MLP、tied vocabulary projection。
- 共同前缀只 prefill 一次；四头 full-attention 读取单份 prefix K/V，后缀为 batch。GDN 首步读取只读 B1 seed，输出独立 B4 矩阵，后续状态各自递推。
- 2B 完整三 token 标签、4B 完整单 token 标签按实际 tokenizer 编码输入，每个标签位置一次合批前向。
- native ByteLevel BPE decoder 从 checkpoint vocabulary 读取，检测 EOS、完整关闭 token 序列及解码字符串关闭标签，兼容跨 token 的 BPE 边界。停止后过滤分支状态及 suffix KV，batch 逐步缩小。
- int4 单路直接读取 compact QMV，小 batch 多行复用 Metal kernel；编译 SiLU、SwiGLU 与 GDN gate 表达式，复用 kernel 对象。
- safetensors 逐层 trace、独立 RoPE 回归及真实输入对拍脚本，失败记录保留。

## 确认并绕过的固定版本 RoPE 问题

首次 2B 四头 command 在第 8 token 分歧，单路一致。逐层定位表明：前 3 个 GDN 层差异约浮点舍入量级，第 4 个 full-attention 开始明显不同。分支 0 的 q/k 正确，分支 1～3 的旋转部分与缺少 offset 旋转一致；非旋转部分约 4e-6 差异。

固定的 MLX core 在 contiguous、T=1、offset.size=1 时选 `rope_single`，dispatch 为 `(dims/2,N,1)`，未覆盖 B*N heads。独立回归在 B2/B3/B4、offset20/37 复现 max-abs 约 2.06/2.04。单纯将 Int offset 换成 rank-0 MLXArray 未解决，失败记录保留。

独立引擎在 B>1 显式传入 B 个相同 offset，选择一般 batched kernel，未修改共享环境或 vendor 源码。修复回归 B1..4 × offset0/20/37 共 12/12，相对于 B1 reference 的 max-abs 全部为 0。实际 2B 首步 attention q 差异从 5.08 降至 7.15e-6，attention 输出从 0.0654 降至 2.26e-6，完整 token 对拍恢复。

这是固定依赖版本的发现；不声称最新 MLX 存在相同问题。首轮 4B 12-token 样例在该错误下也曾偶然 token 相同，不能仅凭短 token 样例证明算子正确。

## 验收证据

`reports/swift_real_suite_20261008.json`：沿用此前 5 条真实指令，重新生成同 int4 权重的 Python serial gold，然后执行 Swift serial 和 Swift B4。每条最多 64 token，自然停止。

| 模型 | Swift serial vs Python serial | Swift B4 vs Python serial | ByteLevel 解码文本 |
|---|---:|---:|---:|
| 2B int4 | 20/20 通道相等 | 20/20 通道相等 | 两路径各20/20 |
| 4B int4 | 20/20 通道相等 | 20/20 通道相等 | 两路径各20/20 |

Python B4 也与本次 serial gold 全部一致。多条输入出现 B4→B3→B2→B1，停止包含 EOS 和 channel close；达到 64 token 上限的分支仍记 max_tokens，不能据此声称闭合率验收通过。所有 command 都只作为输出数据。

- RoPE 回归：`reports/swift_rope_regression_20261008.json`。
- 失败及修复诊断：`swift_2b_first_result`、`swift_2b_arithmetic_result`、`swift_2b_rope_fixed_result`、`swift_2b_rope_vector_result`，均在 reports 下，日期后缀为 `_20261008.json`。
- 完整 Mac 构建：`reports/swift_final_mac_build_20261008.log`。
- 完整 iOS arm64 库构建：`reports/swift_final_ios_build_20261008.log`，`CODE_SIGNING_ALLOWED=NO`。
- 设备记录：`artifacts/swift_device_inventory_20261008.json`；iPhone17ProMax/iPhone18,2 unavailable，未安装或运行。
- 最终依赖/源码/二进制摘要：`reports/swift_summary_20261008.json`。

以上是小样本数值与编译门槛，独立于量化质量、phone RAM、持续吞吐和产品验收。

## 依赖和归属

本机 Xcode26.2 / Swift6.2.3。官方 MLX Swift 0.32.3 和 0.31.6 的 manifest 要求 Swift6.3，保留原始兼容失败记录，不升级共享工具链。采用可编译的官方 0.31.1：

- MLX Swift `399f23159f54699f918965e41897e1178d2352fe`。
- MLX core `ce45c52505c8158ea48d2a54e8caae05efd86bfe`。
- MLX-C `0726ca922fc902c4c61ef9c27d94132be418e945`。
- MLX Swift LM `0ebbdbddbc8f5b9688a5368e9d2a1cb51540d6af` 仅作源码参考，无 runtime 依赖。

依赖均克隆在本项目 `artifacts/swift_vendor/`，未改源码。Swift LM 参考实现的 GDN q/k 归一化与当前 Python 契约存在差异，独立实现采用已验证 Python 的 epsilon=1e-6/128、q scale=1/128。不能未经对拍直接替换。

原有 packed GDN 算术来自 Apple MIT，不是本轮新设计。自有贡献为完整模型接线、共享 seed 合约、int4 dispatch 和状态管理。内核原始来源及 SHA 在 Swift resource `Kernels/provenance.json`，许可证见 `SimpleToolSwift/THIRD_PARTY.md`。

## 运行

在项目目录执行 Python gold 导出，然后在 `SimpleToolSwift` 目录用 Xcode 编译。binary 相对于项目目录：

```sh
/Users/cialtionshi/mlx_env/bin/python make_swift_packet.py --model artifacts/4b_int4_compact --output reports/swift_packet.json --tokens 64 --natural-stop
cd SimpleToolSwift
xcodebuild -scheme SimpleToolSwiftCLI -destination 'platform=macOS' -derivedDataPath ../artifacts/swift_build -jobs 4 build
cd ..
artifacts/swift_build/Build/Products/Debug/SimpleToolSwiftCLI artifacts/4b_int4_compact reports/swift_packet.json reports/swift_result.json
artifacts/swift_build/Build/Products/Debug/SimpleToolSwiftCLI --rope-regression
/Users/cialtionshi/mlx_env/bin/python check_swift_suite.py
```

CLI 对 token 不一致返回2。调试设置 `ST_SWIFT_TRACE=artifacts/trace.safetensors` 记录层输出，trace 持有中间张量，不能用于 RAM/性能验收。普通运行默认关闭。

## 尚未接入的关键工作

1. Prompt/chat template 与完整标签编码目前由 checkpoint Python tokenizer 导出 packet；Swift 仅 native decode，不是完整原生 tokenizer。
2. Swift 未接图像 encoder、图像 embedding/M-RoPE、完整多轮历史增量缓存和应用接口；Python 已有功能不能当作 Swift 已验收。
3. 原生 GDN owned-buffer 原地更新、最新 streaming QMV 与 gate/up+SwiGLU 单 dispatch 尚未接到 Swift 完整模型。当前仍为函数式 MLX 状态，不能声称已省去完整模型双缓冲。
4. KV 后缀当前按步 concatenate，尚未使用固定 capacity ring/buffer；整步 compiled graph、实际 kernel 预热、资源复用和 release 性能待做。
5. 当前构建为 Debug，短样例含 JIT 与 batch 缩小后的首次特化。运行时延、MLX buffer peak 只保留原始诊断，不用于声称比 Python 快或估算 iPhone 上限。
6. iOS 编译不等于实机完整生成，需实机 RAM/压缩内存/温控/功耗/持续生成对拍；98% 效率尚未达到。llama.cpp 的既有独立门槛仍未关闭。
