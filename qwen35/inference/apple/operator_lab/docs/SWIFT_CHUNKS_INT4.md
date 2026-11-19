# Swift int4 多步 GPU 调度实验：2026-10-08

2B/4B 的完整 Swift compiled 路径现在可一次推进 2/4/8 个自回归 token，中间预测及缓存转换留在 GPU 图中，块边界再读回 token、检查独立停止和缩小 batch。正确性验收通过，但四头性能未达噪声门槛：4B 两批 8 步略慢，2B 一批仅约2.2%差异。默认仍为单步；没有新增 FP16 语言模型推理。权重 int4/group64，浮点计算/GDN 状态 FP32。

## 调度与状态语义

`CompiledDecode.advancePending` 把原始完整单步函数展开为固定小块，按块长度复用编译函数。返回 `[count,B]` 预测和最新状态；最后一个预测尚未喂入模型。块内只推进 count−1 次，保持原有 terminal token 不必写进缓存的语义。used 动态更新，位置和 suffix mask 随每一步推进，prefix KV 和模型权重由同一批不可变 MLXArray 引用。

调用方逐个检查块内 token 的 EOS、完整关闭标签及解码文本关闭标记。一旦通道结束，丢弃该通道剩余预测；其已经多算的状态在块边界舍弃。幸存通道保持正确的最终状态，过滤 conv/GDN/suffix KV 后继续，公共 prefix 不参与过滤。此路径不会将终止通道的超前状态当作历史发布，也没有新增 Swift 多轮接口。

最后预测通过下一块的首次 forward 消费，不单独插入第二次 host evaluation。marker 推进和 batch 过滤仍有自己的 evaluation；统计 `decode_evaluation_blocks` 仅为 decode 块数，不是全部 GPU dispatch 或全部同步次数。

多步会让首批输出等待块完成，也可能为已经结束的通道多算最多7步。调用方需要逐 token 低延迟时，应保持单步；目前性能证据也支持保持此默认。

源码在 `SimpleToolSwift/Sources/SimpleToolSwift/{StaticDecode,Validation,BenchDecode,ChunkRegression}.swift` 及 CLI。实现沿用本项目已有 Python `advance_pending` 的缓存语义，未修改 Zcode、vendor 或共享环境。

## 验收

| 证据 | 范围与结果 |
| --- | --- |
| 完整缓存逐位对拍 | 2B/4B × baseline/both × B1..4 × chunk1/2/4/8，共64项通过 |
| 状态内容 | 17个输出token，最后预测pending；完整 conv、GDN矩阵、suffix K/V字节与单步相等 |
| 容量边界 | KV容量精确设为 marker长度+16，最终 used==capacity，最后槽位被使用 |
| 共享前缀 | 每组运行前后所有source cache数据SHA不变 |
| 自然停止 | 两模型 × baseline/both × 四chunk × 五真实指令，80/80 case通过 |
| 不整除的尾部 | 额外16/16 case通过，2B上限31、4B上限11，覆盖3/7长度尾块 |
| 输出语义 | serial/B4的token、解码文本对matching int4 Python serial gold相等，停止原因与chunk1相同 |
| Batch缩小 | 自然停止中实际出现B4/B3/B2/B1；80条case合计丢弃512个B4关闭后预测token |
| 构建 | Mac Release CLI、iOS arm64库编译通过；没有手机实机运行 |

自然停止最多64 token/通道。尾部gold由已验证的Python serial gold截断，未重做模型推理；使用checkpoint tokenizer重新解码，并先验证其对原始完整gold文本相等。来源文件SHA写入新packet，原gold未修改。command始终是数据，未执行。

这证明有限测试范围的同权重数值/停止一致性，不证明量化质量、闭合率、图像或无限多轮正确性。owned GDN原地kernel仍未接入此完整模型。

## 实际性能

M4 Pro、同模型同权重、prefill一次，baseline投影策略。每个chunk配置各single/B4共八个runner，完整预热一次后轮换五轮，固定96 token/头，忽略停止。计时含完整marker推进和decode，排除prefill、reset、首次预热。每轮所有token与chunk1一致。

| 模型/进程 | chunk1 B4 | chunk2 B4 | chunk4 B4 | chunk8 B4 | chunk8时延降低 | chunk8效率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B第一批 | 2.005 s | 2.022 s | 2.054 s | 2.031 s | −1.28% | 63.18% |
| 4B独立复测 | 2.005 s | 1.995 s | 2.043 s | 2.030 s | −1.24% | 63.75% |
| 2B一批 | 1.027 s | 1.028 s | 1.013 s | 1.004 s | 2.18% | 62.82% |

效率为每个配置自己的 `T_single/T_B4`。单头chunk8差异约2.4–2.9%（4B）及6.1%（2B），同样低于10%噪声门槛。2B未独立复测。first-run时延另有记录，但各配置先后共享kernel缓存，不能作为独立冷启动比较。

减少host检查并未得到可信四头收益，因此保持默认单步。结果只支持“此实现与此条件下，减少检查次数不是有效提速手段”，不能独立判定具体的DRAM瓶颈、纯GPU时间或硬件理论上限。外部负载与热状态未受控。

baseline五条真实指令合计B4检查块数：2B从240降至31，4B从82降至12。完整验证进程的MLX峰值最大值，2B约1.252–1.253GB、4B约2.823GB，chunk间差异很小；prefill可能覆盖decode峰值，不能声称减少部署内存。性能benchmark的八runner峰值也不代表单实例RAM。

## 保存证据与复现

汇总 `reports/swift_chunks_summary_20261008.json` 检查所有结果、源码/资源/二进制SHA、构建日志与停止原因；保存 `artifacts/swift_chunks_snapshot_20261008/SimpleToolSwiftCLI` 和 `sources.zip`，避免后续覆盖当前binary导致旧结果无法对应版本。

原始证据：

- `reports/swift_chunks_state_suite_20261008.json`，内含四个完整模型报告，共64项。
- `reports/swift_chunks_natural_suite_20261008.json`、`swift_chunks_tail_suite_20261008.json`，每条另有结果与日志。
- `reports/swift_chunks_{4b,4b_repeat,2b}_ab_20261008.json` 和对应日志。
- `reports/swift_chunks_{state,ios}_build_20261008.log`。

项目根目录：

```sh
ST_SWIFT_EXECUTION=compiled ST_SWIFT_DECODE_CHUNK=8 artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI artifacts/4b_int4_compact reports/swift_4b_natural_packet_20261008.json reports/swift_chunks_result.json
ST_SWIFT_BENCHMARK=chunks artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI artifacts/4b_int4_compact reports/swift_4b_packet_20261008.json reports/swift_chunks_benchmark.json
/Users/cialtionshi/mlx_env/bin/python check_swift_chunks.py --kind state
/Users/cialtionshi/mlx_env/bin/python check_swift_chunks.py --kind natural
/Users/cialtionshi/mlx_env/bin/python check_swift_chunk_tails.py
/Users/cialtionshi/mlx_env/bin/python make_swift_chunks_report.py
```

最后一条读取已保存证据并保存版本快照，不启动GPU推理。SDK通过 `validatePacket(_:execution:decodeChunkSize:)` 显式选择。snapshot仍依赖原项目固定版本MLX及资源目录，不能直接当成iPhone安装包。

先前投影优化见 [SWIFT_PROJECTIONS_INT4.md](SWIFT_PROJECTIONS_INT4.md)。原生prompt编码、Swift图像/完整多轮、owned状态更新接入、手机实机/功耗、量化质量、llama.cpp一致性和98%目标均未验收。下一步的性能探索应回到具体GPU算子和状态读写；本轮不将减少同步次数包装成性能突破。
