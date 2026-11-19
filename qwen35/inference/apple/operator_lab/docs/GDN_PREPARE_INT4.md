# GDN 卷积、Q/K 归一化与历史更新融合

2026-10-08。2B/4B继续原compact int4/group64，FP32激活、GDN和归一化；metadata仍是已有无损half存储。未重新量化、未展开语言矩阵、未新做FP16推理，默认未变。

本阶段将每步GDN的depthwise卷积、SiLU、Q/K RMS归一化/缩放及conv history更新合成一个Metal kernel。直接产生Q/K/V与下一份history，不建立完整conv_input或conv_out中间张量。原不可变seed仍B1只读，第一次分叉写B份独立GDN matrix；之后按原owned packed recurrence更新。完整int4投影、attention和采样路径未改。

## kernel与数值约束

- 支持B1–4、S1、conv kernel4、Dk/Dv128、FP32，无mask/lengths/sharding/training；其他形状回原SharedSeedGDN路径。每个SIMD负责一个128维head，lane连续四元素。value部分只算卷积/SiLU及历史。
- 卷积保持三个历史项后当前项的FP32累加次序；SiLU沿用现有gdn_conv精确exp与符号分支。Q/K平方和按每lane四项、simd_sum，再precise rsqrt；eps是1e−6/128，q乘1/128、k乘FP32的128**−0.5。
- RMS设计参考固定本地MLX Swift core源的RMS_N_READS=4；它不代表当前Python wheel源码。实际Python行为以160结构配置及真实缓存逐位对拍核验。没有全局fast-math或FP contraction设置变化。
- 输入history/weight/qkv只读，输出history独立；没有对borrowed MLX buffer做原地写。没有改变公共KV物理存储、标记序列、独立停止/batch缩小或四通道历史契约。
- 输出归一化/SiLU、GDN state数学仍原实现；`gates`候选额外使用已存在的gate_pair，合并decay/beta计算。三种模式的参数对象全部共享，原base只prefill一次。

候选：`conv`为旧conv_state+原Q/K norm的控制；`prepare`为新融合准备；`gates`为新融合准备+已有gate_pair。没有串接上一阶段的native projection policy，以隔离本阶段收益。

## 验证

| 检查 | 证据 |
|---|---|
| 合成operator | 160配置逐位对MLX实际conv/SiLU/normalize_qk/history；五head配置、B1–4、零/小/正常/较大范围、signed zero、strided，输入SHA不变 |
| 完整缓存 | 一个真实prefix/模型，实际所有marker位置+八个baseline贪心步，B1–4×三候选，共240次step/候选检查；每层functional conv/GDN/KV所有array逐位同baseline、prediction一致、source SHA不变 |
| 完整真实请求 | 两模型五条既有输入，B1–4、chunk1/8、三候选，共240配置/780通道gold比较，全部通过；parameter对象共享 |
| 固定长度性能 | baseline/conv/prepare/gates ×single/B4，96token/头忽略stop，首次+五轮轮换，三进程共144运行，全部token同baseline |
| 自然请求性能 | 五输入、四runner、首次+五轮，三进程共360个B4请求，全部gold/source SHA通过 |
| iOS | 20个独立FP32shader编译；不是完整Swift候选或手机执行 |

配置数不是独立问句数。缓存逐位只覆盖一个prefix加若干步，不构成任意输入等价证明；五条gold不构成量化质量或真实工具上下文验收。模型command仅保存为数据。

## 完整性能

固定每头96token，B4中位秒，包含host/eval、不含prefill：

| 进程 | baseline | conv控制 | prepare | prepare+gates |
|---|---:|---:|---:|---:|
| 4b | 1.9921 | 1.9796 | 1.9429 | 1.9457 |
| 4b_repeat | 1.9938 | 1.9814 | 1.9392 | 1.9425 |
| 2b | 1.0263 | 1.0150 | 0.9831 | 0.9836 |

相对baseline的B4时延减少：

- 4b: conv 0.63%, prepare 2.47%, gates 2.33%。
- 4b_repeat: conv 0.62%, prepare 2.74%, gates 2.57%。
- 2b: conv 1.11%, prepare 4.21%, gates 4.16%。

自然请求中位秒，包含fork、完整标签、采样、stop和batch缩小、不含prefill：

| 进程 / 输入 | baseline | conv | prepare | gates |
|---|---:|---:|---:|---:|
| 4b / 0 | 0.2808 | 0.2777 | 0.2722 | 0.2735 |
| 4b / 1 | 0.3257 | 0.3215 | 0.3168 | 0.3160 |
| 4b / 2 | 0.3434 | 0.3391 | 0.3344 | 0.3332 |
| 4b / 3 | 0.2400 | 0.2361 | 0.2325 | 0.2327 |
| 4b / 4 | 0.2412 | 0.2388 | 0.2339 | 0.2349 |
| 4b_repeat / 0 | 0.2821 | 0.2783 | 0.2730 | 0.2722 |
| 4b_repeat / 1 | 0.3265 | 0.3222 | 0.3172 | 0.3175 |
| 4b_repeat / 2 | 0.3423 | 0.3388 | 0.3334 | 0.3337 |
| 4b_repeat / 3 | 0.2403 | 0.2374 | 0.2325 | 0.2308 |
| 4b_repeat / 4 | 0.2409 | 0.2389 | 0.2339 | 0.2350 |
| 2b / 0 | 0.5797 | 0.5738 | 0.5529 | 0.5553 |
| 2b / 1 | 0.5316 | 0.5250 | 0.5088 | 0.5096 |
| 2b / 2 | 0.3605 | 0.3586 | 0.3462 | 0.3525 |
| 2b / 3 | 0.1216 | 0.1206 | 0.1160 | 0.1164 |
| 2b / 4 | 0.5356 | 0.5295 | 0.5107 | 0.5160 |

首轮按顺序预热，不是公平冷GPU/磁盘/编译比较；首次路径另存原JSON，外部负载未锁定。single/B4所有数据均保留，不用single变慢推导效率改善。各runner同时保留缓存，buffer峰值不能代表部署RAM；本阶段未新增physical footprint/功耗/DRAM或kernel硬件计数。

源码将prepare合为一个custom Metal调用，但未捕获完整新pipeline的实际GPU dispatch数；不能按Python算子数量声称“若干dispatch变成一个”。中间张量不再显式建立也不等于已测整体内存下降。

默认仍不变。更深融合通过正确性，但完整速度仍应按同一10%门槛判断，不能用结构变化直接称性能突破。后续需要把候选与已测int4调度组合并在完整Swift/iPhone上验收；原生prompt、图像/完整多轮、量化质量、严格llama.cpp对拍和98%目标仍未完成。

## 复现

```bash
cd /Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine
/Users/cialtionshi/mlx_env/bin/python check_gdn_prepare.py
/Users/cialtionshi/mlx_env/bin/python prepare_gdn_validation.py
/Users/cialtionshi/mlx_env/bin/python run_prepare_validation.py
/Users/cialtionshi/mlx_env/bin/python build_prepare_ios.py
/Users/cialtionshi/mlx_env/bin/python run_prepare_performance.py
/Users/cialtionshi/mlx_env/bin/python make_prepare_report.py
```

原始报告：`reports/gdn_prepare_*20261008*`与`reports/prepare_*20261008*`。候选入口`install_prepare_gdn(model,mode="prepare")`＋`CompiledStep(model)`，要求原GDN或SharedSeedGDN；不是默认部署入口。汇总验证source/checkpoint/ios/zip SHA；snapshot保留所有源码、原始证据及iOS库，旧阶段全部保留。
