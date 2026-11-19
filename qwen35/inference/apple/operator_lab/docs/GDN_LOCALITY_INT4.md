# int4 底层实验：共享 GDN seed 的 GPU 调度局部性

2026-10-08。继 `FIXED_HISTORY_INT4.md`，同一套2B/4B compact int4权重，FP32计算与状态，未新做FP16语言推理。默认保持原配置。本阶段新增两种工作排列并接入完整Python模型；整机差异低于10%门槛，保留实验，不采用为默认，未移植完整Swift候选。

## 调度与所有权

GDN的逻辑索引仍为 `n=b*Hv+hv`，每个SIMD32用四个lane处理同一value行的128个key元素，group为 `(32,2,1)`。只改变共享base（SB=1）时线程位置到逻辑索引的映射：

- `head`：z中的branch为内层，同head各分支更邻近。
- `value`：branch进入y，邻近SIMD组处理同一head、同一value行的分支。共享grid为 `(32,Dv/8*B,Hv)`。

owned base使用原排列；B1映射退化为原布局。写回、history复制、逐步FP32舍入、归约顺序均不变；未向输入写入、未降低精度。借出的共同prefix/base保持只读，独立分支history按停止后的存活索引重排。完整标签来自实际tokenizer；4B标签是一token，2B标签是三token。

这不是硬件cache驻留保证：GPU线程块执行顺序及L2命中率没有计数器验证。排列旨在让相同读取更邻近；base第一次write后变为owned，公共状态读取的机会随之减少。

## 正确性

| 检查 | 覆盖及结果 |
|---|---|
| 两排列×普通/融合kernel | 12640项输出与重建matrix逐位相同；五组head/value形状、B1–4、间隔2/4/8/16、shared/owned、NaN未使用槽位、gate边界、分支重排；seed SHA不变 |
| 完整Python int4 | 两模型×两排列×四间隔×1/8步×五真实指令：160配置、1280通道token对拍，serial/B4一致，source SHA不变 |
| 完整缓存 | 两模型×两排列各48配置：192组卷积、重建GDN及完整KV逐位一致；19token、精确suffix容量与最后pending位置，source SHA不变 |
| 原生Metal | 24配置×三排列，6个kernel输出逐字节对MLX gold；每个command后检查，全部input只读验证通过 |
| iOS shader | 240变体以 `-fno-fast-math` 编译通过；不是完整Swift候选或手机执行 |

真实指令仍是每模型五条已有输入，配置数不是独立问句数。当前4B这些输入的command输出很短，2B部分通道达到上限；没有新增工具协议或量化质量验收。模型命令未执行。

## 算子计时

Python孤立计时包含提交/同步，共84配置、五轮轮换、每轮40次。为了区分GPU执行与Python提交，另编译原生Metal runner：同一层输入重复160次dispatch/command、七轮轮换，以command GPU时间除dispatch数。每次结束检查六个输出和输入不变。

原生4B形状、B4、shared base的GPU µs/dispatch：

| interval / phase / operation | baseline | head | value |
|---|---:|---:|---:|
| 4 / 0 / defer | 33.10 | 33.45 | 35.04 |
| 4 / 2 / defer | 49.64 | 48.29 | 47.42 |
| 4 / 3 / write | 71.96 | 66.51 | 66.54 |
| 16 / 0 / defer | 55.18 | 52.63 | 52.04 |
| 16 / 14 / defer | 148.34 | 135.56 | 132.60 |
| 16 / 15 / write | 154.52 | 145.23 | 144.16 |

14-history步骤中value排列约减少10.6%，但无history的4间隔步骤略慢。owned状态控制基本相同，2B形状变化很小。原生实验重复读取同一层buffer；原runner的“cache-hot”标签只描述重复输入，cache驻留未测。它没有完整层扫描、模型投影、DRAM计数或iPhone数据，不能直接外推模型吞吐。

## 完整模型与实际请求

同一权重与共同prefill，十四个runner，96token/头、完整标签、忽略stop，五轮以3个位置轮换次序。两次独立4B进程及一次2B进程，逐轮token均等于dense基线。首轮按顺序预热，不是公平冷GPU比较；外部负载未锁定。

B4中位秒数：

| 进程 | dense1 | fused16原排列 | head16 | value16 |
|---|---:|---:|---:|---:|
| 4b | 1.9957 | 2.0280 | 2.0246 | 2.0207 |
| 4b_repeat | 2.0279 | 2.0644 | 2.0761 | 2.0370 |
| 2b | 1.0262 | 1.0675 | 1.0671 | 1.0672 |

4间隔与全部single时延见 `reports/locality_summary_20261008.json` / 原始benchmark。没有通过放慢单路制造效率收益；本阶段未报告新的效率达标。相对同间隔fused基线的完整时延差异均低于10%门槛。

另对每模型五输入做自然B4请求，单个输入prefill一次、四runner复用同一个只读source；计入fork、标签、预测读取、独立stop及batch缩小。五轮轮换，每次原始token对gold，前后source SHA一致。Python请求窗口包含host开销，不含prefill。以下选取输入，完整五输入/全部样本在原始报告：

| 进程 / 输入 | dense1 | fused16原排列 | head16 | value16 |
|---|---:|---:|---:|---:|
| 4b / case0 | 0.2834 | 0.2796 | 0.2775 | 0.2767 |
| 4b / case2 | 0.3566 | 0.3538 | 0.3543 | 0.3631 |
| 4b_repeat / case0 | 0.2855 | 0.2809 | 0.2773 | 0.2762 |
| 4b_repeat / case2 | 0.3451 | 0.3443 | 0.3421 | 0.3417 |
| 2b / case0 | 0.5901 | 0.6159 | 0.6110 | 0.6083 |
| 2b / case4 | 0.5383 | 0.5624 | 0.5579 | 0.5563 |

4b 相对同间隔原排列的自然请求时延减少范围为 -2.61% 到 1.20%；负数表示更慢。
4b_repeat 相对同间隔原排列的自然请求时延减少范围为 0.57% 到 1.67%；负数表示更慢。
2b 相对同间隔原排列的自然请求时延减少范围为 -0.71% 到 1.24%；负数表示更慢。

这些差异没有达到可信新提速门槛；不接入默认、HTTP、图像、多轮提交或完整Swift候选。benchmark中的peak包含多个runner，不能当作部署RAM；本阶段没有新增物理内存、功耗或手机测量。

## 复现和快照

```bash
cd /Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine
/Users/cialtionshi/mlx_env/bin/python run_locality_initial.py
/Users/cialtionshi/mlx_env/bin/python run_locality_model.py --phase validate
/Users/cialtionshi/mlx_env/bin/python run_locality_performance.py
/Users/cialtionshi/mlx_env/bin/python build_locality_ios.py
```

Python候选使用 `CompiledFixedHistoryStep(model,16,fused=True,state_type=HeadLocalityState)` 或 `ValueLocalityState`；先调用 `prepare_deferred_model(model)`。默认类仍为 `FixedHistoryState`。旧源码在 `artifacts/gdn_locality_before_20261008/`，新源码和native binary在 `artifacts/locality_snapshot_20261008/`；报告含checkpoint、source、binary、archive及fixture哈希。fixture生成器可重建原生输入/gold，源码zip不含大体积.bin。

手机、完整原生prompt编码、图像/多轮Swift路径、量化质量、llama.cpp严格对拍和98%门槛仍未完成。此次证据说明仅重排初始shared base读取难以获得整体突破；接下来应处理更长阶段的状态表示/重放成本或更大的int4投影成本。当前五gold的对拍也不能代替真实工具上下文验收。
