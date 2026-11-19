# int4 底层优化与 iPhone 迁移入口

更新：2026-10-08。4B 部署固定 int4，2B 本轮也使用 int4；2B int8 尚未测。没有新增 FP16 模型推理实验。原始 *_fp16 目录仅用于离线量化输入。

本轮可用候选：共享权重 int4 kernel + 可选整步编译 + 多投影连续输出。固定长度改善约 8–9%，低于此前定义的 ±10% 噪声带，尚不能宣称稳定性能胜出。默认仍保持此前验证的 eager int4 路径；新候选通过参数启用。

## 同进程性能对照

每头 96 token，屏蔽 EOS，强制四路保持 B=4；完全相同公共前缀 KV 已热。A/B 共用同一份量化参数存储，交替三轮中位。冷加载、首次编译不计入热时延；其他进程负载未锁定。

|模型|路径|四头请求墙钟|聚合 decode tok/s|相对本轮基线|匹配单头/四头墙钟效率|
|---|---|---:|---:|---:|---:|
|2B|原 int4 kernel|1.651s|237.6|1.000×|63.2%|
|2B|整步编译|1.541s|254.5|1.072×|65.4%|
|2B|多输出投影|1.634s|239.8|1.011×|64.5%|
|2B|编译 + 多输出|1.513s|259.5|1.092×|66.3%|
|4B|原 int4 kernel|3.017s|127.3|1.000×|59.7%|
|4B|整步编译|2.809s|136.8|1.074×|62.0%|
|4B|多输出投影|2.931s|131.1|1.029×|60.8%|
|4B|编译 + 多输出|2.799s|137.2|1.078×|61.8%|

2B 的多输出 kernel 在编译之外额外改善约 2%；4B 额外改善约 0.3%。投影合并自身没有解决主要瓶颈。整步编译有冷编译成本，前缀长度、容量、B=4→3→2→1 都可能触发新图；投影 pipeline 预热并不消除这些成本。

共享环境的 GDN 包含已存在的 packed 实现。本轮没有修改共享环境；每份 A/B 报告记录其文件 SHA-256，开始与结束相同。不能把跨时段绝对速度变化归因于本项目的新 kernel。

4B 短生成另做五轮交替：每头 12 token，原 kernel 378ms，编译 + 多输出 346ms，观测提速 1.090×；仍小于 10% 门槛。原始数据见 reports/optimize_4b_split_12_20261008.json。

## 正确性和部署产物

|对象|证据|结果|
|---|---|---|
|2B 持久化 int4|artifacts/2b_int4/int4_receipt.json|1.178GB，694 张量重载逐位一致：True|
|2B 自然停止 / 编译多输出|reports/2b_int4_split_compiled_cases_20261008.json|stock MLX 同 int4 权重单路对拍 20/20，包含 batch 缩小|
|4B 持久化 int4|artifacts/4b_int4/int4_receipt.json|2.632GB，924 张量重载逐位一致：True|
|4B 自然停止 / 编译多输出|reports/4b_int4_split_compiled_cases_20261008.json|stock MLX 同 int4 权重单路对拍 20/20，包含 batch 缩小|
|4B 截图两轮|reports/4b_int4_split_compiled_image_session_20261008.json|每轮对完整 eager 重算 4/4；第二轮视觉编码 0 次，复用 691 token，新增 prefill 109 token|
|结构与数值测试|reports/optimization_tests_20261008.log|9 项通过；连续输出对原 kernel 逐位一致，覆盖非连续输入、M=1/2/3/4/5、尾行|
|独立 Metal / Swift kernel|reports/native_int4_verify_final_20261008.json；native_int4_build_verify_v2_20261008.log|Mac GPU 12/12 组与 MLX custom 逐位相同，含 M=1；iOS shader 编译、Swift 类型检查通过，未在 iPhone 执行|
|私有 HTTP 两轮|reports/http_int4_smoke_20261008.json|4B int4 + 编译多输出 + 启动预热，四通道与第二轮缓存复用通过，临时服务已退出|

量化是现有 MLX RTN affine/group64/int4，未做新校准。FP32 scale/bias、归一化参数和 GDN 状态并不代表 FP16 权重推理。矩阵权重及 tied embedding 均保持打包，不生成完整浮点权重副本。跨量化语义质量、command 可解析率仍未验收。视觉暂时沿用既有独立 projector 过桥；语言 int4 包不含完整视觉权重。

## 已尝试的更深 kernel 方向

|方向|实测结论|保留证据（reports/）|
|---|---|---|
|每行 4/8/16/32 lane、标量与 float4 dot|没有稳定收益；较少 lane 常更慢|tuned_kernel_20261008.json|
|group64 先算整数 dot 再应用 scale/bias|约 0.66–0.85×，更慢，FP32 求和顺序改变|group_affine_micro_20261008.json|
|SIMDgroup float32 8×8 矩阵、NT/KT tile、threadgroup padding|padding 改善实验内表现，但仍慢于原 int4 GEMV|matrix_micro_v2_20261008.json、matrix_padding_20261008.json|
|GDN 后处理与动态尾长注意力小图编译|2B 约 3%、4B 约 1.5%，未采用默认|optimize_*_blocks_v2_96_20261008.json|
|线程组 32/64/128/256/512、展开循环、safe/fast math|差别多数约 3–5%，没有稳健突破|gemv_launch_20261008.json|

同步模块探针 profile_blocks_2b_20261008.json 强制每模块同步，会增加约 65% 总时间，仅用于找候选热点；不能当成真实 GPU kernel 占比。首次矩阵编译、首次 shapeless attention 编译和首轮重复权重存储的失败/有偏尝试均保留，没有用于上表候选速度。


## GPU 时间线

模型、前缀和 kernel 全部先预热，再用 xctrace 附加到本项目私有 worker PID；每次捕获两轮、每头 12 token。过滤本 PID、active、depth0 区间，合并重叠区间，避免重复计算 GPU 时间。

基线：578 个 encoder 片段，GPU 活跃并集 0.701s，占首末 GPU 区间跨度 56.8%；编译多输出：536 个片段，0.813s，67.2%。

这不是逐 kernel 归因，也不等于 DRAM 带宽利用率。采样中请求墙钟基线合计 1.330s、编译合计 1.393s，明显受到 Instruments 干扰，与无采样热测速方向不同。GPU 区间之间有空隙，但不能把剩余比例直接叫 CPU 调度成本，尚未证明短生成调度开销 ≤5%。

摘要见 reports/int4_gpu_trace_summary_20261008.json 与 reports/int4_gpu_compiled_trace_summary_20261008.json。首次较长采样在 Instruments 保存阶段超时，失败记录保留；后续缩短采样并允许分段等待保存，两次 trace 成功。

## 仍离极限多远

- 2B：当前 259.5 tok/s，条件带宽参考 811.8 tok/s，约 32.0%。
- 4B：当前 137.2 tok/s，条件带宽参考 356.0 tok/s，约 38.6%。

参考假定 M4 Pro 273GB/s、每步权重仅读一次、四路 GDN 状态各读写一次、单份前缀 KV，仅作为流量模型；不是实测 DRAM 利用率，也不是 iPhone 的硬件极限。四路效率仍约 62–66%，98% 目标尚未达到。

4B 语言参数 2.632GB，加五份 GDN（单份不可变 fork 源 + 四路状态）、695-token 单份前缀 KV、四路 97-token 尾 KV，下限约 2.97GB。这没有包括 canonical 历史缓存、ViT/projector、临时激活、allocator 缓存、PSO 和系统开销；尚未测手机峰值与持续温升。

## 可运行入口

    cd /Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine
    PY=/Users/cialtionshi/mlx_env/bin/python
    $PY run.py --model artifacts/4b_int4 --split-projections --compiled --warm-kernels --compare --max-tokens 96 --prompt '请把明天上午九点设置成提醒' --output reports/my_int4_candidate.json
    $PY serve.py --model artifacts/4b_int4 --split-projections --compiled --warm-kernels
    zsh build_native_int4.sh

离线转换入口 export_int4.py 保存未融合的兼容 tensor 路径、量化配置、tokenizer 和逐张量 SHA-256；拒绝覆盖既有输出目录。4B 加载器拒绝 int8，公开 run 入口不再接受不量化选项。

Python warm_int4_kernels 去重预热真实投影形状与 B=2/3/4，M=1 回退 stock MLX；4B 实测 21 个形状、约 91ms，见 reports/4b_int4_warm_kernel_smoke_20261008.json。原生 Int4KernelManager.swift 将普通/双/四输出 × B=1/2/3/4 归为十二个可缓存 PSO，覆盖停止后缩到单路；N/K 通过参数传递，kernel 不占用整份反量化 scratch。原生版本是投影算子入口，尚未接完整 Swift GDN、共享 KV、M-RoPE 或 App。

[MLX Swift 官方源码提供自定义 Metal kernel 接口](https://github.com/ml-explore/mlx-swift/blob/main/Source/MLXFast/MLXFastKernel.swift)，[官方模型库已有 Qwen3.5](https://github.com/ml-explore/mlx-swift-lm/blob/main/Libraries/MLXLLM/Models/Qwen35.swift)。后续集成需固定依赖提交、保留四通道 fork 契约，在 iPhone 重新验收。
