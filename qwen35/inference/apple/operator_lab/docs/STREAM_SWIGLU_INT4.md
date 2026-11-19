# int4 点积寄存器调度与 MLP 算子融合

2026-10-08。本轮继续使用已有 4B/2B int4 group64 语言权重、FP32 激活/计算/GDN 状态，没有重新量化、展开完整稠密权重或新增 FP16 语言推理。元数据半宽存储仍为已经证明无损的存储表示，不能与 FP16 计算混淆。独立工程与共享环境的边界保持不变。

最新候选将 grouped QMV 的逐步输入读取与 int4 gate/up + SwiGLU 融合相结合。两批独立进程各五轮 4B 交替对照均观测约4%的新增时延改善，四头效率约67%；低于此前10%噪声门槛，保留为显式候选，尚不宣称突破或达成98%。

## 本轮算子变化

1. **Grouped QMV。** 借鉴 Apple MLX 单路快速 QMV 的 FP32 算术，将 scale/bias 提到16值点积之外；四路共读打包权重，各路保持单路 QMV 的 reduction 顺序。原始 `grouped_qmv.py` 同时保留大量临时输入，实际四路词表投影和完整模型明显变慢；失败路径与原始报告保留，不进入产品选项。算术布局标注 Apple MIT 来源。
2. **逐步读取。** `stream_grouped_qmv.py` 每次只保留每路4个输入，而非16个；仍保持同一 arithmetic/reduction。对真实4B的7种矩阵、B2/3/4、9个配置分别验证，均逐位等于逐路 stock MLX 快速QMV。微测试两路部分矩阵约12–16%改善，四路多在2–6%；这些同步算子时延含host开销，不是纯GPU事件时间。
3. **按形状选择。** 四路词表和K>4096的down projection保留原row-reuse，其他适用路径使用stream kernel。单路继续原compact QMV；不适用K和prefill保留原路径。Grouped算术与旧逐权重解量化可能有FP32舍入差异，不能用合成逐位测试替代真实token对拍。
4. **MLP融合。** `fused_swiglu_int4.py` 一次dispatch计算gate/up两个int4点积，随后SiLU和乘积直接写输出；原打包tensor保持分开，共读输入，没有复制或合并整个权重。省去两个投影中间输出及后续pointwise dispatch。各点积保持旧row-reuse顺序，投影和activation联合输出逐位对齐。4B的9216×2560形状按B选择tile：B2/B4用R1/T256，B3用R4/T128；其他形状暂用R4/T128，不能外推4B调参收益到2B/iPhone。
5. **预热管理。** `kernel_management.py` 区分stream与普通row-reuse specialization，覆盖实际B2/3/4的融合配置，跳过不会调用的融合前gate/up小batch kernel，保留单路fallback预热。HTTP验证了实际预热类型和总时长记账。whole-model编译图、GDN和attention仍不在该预热范围。

R8的融合MLP配置在本机微测试明显较慢，不采用。这里能确认的是代码临时值规模和实测时延，未取得GPU register/spill计数，不把寄存器溢出推断写成已测事实。

## 完整模型对照

基线已含int4 compact metadata、row-reuse、compiled，两个A/B模型共用权重及元数据tensor。热公共前缀，每路强制96 token、屏蔽EOS/忽略闭合停止，单头/四头五轮交替。不同进程分别重新加载与编译；外部竞争负载未锁定。

|独立进程|基线单头|基线四头|新候选单头|新候选四头|四头时延减少|新单/四效率|
|---|---:|---:|---:|---:|---:|---:|
|第一批|1.2804 s|1.9821 s|1.2771 s|1.9049 s|3.89%|67.04%|
|复测|1.2774 s|1.9819 s|1.2826 s|1.9020 s|4.03%|67.43%|

聚合decode吞吐约194→202 token/s。两批四通道固定token均相同；单头时延基本相同。这是相对当前已优化主线的候选差异，不将历史不同行为/条件的成绩相乘来宣称总体加速。第一批单stream约2.6%，固定R4融合MLP单项约2.3%；这些实验条件独立，不能直接相加。

原始第一版grouped路径四头约4.05s，显著比约1.98s基线慢，见 `4b_grouped_qmv_ab_20261008.json`，未隐藏。微测试扫描记录在 `4b_grouped_qmv_micro_20261008.json`、`4b_stream_grouped_qmv_micro_20261008.json`、`4b_fused_swiglu_micro_20261008.json`。两批组合A/B：`4b_stream_swiglu_ab_20261008.json`、`4b_stream_swiglu_ab_repeat_20261008.json`。

独立单模型的四头中位数1.8983s，MLX buffer峰值2.828GB，未低于上一轮约2.821GB；process footprint峰值约3.466GB，系统压缩与外部负载未锁定，不归因于候选的内存改善。固定token SHA与原路径相同。见 `4b_stream_swiglu_memory_20261008.json`。A/B进程持有两套公共前缀，不能用其峰值推算手机单模型内存。此轮没有证明iPhone RAM节省或完整模型内存改善。

## 端到端正确性与原生移植

- 4B与2B五条实际指令，对安装候选前stock MLX同int4权重单路golden各20/20；自然EOS/标签停止与batch缩小保持。
- 4B实际640px截图单轮与同前缀单路4/4；两轮截图会话对完整重算各4/4，第二轮复用279个前缀token，新增98个prefill token，没有重新视觉编码。仍是增量prefill+decode，不称纯decode。视觉bridge继续复用既有外部f16 projector，本轮语言模型始终int4。
- 私有HTTP服务两轮、完整四通道、缓存复用与实际候选PSO预热通过；验证后已退出。未运行生成的command文本。
- 21项测试通过，新增覆盖各分支对stock单向量QMV、strided输入/输出尾行、gate/up联合投影与activation逐位相等。未声明跨量化语义质量验收。
- `native/SimpleToolSwiGLU.metal`、`native/SwiGLUKernelManager.swift`独立于Python/MLX，实现B2..4、R1/2/4/8、T64/128/256、原/窄元数据；PSO按B/R/存储格式缓存。输入/output长度、形状和明显别名有检查；外部创建的底层buffer别名仍由调用者保证。
- Mac原生288组合逐位相等，含72个真实4B首层int4参数组合；其余为结构fixture。**必须关闭fast-math**，首次默认编译不相等的失败记录保留。iOS metallib与Swift类型检查通过。仅证明融合投影/activation，不是完整模型token或手机性能验收；原生速度尚未测。

真实文本、截图、HTTP与测试证据均为 `reports/*stream_swiglu*20261008*`。原生证据：`native_swiglu_precise_build_20261008.log`、`native_swiglu_ios_typecheck_20261008.log`；失败：`native_swiglu_build_20261008.log`。

## 运行与下一步

在工程目录中，当前候选显式开启：

```sh
PY=/Users/cialtionshi/mlx_env/bin/python
$PY run.py --model artifacts/4b_int4_compact --row-reuse --compact-metadata \
  --compiled --warm-kernels --stream-grouped-qmv --fused-swiglu \
  --cache-limit-mib 256 --compare --prompt '请把明天上午九点设置成提醒'
```

服务同样可用 `--stream-grouped-qmv --fused-swiglu`。没有更换默认为这个候选；各阶段代码和证据均可独立复现。原生构建验证：`zsh build_native_swiglu.sh`；汇总：`python make_stream_swiglu_report.py`只读取保存证据。

进一步的实机推进仍要把int4投影、GDN状态所有权、共享KV、M-RoPE与停止/会话调度串成完整iOS引擎。此轮还未完成这一步，未解决llama.cpp的单路/batch分歧，也未完成CUDA及跨量化质量门槛；Mac条件带宽估算不能当作真实硬件极限或iPhone性能。
