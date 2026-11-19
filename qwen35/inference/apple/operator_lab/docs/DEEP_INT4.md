# 更深入的 int4 kernel 迭代

更新：2026-10-08。4B 保持 int4 打包权重、FP32 浮点计算与 GDN 状态；没有重新量化，也没有新增 FP16 模型推理。工作仍在独立目录，不修改 Zcode、共享 MLX 环境或常驻服务。

本轮主要结果：多行寄存器复用改变了实际 Metal 内循环。4B 两个独立进程各五轮交替，原 kernel → 新 kernel + 编译均观测约 1.24×；第二批样本波动小，跨过 10% 噪声门槛。它尚未达到 98% 四路效率或完成 iPhone 验收。

## 改了哪一层

原实现一个 SIMD group 负责一行，每行重发输入向量读取。新实现一个 SIMD group 同时负责 4/8 行：每个 word 的输入只加载到寄存器一次，再用于多行 int4 点积。每行仍按相同 word/j 顺序 FP32 累加，再作相同 SIMD reduction，因此无需降低精度或展开整份浮点权重。

按主要形状选择 M4 Pro 初始策略：大词表投影用 4 行 / 256 线程；输入宽 ≤2560 的普通大投影用 8 行 / 256 线程；更宽的 down 投影用 8 行 / 128 线程；小于 256 行及单路保留原路径。这个选择尚不是 iPhone 的实测最优策略。

Python 算子位于 simpletool/row_reuse_int4.py；原生 Metal 额外保留原函数作对照。Swift pipeline key 包含 batch、输出数和行复用数；N/K 保持 runtime 参数，按需缓存 PSO。

## 完整模型同进程对照

每头 96 token、屏蔽 EOS、相同热公共前缀；各 A/B 模型共用同一份 int4 参数，单头/四头交替五轮。新 kernel 在 B=2/3/4 生效，M=1 保留 stock MLX。加载与首次编译不计入热墙钟。

|批次|原 kernel 四头|编译旧 kernel|新 kernel|编译新 kernel|总观测提速|新路径单头/四头效率|
|---|---:|---:|---:|---:|---:|---:|
|4B first|3.970s|3.706s|3.411s|3.200s|1.240×|72.9%|
|4B repeat|2.517s|2.340s|2.222s|2.030s|1.240×|68.5%|
|2B|1.707s|1.550s|1.596s|1.421s|1.201×|71.6%|

4B 单独新 kernel 两批分别约 1.164× / 1.133×；编译在此之上继续减少总时延。第二批四头基线五样本为 2.523/2.517/2.514/2.516/2.518s，新 kernel + 编译为 2.055/2.032/2.030/2.025/2.027s。

第一批 4B 和 2B 后段有明显瞬时扰动，原始样本全部保留；2B 尚未独立复测。不同进程绝对速度不同，不能把跨时段变化当成 kernel 收益。所有报告记录共享 GDN 文件的开始/结束 SHA-256，均一致；其既有 packed 实现不是本项目的新贡献。

第二批 4B 聚合 decode 189.3 tok/s，相对于此前流量模型的 356.0 tok/s 条件参考约 53.2%。这不是实测 DRAM 利用率，也不能用于推断 iPhone 峰值。

## 已验的边界

|对象|证据|结论|
|---|---|---|
|2B 自然停止|reports/2b_int4_row_reuse_compiled_cases_20261008.json|stock MLX 同 int4 权重串行 20/20；包含 B=4→3→2→1|
|4B 自然停止|reports/4b_int4_row_reuse_compiled_cases_20261008.json|stock MLX 同 int4 权重串行 20/20；包含 B=4→3→2→1|
|4B 截图两轮|reports/4b_int4_row_reuse_image_session_20261008.json|各 4/4 对完整 eager 重算；第二轮复用 691 token，视觉重算 0，增量 prefill 109 token|
|HTTP 两轮及启动预热|reports/http_row_reuse_smoke_20261008.json|私有随机端口，四通道及第二轮缓存复用通过；子进程已退出|
|结构数值测试|reports/deep_int4_tests_20261008.log|12 项通过：新 row kernel 的尾行/K=320/非连续输入/B1..4，各 1/2/4/8 行及线程组；编译块末尾不足一块、父缓存不变；GDN conv 历史与数值|
|原生 Metal|reports/native_row_reuse_verify_20261008.json|48/48 组逐位等于 MLX custom；原/2/4/8 行、B1..4、单/双/四输出；iOS shader 编译和 Swift 类型检查通过，未在手机运行|

完整模型只对同一 MLX int4 权重的推理路径负责。没有宣称跨 CUDA BF16/不同量化 token 相等，也没有完成量化语义质量、command 可解析率或 App 实机验收。截图仍使用此前的独立 projector 过桥，语言权重包不包含完整视觉权重。

## 继续向下改了但没有采用的路径

连续自回归 GPU 块：CompiledStep.advance_pending 将 2/4/8 个连续步骤接在一个编译图中，中间 token 直接作为下一步 GPU 输入，不依赖草稿模型。停止后的额外 token 丢弃，存活分支在块边界压缩；它会延迟检测停止并多算少量已停止分支。固定长度四路 token 全相等，合成测试已通过，但真实自然停止的整套验收尚未做。

2B 编译新 kernel 单步同步 1.111s，8-step 块 1.099s，只改善约 1.1%；不作为推荐入口。见 reports/optimize_2b_chunks_96_20261008.json。

GDN gate 双输出：把 decay 与 beta 合并为一个 kernel，五个 B/H 形状逐位相同，但序列化 microbench 没有稳定收益，未加入默认。见 reports/gdn_gate_pair_micro_20261008.json。

GDN 卷积融合：把 concat/history shift、depthwise kernel4、SiLU 合成一个双输出 kernel；实测 B1..4 输出与新历史逐位相同，合成 stride/尾线程测试通过。4B B=4 microbench 约 15% 算子收益，但完整模型 eager 仅约 0.6%，编译路径仅约 0.1%；不作为推荐入口，也未做真实指令整套验收。见 reports/gdn_conv_state_micro_20261008.json 与 optimize_4b_row_reuse_conv_96_20261008.json。

这两处结果说明仍要按完整模型的实际成本继续找瓶颈，不能把算子 microbench 收益直接当成最终时延改善。

## 内存与 iPhone

4B 复测运行期间，MLX 活跃 tensor 最大约 2.77GB，整个进程自开始计的 MLX buffer 分配峰值约 3.20GB。这个数来自同参数 A/B 文本测量，包含框架 buffer 峰值；不等于完整 RSS、不包含独立视觉进程，也不是手机内存测量。2B 连续提交块的进程 MLX buffer 峰值约 1.405GB，未观察到块长度造成峰值增长，但这一结论仅覆盖本次文本形状。

下一层仍需优化解量化指令、参数存储流量及 GDN 递推矩阵的读写，在手机重新选择寄存器 tile 和线程组；不能将 M4 Pro 的选择直接作为 iPhone 最优配置。4B 继续严格 int4，2B int8 对照仍未做。

## 复现

    cd /Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine
    PY=/Users/cialtionshi/mlx_env/bin/python
    $PY run.py --model artifacts/4b_int4 --row-reuse --compiled --warm-kernels --compare --prompt '请把明天上午九点设置成提醒'
    $PY serve.py --model artifacts/4b_int4 --row-reuse --compiled --warm-kernels
    $PY bench_optimization.py --model artifacts/4b_int4 --strategy row_reuse --rounds 5 --output reports/my_row_ab.json
    $PY validate_cases.py --model artifacts/4b_int4 --row-reuse --compiled --output reports/my_row_cases.json
    zsh build_native_int4.sh

采用显式 --row-reuse；公共默认继续保留旧路径，避免把 M4 Pro 的选型直接默认套给其他芯片。FP32 的 GDN 递推、共享单份 full-attention 前缀、各分支独立可写状态、真实 tokenizer 标记与独立停止契约保持不变。
