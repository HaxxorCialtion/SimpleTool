# int4 元数据无损压缩与直接读取 kernel

更新：2026-10-08。4B/2B 的语言权重均保持 affine int4 group64；这轮没有重新量化，也没有新增 FP16 模型推理。16 位仅用于存储已经证明可无损恢复的 scale/bias；解量化乘加、激活、embedding 输出、GDN 状态仍为 FP32。

本轮主要价值是减少 int4 的额外元数据。完整审计 4B 的 498 个与 2B 的 374 个 scale/bias tensor，全部可缩成半宽存储再恢复到原 FP32 值，没有误差。uint32 打包权重及其他 tensor 完全不变，保存后对所有恢复 tensor 的 SHA-256 验证通过。

|模型|原 tensor payload|压缩 tensor payload|节省|重载验收|
|---|---:|---:|---:|---|
|4B|2.632 GB|2.369 GB|262.8 MB|全部恢复值逐位相同|
|2B|1.178 GB|1.060 GB|117.6 MB|全部恢复值逐位相同|

新产物：artifacts/4b_int4_compact 与 artifacts/2b_int4_compact。旧 int4 与原始转换输入完整保留。每个目录的 compact_int4_receipt.json 包含逐张量恢复 SHA-256、最终文件 SHA-256、来源及 dtype。

对量化矩阵，元数据由每 64 权重 8 字节减至 4 字节：int4 主体不变，其含元数据的存储成本由 5.0 降为 4.5 bit/weight。这不是每个模型参数都量化的假设；上表按真实全部 tensor 汇总。

## 计算与 kernel

- 四路/三路/两路：原多行寄存器复用 Metal kernel 直接读窄 scale/bias，先在寄存器中扩大为 float，再执行相同 FP32 解量化、乘加及 SIMD reduction，不展开完整元数据。
- 单路：新增 compact_qmv.py，沿用 Apple MLX 快速 QMV 的 16 值/线程计算与 reduction 顺序，直接读取窄元数据。适用 M=1、K 可被 512 整除、N 可被 8 整除；真实主矩阵形状验证逐位相同。复制的 load/qdot 算术标注 Apple 版权并附 MIT license。
- 不适用形状和 prefill：stock MLX 将元数据临时扩大为 FP32；这会增加瞬时流量与 buffer，不能说所有路径都直接读取压缩参数。
- embedding 查表：只先扩大被索引的 scale/bias 行，再 FP32 解量化；避免 embedding dtype 随存储类型变成半精度。
- 启动管理：PSO/MLX specialization key 区分元数据 dtype；预热补入单路 QMV。读取已经压缩的产物时不为检查同一表示而重新扩大整份元数据；从 FP32 安装压缩时仍拒绝任意不能精确恢复的值，并在全部检查通过后才修改模型。

原生 Metal/Swift 加入同样的存储 specialization：输入/输出/计算为 float；scale/bias buffer 可是 2 或 4 字节元素。PSO key 和长度校验包含存储格式。这是投影算子能力，未实现完整 iPhone GDN/KV/M-RoPE 引擎；Python 的快速单路 QMV 也未宣称已经移植为独立 iOS 完整推理路径。

## 完整 4B 对照

基线已经是上一轮 row reuse；不是最初旧 kernel。五轮同进程轮换、每头 96 token、热公共前缀、相同 int4 权重及 FP32 计算。仅元数据在两个 A/B 模型中分别持有，uint32 权重仍共用。

|路径|单头中位数|四头中位数|单/四墙钟效率|
|---|---:|---:|---:|
|row reuse|1.433s|2.231s|64.2%|
|row reuse + 编译|1.385s|2.030s|68.3%|
|压缩 + 直接 QMV|1.421s|2.214s|64.2%|
|压缩 + 直接 QMV + 编译|1.301s|2.000s|65.1%|

四头编译路径约多改善 1.5%，低于 10% 噪声门槛，不称为确定速度收益，也尚未独立复测。上一轮旧 kernel→row reuse+编译约 24% 的独立复测仍见 DEEP_INT4.md。2B 压缩路径尚未做完整速度 A/B。

最初版本让单路走 stock 临时扩大，测出单路约 16% 变慢，效率数值因此虚高到约 80%；该失败方向报告保留于 reports/optimize_4b_compact_metadata_96_20261008.json，不能拿来证明四路效率。直接 QMV 消除了这项代价。首次小 N 不对齐的逐位检查失败也保留于 reports/compact_qmv_unit_20261008.log；最终对不对齐形状明确回退，没有放宽逐位验收。

## 内存实测边界

两份独立进程各加载一份 4B，做一次冷编译文本推理和一次热推理；同 96 token/头，四头 token 全相等。不是共用参数 A/B 进程的峰值。

|Mac 文本指标|原 row reuse int4|压缩 int4|
|---|---:|---:|
|加载后 MLX 活跃 buffer|2.632 GB|2.369 GB|
|MLX 进程 buffer 峰值|3.084 GB|2.821 GB|
|末次读取 allocator cache|0.638 GB|0.749 GB|
|系统 physical footprint 峰值|3.751 GB|3.606 GB|
|系统 ru_maxrss|3.242 GB|2.995 GB|

系统数据通过本进程 task_info(TASK_VM_INFO) 及 getrusage 读取，offset 对照本机 SDK 头文件。实际 footprint 峰值只下降约 144.5 MB，少于权重节省；allocator cache 及系统记账有额外成本。较早一对进程的 RSS 甚至反向变化，原始 reports/4b_single_model_memory_* 保留。不能只看 MLX buffer 就声称设备进程占用相同比例下降。

本轮没有调低 allocator cache 上限，也没有在采样前主动释放热缓存。物理峰值仅一对 Mac 进程实测，不是 iPhone jetsam 预算，不包含独立视觉进程；iPhone 的总内存和延迟仍未实测。下一轮应比较 cache/command-buffer 策略的速度与真实 footprint 代价，并移植完整原生引擎。

## 验收

- 4B/2B 新产物：全部恢复 tensor 逐位相同；各五条真实指令、四个自然停止通道 20/20 对 stock MLX 同 int4 权重串行 gold，包含 B 缩减。见 reports/4b_compact_artifact_cases_20261008.json 与 2b_compact_artifact_cases_20261008.json。
- 4B 截图两轮：各 4/4 对优化路径完整 eager 重算。640px 图使用已有视觉缓存，260 个视觉 token，第二轮复用 279 前缀 token，增量 prefill 98，视觉重编码 0。尚未测冷视觉编码或手机视觉路径。
- 私有 HTTP：压缩 artifact + 编译 + 预热，两轮四通道及第二轮 KV 复用通过；仅终止自建子进程。见 reports/http_compact_int4_final_smoke_20261008.json。
- 最终 15 项结构/数值测试通过：逐位 QMV、输入 stride、压缩拒绝与安装原子性、embedding FP32、重复安装、共享前缀与分支状态、缩 batch、分块与其他既有 kernel。见 reports/compact_int4_final_tests_20261008.log。
- 原生 Metal：原格式 48/48 回归、窄元数据 48/48 逐位一致；iOS18 shader 编译与 Swift 类型检查通过。手机未运行。这不等于完整 iPhone App 验收。

没有新增量化质量评价，也没有把模型输出当命令执行。视觉仍依赖已有 projector bridge，语言包不包含完整视觉权重。

## 使用

    cd /Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine
    PY=/Users/cialtionshi/mlx_env/bin/python
    $PY run.py --model artifacts/4b_int4_compact --row-reuse --compact-metadata --compiled --warm-kernels --compare --prompt '请把明天上午九点设置成提醒'
    $PY serve.py --model artifacts/4b_int4_compact --row-reuse --compact-metadata --compiled --warm-kernels

2B 替换模型目录为 artifacts/2b_int4_compact。未指定 --compact-metadata 时仍可按原 FP32 元数据加载此标准命名 artifact，恢复原路径；显式打开压缩才直接读取窄参数。默认入口不变，M4 Pro tile 不能当作 iPhone 已验证选型。

4B 必须 int4 的底线已在加载器约束；2B int8 仍是可选后续对照，尚未做。本轮保持目标 active：98% 效率、iPhone 完整引擎、实机验收仍未达到。
