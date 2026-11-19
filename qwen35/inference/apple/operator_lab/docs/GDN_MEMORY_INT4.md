# int4 推理：GDN 状态所有权、原地更新与 allocator 管理

2026-10-08。本轮语言模型始终为 4B/2B affine int4 group64，浮点计算和递归状态为 FP32，没有新增 FP16 语言模型推理。元数据窄存储继续遵守 [COMPACT_INT4.md](COMPACT_INT4.md) 的无损恢复契约。所有文件留在独立工程中。

本轮实现了共享 GDN 首步和原生 Metal 的分支状态原地更新。Mac 完整模型严格对拍通过，但新增速度收益不足以确认；MLX 完整过程的峰值内存没有下降。原生算子独立测试证明可少分配 201.3 MB 递归矩阵缓冲，尚未接成完整 iPhone 模型引擎，不能当作手机内存或速度成绩。

## 共享首步与真实模型

`simpletool/shared_gdn_seed.py` 让四分支的首步读取同一份只读 GDN 前缀矩阵，直接生成四份独立新矩阵。卷积历史仍按分支复制。后续递归使用原 MLX 路径；通道停止后缩小 batch。公共 full-attention KV 仍存单份。

原 packed GDN 的算术布局来自已安装 MLX-LM / Apple MIT 源码，保留版权与许可。现有 packed 布局不是本工程原创；这轮修改的是 batch1 前缀输入与 batchB 独立输出的所有权契约。共享环境文件没有修改。

五轮同进程轮换，4B int4 compact + row reuse + compiled，热共同前缀，每头固定 96 token：

|路径|单头中位数|四头中位数|单/四墙钟效率|
|---|---:|---:|---:|
|现有主线|1.2753 s|1.9851 s|64.24%|
|共享 GDN 首步|1.2746 s|1.9785 s|64.42%|

四头固定 token 全部相同；约 0.3% 差异低于噪声门槛，不能宣称提速。基线是已经优化的 int4 路径，不是旧 kernel。与历史 68.5% 来自不同运行条件，不能直接归因于本轮退化或提升。

同一套 int4 权重，以安装自定义算子前的 stock MLX 单路为 golden，4B 与 2B 五条真实指令各 20/20。4B 截图两轮各 4/4 对完整重算，第二轮复用 279 token、增量 prefill 98 token；不称纯 decode。HTTP 临时服务预热、两轮复用和 256 MiB allocator 配置通过，服务已退出。19 项测试通过，包括完整递归矩阵逐位对拍、前缀不变、分支所有权及混合模型 eager/compiled 对拍。

独立 4B 进程在 256 MiB allocator 配置下，共享首步的 MLX buffer 峰值仍约 2.821 GB，与未共享首步基本相同。后续步骤仍有旧状态与新状态同时存活；清除初始缓存引用也没有测出峰值节省。因此保留 `--shared-gdn-seed` 为显式选项，不声称已经降低完整模型峰值。

证据：`reports/4b_shared_gdn_seed_ab_20261008.json`、`{4b,2b}_shared_gdn_seed_cases_20261008.json`、`4b_shared_gdn_image_session_20261008.json`、`4b_shared_gdn_http_20261008.json`、`4b_shared_gdn_cache_256m_20261008.json`、`shared_seed_all_tests_fixed_20261008.log`。

## 原生 Metal GDN

`native/SimpleToolGDN.metal` 和 `native/GDNKernelManager.swift` 独立于 Python/MLX，支持 Dk128、预归一化 q/k、标量 g/beta、B1..4。前缀矩阵只读；首步写入独立分支矩阵。分支矩阵拥有独立所有权后，kernel 将自己的 32 个元素保留在寄存器，完成两个读出后直接写回原矩阵。禁止把共享前缀当作可写分支。外部调用者仍需保证不同 buffer 的底层内存不重叠；对象身份检查不能识别外部构造的别名。

两个 pipeline 在 manager 初始化时创建并复用：普通输入/输出、owned 原地更新。owned specialization 只通过 buffer5 读写，移除重复的输出绑定。支持单 compute encoder 追加多个互不依赖层；若随后读取此前 dispatch 的结果，调用者必须加 barrier。未把这个原生能力接入 MLX 的不可变数组接口。

**必须用 `-fno-fast-math` 编译 GDN。** 默认 fast-math 首次对拍出现约 1e-9 的差异，原始失败保留。关闭后，HK/HV=(16,32)、(16,16)、(8,16)，B1..4，两种 SIMD group、两种后续状态更新方式，共 48 个组合、每个两步的 y 与完整矩阵逐位相等，前缀不变。覆盖 4B/2B 实际 head 比例。Mac 验证通过，iOS metallib 编译及 Swift 类型检查通过；未在手机运行。

独立算子实验使用 4B 的 24 个 GDN 层、B4、HK16/HV32、宽128；输入是固定随机 fixture，只有 recurrence，没有 int4 projection、attention、卷积或完整模型。每个层使用独立 buffer，五轮每轮32步，预热后记录 command GPU 时间和同步墙钟。

|原生模式，2 SIMD group|矩阵 buffer 总数/大小|每24层 GPU时间|每24层同步墙钟|
|---|---:|---:|---:|
|双缓冲，分 encoder|403 MB|1.860 ms|2.113 ms|
|双缓冲，单 encoder|403 MB|1.869 ms|2.053 ms|
|原地更新，分 encoder|201 MB|1.952 ms|2.148 ms|
|原地更新，单 encoder|201 MB|1.961 ms|2.189 ms|

精确少分配 201,326,592 字节递归矩阵，即 192 MiB。其余 fixture/output 约4.7 MB，分别使全部显式 buffer 为407.4/206.1 MB。原地更新仍约慢5%，单 encoder 的收益小且不稳定，不能宣称性能提升。移除重复绑定前的独立实验也保留，差异不足以证明 specialization 提速。缓冲分配数字不是全模型进程 footprint，也不是实机 iPhone 省内存验收。

构建：`zsh build_native_gdn.sh`。算子测试：`xcrun swiftc -O native/GDNKernelManager.swift native/bench_gdn.swift -o artifacts/bench_gdn_native`，再执行 `artifacts/bench_gdn_native artifacts/native_gdn_fixture artifacts/simpletool_gdn_mac.metallib`。数字见 `reports/native_gdn_owned_schedule_bench_20261008.json`；48组合见 `native_gdn_owned_build_20261008.log`。

## allocator cache 与贪心输出

新增 `--cache-limit-mib` 到 run/serve/check_session，作用仅为当前 MLX 进程的空闲 buffer 复用缓存软上限，不限制活跃 tensor，不是应用 RAM 硬上限。服务 health 返回实际策略。没有在每个 token 清理 allocator。

独立进程同输入、每头96 token、五轮。以下旧 seed 路径之间比较；所有 token SHA-256 相同，未锁定外部负载，系统压缩量在各进程间差异很大。

|空闲缓存策略|四头中位数|测量结束缓存|MLX buffer峰值|进程 footprint峰值|
|---|---:|---:|---:|---:|
|框架默认|1.9823 s|680 MB|2.821 GB|3.645 GB|
|256 MiB|1.9841 s|269 MB|2.821 GB|3.466 GB|
|64 MiB|2.0348 s|68 MB|2.821 GB|3.342 GB|
|0，单独复测|2.0841 s|0|2.821 GB|3.319 GB|

256 MiB 没有可确认的速度损失，适合继续作为内存实验配置。缓存可以短时超过软限。footprint 的减少仅是这些独立进程的观测，不能据此预测 iPhone。首次 cache=0 与本工程私有验证意外重叠，已标记排除并单独重跑，未删除记录。

另做 int4 LM projection + argmax 融合，输出由约3.97 MB完整 logits减为约0.248 MB候选分块；处理 tie、NaN、屏蔽token及尾线程。4B 真实指令20/20，算子测试通过，但完整模型五轮没有提速，因此 `--fused-head` 只留在实验验证入口，没有加入产品默认路径。

汇总和源码/产物 SHA-256：`reports/gdn_memory_summary_20261008.json`，可用 `python make_gdn_memory_report.py` 从已保存证据重建，不启动推理。

## 下一道门槛

98% 四头效率尚未达到。本轮没有解决 llama.cpp 的单路/batch 贪心差异，没有做跨量化质量验收，也没有交付完整 iOS 模型引擎。现有图像流程仍复用外部 f16 视觉 projector；本轮未新增 FP16 语言推理。后续需要将原生 int4 projection、GDN 所有权、KV/M-RoPE 与调度串成真实模型，并在连接且可用的手机上测峰值、稳定时延和持续功耗。
