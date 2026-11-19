# 已遇到的问题与合并禁区

区分：下面标“已定位”的有当前实现/历史验收证据；“约束/风险”是合并时必须维护的边界，不冒称过去已经复现过相同故障。

## 1. 内存被留住不等于 mutex 死锁（已定位）

旧 whole-step compiled tape 会持有模型权重常量；只把 Swift session 置 nil、加 autoreleasepool 或只 clear allocator cache，不足以保证释放。当前 `ChatSession.decode` 使用 `CompiledDecode(..., compileStep:false)`，保留静态 KV tail 和融合 int4 kernels，而不启用整步 compiled tape。

卸载顺序：设置线程安全 stop -> 当前推理在 token/chunk 边界返回 -> 在同一独占队列释放 session 和任务缓存 -> 同步 GPU/CPU stream -> 清理 `mlx_detail_compile_clear_cache()` -> `Memory.clearCache()`。实现见 `ChatMemory.swift`，App 参考见 `Integration/ChatStore.swift` 的 EngineBox.release/unload。不要保存捕获 session/model 的闭包、诊断 MLXArray 或旧 cache 快照。

历史 `iphone_chat_final_ownership_20261008.json` 的 2B->4B->2B 探针：MLX activeMemory 卸载后均归零。它是旧模型 allocator 指标，不是本轮手机 process footprint，也不是全部系统内存归零。

## 2. allocator 内存上限、wired memory 与锁（约束/风险）

固定版本 `Memory.swift` 明确说明：超过 memoryLimit 时分配会等待已调度任务。C++ Metal allocator 的 set_cache_limit/set_memory_limit/set_wired_limit 有 mutex。把 memoryLimit 压到活跃权重+KV+工作区以下，或在推理并发中反复调整全局 limit，会引入等待/内存压力；不能把低上限当成自动降内存方案。

ChatSession 默认 allocator cacheLimit 为 128 MiB，iPhone runtime 对大于2GiB的packed模型进一步设为16MiB，没有在每轮强制修改 memoryLimit 或 wiredLimit。cacheLimit 限制的是可复用空闲缓冲，不是模型活跃内存/整个 App RAM。不要额外给模型做全量 mlock/wired，也不要未经设备验证照抄 Mac 的推荐 working set 作为 iPhone 预算。

此次从现存代码确认的是编译图持有权重和上述 allocator 行为，未找到能独立证明另一个“mutex 死锁已经修复”的复现记录；不能把等待、iOS 锁屏拒绝启动、Jetsam 和真正锁死混称一个 bug。

## 3. 单一所有者与取消（已落实）

ChatSession/Int4Model/LayerState 可变，不是线程安全对象。不要因标注 @unchecked Sendable 就并发调用。一个 serial inference lane 管理加载、预热、生成、记忆任务、切换和清缓存。只有 StopFlag 自身用 NSLock；禁止持锁等待 GPU 或 DispatchQueue.main.sync，避免与 UI 形成等待环。

UI 使用 request epoch 忽略旧任务迟到结果。后台先取消/卸载；前台如果卸载尚未完成，只登记待重载，不启动第二个 load。否则会出现内存叠加、刚加载又回到待加载或旧回调覆盖新状态。

## 4. 不把所有 Mac 最快实验变成 iPhone 默认（已落实）

当前 ChatSession 使用 ProjectionPolicy.both、deferredInterval=1、dynamic history、baseline GDN prepare、64-token prefill chunk、compileStep=false。目录内保留 kernel 和诊断策略以便回归，存在某个 kernel 并不等于默认启用。

保留 packed uint32 int4。原导出scale/bias仍为FP32，build35 A16模式在加载时转换浮点张量，GDN递归矩阵/gates保留FP32。iPhone runtime现默认W4A16＋KV8；低层ChatSession兼容默认仍为FP32，调用者必须明确选择。过去 FP16 compute 有 greedy 分歧记录，因此改成 W4A16/W4A8、重排 GDN 算术或融合投影，都需要同量化权重对拍；不能用吞吐增益替代正确性。build35速度收据分别记录同模式重复、B4/B1与跨精度差异，不以执行完成宣称所有跨精度token一致。

## 5. 量化、张量与 tokenizer（已遇到过协议/下载问题）

- HF hash/长度一致只证明下载一致，另做 finite 检查。转换后必须重载逐张量比对；receipt 记源 revision/hash。
- Qwen3.5 HF RMSNorm 是 offset 权重，MLX 要 +1；conv1d [C,1,4] 要转为 [C,4,1]，只处理一次。不要对已导出的 int4 重做 sanitize。
- 4B int4/group64；此包跳过 vision/MTP，保留完整 source 在原下载目录。没有视觉推理承诺。
- token ID、EOS、marker 从 tokenizer 读取。新模型带单 token marker，旧三 token BPE 假设不可沿用。
- 头结束必须过滤对应 KV/GDN 行；不能拿第一路的停止判定终止全部四头。speech 流式，command/expression 等必须等完整。
- 系统提示词、tokenizer、模型、任务类型变化都可能使缓存失效；不根据“差不多相同”复用 KV。

## 6. 内存统计与 iOS（约束/风险）

MLX active/cache/peak 不含全部 Core ML、llama.cpp、CPU、音频缓冲和系统开销。用 process footprint/真机退出原因评估最终 App；Mac 统一内存和历史峰值不能直接当 iPhone Jetsam 上限。旧 App 使用 increased-memory-limit entitlement，main agent 必须核查新 target 和最终签名 profile，不能只复制 plist 就认为生效。

## 7. 隔离新旧验收

Evidence/Historical 是 1014 真机/TTS/唱歌和旧权重卸载记录；对应旧 IPA 已在先前清理中删除。Evidence/Current 是此次新权重转换及 Mac 测试。新手机包、完整业务 prompt、检索与音频共存仍归最终合版验收，不能引用旧 1014 的“11 项通过”替代。

## 8. Swift 编译通过不等于 Metal 可运行（本轮实际复现）

纯 `swift build` 编译成功后，运行 probe 报 `Failed to load the default metallib`。原因是该包的 MLX Metal 资源需 Xcode 构建；请使用 `Scripts/build_smoke.sh`，不要把 CLI 可链接误认为运行已通过。最终 App 必须带 Cmlx bundle/default.metallib 和 SimpleToolSwift/Kernels 资源。保留首轮失败日志作为证据。


## 9. 分词/诊断临时对象与 4B 余量（2026-11-05）

首轮 4B 短推理和 512-token 测试通过，后续长测进程退出；未收集到系统退出原因，保留 `iPhone/llm-4b-first-incomplete.json`，不把它直接定性为 Jetsam。

已把 ChatTokenizer 每次 encode 的 Regex/NSString 临时对象限制在局部 autoreleasepool，并在原生调度层单独包住 ChatSession 加载；探针 JSON 落盘也单独排空临时对象，避免一个覆盖整个 benchmark 的外层 pool 留住批量临时分配。修正后 4B 两轮完整测试、同进程五次模型切换均通过。fixture_1024 构造前后 footprint 基本不增长。这个结果支持该释放范围修正，但不是对首轮退出原因的最终证明。

实际 profile 不含 increased-memory-limit，正确 key 启用时被 Xcode 拒绝。独立普通签名仍然运行成功，不能据此禁用全部 4B，也不能假定普通签名能同时常驻 4B+所有音频/embedding。

第一轮 active-decode 排空测量和 idle OS 切换分别记录；第二轮另补了真实 OS 打断活跃管线，第三轮补首次加载，具体命中时序以 trace 为准，不能混写成任意 kernel 都可抢占。TTS C 调用、Core ML 单段 prediction 取消边界与最坏后台等待仍要由最终 App 的实际输入预算控制。


## 10. 多引擎共存并非分别跑通的自然推论（第二轮真机）

2B+embedding+TTS+VISinger2全驻留、串行执行已完成五轮。4B旧策略曾在长prefill阶段退出，另一次进程仍活着但数分钟无进展；未拿到系统退出原因或线程栈，不能断言是Jetsam或mutex死锁。三次未完成receipt均保留。

将embedding/音频与4B权重分阶段释放、每阶段单独run排空pool后，mixed流程连续三轮通过；大模型cache16MiB与提前交出transient GDN引用继续降低MLX保留/峰值。整个进程仍一度接近普通profile额度，最终App UI、其他SDK和资源的开销必须另计。

TTS取消不能打断C调用，新segment接口把输入拆为不超过28字符；OS切走后实测等待显著缩短，仍不是硬时限。Core ML首次load也只有模型之间的取消点，单模型load及单prediction内不强行终止；首次load被OS打断的最坏等待未完成验收。

Core ML CPU/GPU选择的输出虽然时长/样本数一致、有限且非静音，波形余弦对all参考却明显降低；因此没有把更快路径替换为默认。音色/听感需另验收，不能用finite检查冒充数值或听感等价。

## 11. 首次 Core ML 加载触发主线程 watchdog（第三轮真机，已定位）

build6 的首次 VISinger2 vocoder load 中真实切后台，crash report 明确是 FRONTBOARD `0x8BADF00D`、scene-update 超过10秒；主线程栈落在 `IPhoneInferenceRuntime.suspendAndDrain` 的 DispatchQueue.sync。这次有退出证据，不要与前述未知退出混称。

build7 的生命周期立即关闭 admission、取消所有已接收请求，将 stream 同步作为异步 lane 屏障；主线程不等待。屏障与 run 入队共享 metadata lock，因此恢复前台后也不能越过未完成的 drain 启动第二个模型。有限 UIApplication 后台任务完成或到期时结束；到期不清空 lane、不放行后台任务、不声称 GPU 可以继续无限运行。真实35秒后台实验中，额度曾到期，返回前台后 drain 完成、随后重新加载和合成通过。同步 drain 仅保留给非主线程诊断。

## 12. 不等长头完成时的旧/新状态重叠（build11实验）

将forward分块或增加stream同步，MLX峰未明显下降；原始build8/9/10记录保留。对完成头的batch筛选原先一次性为全部层构造take结果，旧B4与新B3/B2会短暂同时保留。按层块筛选、求值、同步后，4B同输入实验峰值减少约133MiB，token严格一致。iPhone runtime的大模型默认stride4，小模型保持原路径。不要根据process采样峰的随机变化替代MLX活跃峰与token回归，也不要为降低峰值改成假串行四头。

## 13. 降低MLX峰不等于整进程安全（build12→14）

build12分块策略已将4B的MLX峰降低约133MiB，但动态640MiB预算会因腾出空间而让4B继续驻留，随后Core ML/音频/driver将余量再次吃尽；12轮虽完成，采样最小余量约1MiB。不能把all_passed或allocator改善当作总内存安全。

build14把阶段所有权落实到API：大于2GiB的主模型在embedding/音频前无条件卸载，加载大模型前释放辅助引擎；2B按测量余量保留共驻。卸载后若MLX active仍超过64MiB则报外部借用/持有错误，不继续加载。不要把ChatSession存到外部，或在同一run里持有session再切引擎；请求分成独立run，排空每段autoreleasepool。最终产品仍需为页面/SDK等额外开销验收。

## 14. 长prefix、诊断缓存与取消重试（第四轮）

精确512/768/1024命中+256/512新增的网格全部可执行，但build15/16全程采样曾接近进程额度。按阶段采样确认最差点在额外长对拍/取消恢复诊断：多个请求共用一个run的大pool，并在普通reply取消后用benchmarkReply保留旧prompt cache同时构建新诊断cache。不能把这种诊断开销冒充普通B4请求峰，也不能删除旧记录掩盖风险。

build17逐头、取消、重试分成独立run；重试用普通reply复用有效prompt cache，并核验tokens与原B4完全一致，逐段保留峰值。初始化预算也提前应用到system首次prefill。最终数字以CONTEXT_SUMMARY.json和对应build receipt为准；普通请求的长prefix本身仍增加GDN/KV/driver开销，不能据短prefix结果宣称所有上下文安全。

## 15. 首次system prefill也必须取消（build17→18）

补测短system首次4B加载切后台时，build17出现SIGABRT；ips触发线程为Metal CompletionQueue，栈进入`mlx::core::gpu::check_error`。ips没有包含完整Metal错误文本，因此不声称已确定唯一底层错误码。代码确认当时ChatSession.init未接收取消：weights/tokenizer结束后即使请求已停，仍会继续system GPU预热。

build18把当前lane请求的InferenceStop传入ChatSession初始化，在加载前、权重后、分词后、每64-token system prefill chunk之前及完成后检查；不再让已取消请求继续提交后续预热。取消部分初始化后需stream同步并释放编译/allocator cache。直接使用ChatSession的集成方也应提供cancelled闭包，默认false仅为旧同步调用兼容。

探针还修正了进程监视：必须跟踪launch返回的原PID。进程崩溃后系统或host命令重新拉起同bundle，不代表原run仍在运行；不能只因同名进程存在就继续无限等待旧receipt。旧ips/未完成记录保留。

## 16. 音频文件与embedding输入边界（build22）

原生成功不等于请求已成功返回。TTS/VISinger2曾在写出WAV后、run最终取消检查前被取消，留下无人接收文件；失败原始记录保留。当前以run为文件提交边界，失败/取消仅回收该run新建的音频，成功文件归调用方。TTS私有staging + move避免覆盖已有音频。不要在失败处理里扫描并删除整个GeneratedSinging或调用方输出目录。

embedding C ABI使用NUL终止字符串，Swift入口明确拒绝嵌入NUL，避免无声截断。token预算须包含query/document前缀及模型special tokens；不要以字符数、重复次数估算2048上限。`embeddingTokenCount`与encode使用相同分词规则，前者不跑forward。第五轮边界与恢复检查见`IPHONE_EDGE_RESULTS.md`。



## 17. Apple Speech 任务结束与取消不是同步释放

build27文件转写成功，但紧接着取消一轮再重试时，系统返回 kAFAssistantErrorDomain/1110 No speech detected。首次成功不等于快速取消/重试可靠；失败receipt保留在Build27-ASR。

当前适配器对成功final不再强行cancel已结束任务；保留退出中的recognition handle，下一次请求以异步等待state.completed为屏障（最多3秒，超时报不可用，不阻塞主线程）。start因此为async throws。每轮UUID隔离晚到回调，取消后不能把partial写进草稿；回收录音session后才交出结果。权限弹窗导致inactive时不能取消正在申请权限的业务Task，真正后台仍须取消。


## 18. 构建输入、A16 kernel 与调优证据（build35→40）

build35安装后才写入的A16融合kernel没有自动进入旧包。运行结束后才计算的源码hash不能证明当时编译了这些源码；旧MANIFEST已单独加SOURCE_CORRECTION.md，不删除旧速度结果。build36起每轮保留独立源码快照、编译日志、可执行文件hash；build40交付前再次核验源码和二进制均与快照一致。

W4A16表示激活/输入输出为FP16，不等于所有乘加均用FP16；当前自定义int4投影的解包、反量化及累加仍为FP32，GDN递归状态/gates FP32。packed权重和scale/bias原文件保持不动。A16单头compact QMV可直接读取FP32 metadata，不要因scales不是half而把全部单头又退回每步转换metadata的通用算子。

四头默认保留一份prefix KV和各头GDN状态，B2–B4投影共享权重读取并融合gate/up/SwiGLU。性能候选以同一二进制内legacy/production交替测试；不能把历史不同温度的吞吐变化全归因于dtype。算子profile会强制同步、改变调度，只用于定位；微测tile更快并不保证整轮更快。Q8融合、GDN prepare、stride8和tile表本轮均保留诊断能力，未作为build40默认。


## 19. 执行成功不是所有输入都通过数值验收

A16 speed探针为保留诊断信息，会记录跨batch差异而不立即终止；所以 `all_passed=true` 必须结合 `parity` 和 `real_chat_samples[].speech_reference_equal` 读取。build40的0.8B固定fixture通过，长中文聊天不通过。build42的consistent候选只建立了Mac上述fixture对拍，未完成iPhone速度验收。最终汇总脚本使用显式build与原始token重新校验，任何缺项/差异不能生成全通过报告。


build43已收尾：0.8B启用consistent以恢复上述长中文fixture；2B/4B保留reuse-qmv。三模型完整速度及五项回归均通过，相关字段与raw token由汇总器再次校验。不能把这些fixture成功推广为任意输入/跨精度的严格逐token等价；4B旧65与当前40–44的差异也不能只凭thermal状态做因果归因。见IPHONE_DECODE_OPT_FINAL_RESULTS.md。


## 真实对话循环与精度边界（build46）

用户三轮实际历史在iPhone A16＋KV8 raw greedy稳定循环，而A32正常；Mac同设置也不能预测手机结果。切换旧投影、单头/四头和预热均不能消除。部分计算升FP32只是改变分支，跨batch仍可能不一致，不要用单句成功宣称数值问题已解决。Lab现显式使用conversation防重复策略，低层和benchmark仍raw greedy；策略、token、EOS必须分别审计。详见IPHONE_REPEAT_FIX.md。
