# 聊天温度参数（build47）

2026-11-07。Lab默认温度为0.7，四个头都采用这个温度；speech/inner另外保留build46的重复约束。expression/command不加文本重复惩罚，但同样进行温度采样。

- App入口：模型 → 对话参数 → 采样温度，0–2，步长0.05，自动保存，下次发送生效。
- 旧存档没有temperature字段时，默认迁移为0.7；不会清空历史或自定义提示词。
- 后端接口：独占推理队列上设置`session.decodingPolicy = .conversation; session.temperature = 0.7`。低层temperature属性默认0.7，限制有限数值0...2；低层policy仍默认greedy用于兼容，只有conversation启用该采样参数。
- 计算：先处理speech/inner重复约束，再对每个头以MLXRandom.categorical(logits/temperature)采样。temperature0改用argmax。没有额外增加top-k/top-p筛选。
- 原始greedy、固定步数benchmark及memoryReply保持有效温度0；随机采样不用于单头/四头逐token对拍。Lab调用结束后恢复原session参数，避免影响后续任务。
- 每轮ChatDiagnostics记录实际temperature；Lab事件日志也记录本次生效温度。

这是本进程Swift参数，不是新开一个远程HTTP设置服务。

```swift
session.decodingPolicy = .conversation
session.temperature = 0.7
let turn = try session.reply(query, preparedPayload: payload)
```

验证证据在`Evidence/Current/iPhone/Build47-Temperature/`；build46及以前记录保持原版本含义。

已执行：Mac 16384次二分类抽样频率检验符合temperature0.7的softmax分布；真机新设置默认0.7、旧存档迁移0.7、1.15持久化回读通过。4B实际对话0.7与0、用户原历史回放0.7与0都完成，diagnostics温度核对通过；纳西妲单/四头对拍显式设0并通过。卸载后MLX active为16字节（未宣称为0）。测试后已恢复普通Lab。原始数值分歧仍按build46记录保留，本次未声称修复跨精度token等价。
