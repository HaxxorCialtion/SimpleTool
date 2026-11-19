# SimpleTool H200 固定公共前缀与变化环境输入：完整测试报告

日期：2026-11-13。五个模型全部完成。

本轮全部速度不含 content。所有模型均为 5% 数据版本。每模型 180 个不同场景输入、600 个额外变化观测，以及 320 次纯 infra 冷/热对照请求。各场景不是两道题重复十五次。

场景为固定策略和 tools 下的推理 fixture，没有执行真实机械臂、游戏或数字人工具。变化部分补齐到 512 tokens，包含显式标注的无作用诊断标点；原始观测本身不一定有 512 tokens。纯 infra 则严格为固定 512＋变化 512。

## 1. 设备、模型与精度

节点 `qb-prod-gpu1019`，分配 4×NVIDIA H200 141GB。模型依次运行，4B/9B/35B 各使用单卡，122B 使用 TP2，397B 使用 TP4。没有把五个模型同时压满同一 GPU。

| 模型 | TP | 模型 dtype | 权重量化 | async scheduling |
|---|---|---|---|---|
| 4B INT4 | 1 | bfloat16 | awq | True |
| 9B INT4 | 1 | bfloat16 | awq | True |
| 35B-A3B FP8 | 1 | bfloat16 | fp8 | True |
| 122B-A10B FP8 | 2 | bfloat16 | fp8 | True |
| 397B-A17B FP8 | 4 | bfloat16 | fp8 | False |

| 模型 | backbone 层数 | hidden size | attention Q/KV heads | GDN key/value heads | MoE experts / top-k |
|---|---|---|---|---|---|
| 4B INT4 | 32 | 2560 | 16/4 | 16/32 | dense/— |
| 9B INT4 | 32 | 4096 | 16/4 | 16/32 | dense/— |
| 35B-A3B FP8 | 40 | 2048 | 16/2 | 16/32 | 256/8 |
| 122B-A10B FP8 | 48 | 3072 | 32/2 | 16/64 | 256/8 |
| 397B-A17B FP8 | 60 | 4096 | 32/2 | 16/64 | 512/10 |

这些模型每四层安排一层 full attention，其余为 GDN；它们不是纯 Transformer。工具分支共享同一套 backbone 和 LM projection，通过不同特殊起始 marker 启动各序列；不增加七份模型权重。MoE 的各分支可选择不同 expert，本轮未测 DRAM counter 或证明每份权重只读一次。原始模型配置与 tokenizer 哈希见 `MODEL_METADATA.json`，依赖版本见 `ENVIRONMENT.json`。

4B/9B 使用 AWQ INT4，group_size=64、zero_point=true，实际加载 backend 见运行日志；A 为 BF16 计算配置，不是 int8 activation。KV/GDN dtype 配置为 auto，没有开启 KV8。大模型使用已验证的 FP8 权重视图、BF16 model dtype；FP8 GEMM 内部的 activation operands 量化由 backend 决定，不能笼统称为所有层 W8A16。35B 实际基座是 Qwen3.6-35B-A3B，其余为 Qwen3.5。

本表混合 INT4/FP8、稠密/MoE 和不同 TP，适用于部署预算，不是同精度的参数 scaling 对照。

## 2. 公共缓存实际命中

先对固定前缀单独 prefill 并保留公共末端状态，再逐题提交变化输入。测量来自引擎返回的实际 `num_cached_tokens`。纯 infra 热路径每题命中 512、新算 512，完整输入 prompt 的命中比例为 50%；固定公共部分自身覆盖率为 100%。冷对照使用唯一 cache_salt，跨题命中为 0，但同请求跨 head 共享仍开启。

真实场景的前缀是完整 system＋策略＋tools，按实际 tokenizer 得到长度；变化后缀为 512。每个工具头复用本次全部公共输入，只处理其自己的 1-token marker。因此跨 head 公共覆盖率为 100%，包含 marker 的分支 prompt 命中比例为 N/(N+1)，不是整个系统完全没有计算新输入。

| 模型 | 机械臂 P/S | NPC P/S | 数字人 P/S | 每场景不同输入数 |
|---|---|---|---|---|
| 4B INT4 | 399/512 | 407/512 | 420/512 | 60 |
| 9B INT4 | 399/512 | 407/512 | 420/512 | 60 |
| 35B-A3B FP8 | 399/512 | 407/512 | 420/512 | 60 |
| 122B-A10B FP8 | 399/512 | 407/512 | 420/512 | 60 |
| 397B-A17B FP8 | 399/512 | 407/512 | 420/512 | 60 |

旧动态 800 题的跨题公共命中为 0，原因是旧实验只发布各题完整末端状态，没有先独立保存 system/tools 边界；同时大多数新 prefill 形状走 eager。它不能证明公共 system 文本不应或不能命中，也没有证据表明当时主要是 KV cache 查找或显存池满造成变慢。

## 3. 纯 infra：512 固定＋512 变化

下面均为热前缀、每 head 强制生成 32 tokens，20 个不同后缀/格。TPOT 为单 head 相邻输出 token 的主机端间隔统计，已包含原生 decode 调度；B=7 不含 content。prefill 列为一次公共状态准备和发布耗时，而非纯 GPU kernel 时间。

| 模型 | 公共 prepare P50 ms | B1 TPOT ms | B2 TPOT ms | B4 TPOT ms | B7 TPOT ms | B7 TTFT P50 ms | B7 全部结束 P50 ms |
|---|---|---|---|---|---|---|---|
| 4B INT4 | 18.782 | 3.510 | 3.521 | 3.566 | 3.669 | 33.101 | 146.997 |
| 9B INT4 | 22.038 | 4.370 | 4.400 | 4.445 | 4.551 | 37.260 | 178.431 |
| 35B-A3B FP8 | 23.755 | 3.837 | 3.986 | 4.471 | 4.988 | 39.094 | 193.848 |
| 122B-A10B FP8 | 33.667 | 5.308 | 5.472 | 6.260 | 6.891 | 51.493 | 265.500 |
| 397B-A17B FP8 | 42.881 | 8.074 | 8.273 | 9.208 | 10.380 | 62.606 | 384.828 |

冷/热公共状态准备对照（同一批输入、同样的跨 head 共享）：

| 模型 | 冷 P50/P95 ms | 热 P50/P95 ms | B7 TPOT P50/P95 ms | 热完整 prompt 命中/新增 |
|---|---|---|---|---|
| 4B INT4 | 24.061/24.129 | 18.782/18.903 | 3.669/3.674 | 512/512 |
| 9B INT4 | 32.434/32.509 | 22.038/22.259 | 4.551/4.565 | 512/512 |
| 35B-A3B FP8 | 29.828/30.402 | 23.755/23.846 | 4.988/5.023 | 512/512 |
| 122B-A10B FP8 | 43.669/43.984 | 33.667/34.114 | 6.891/7.061 | 512/512 |
| 397B-A17B FP8 | 56.520/56.925 | 42.881/44.059 | 10.380/10.668 | 512/512 |

B1/B7 TPOT 比例可用于观察 batch 后每 head 间隔的变化；B1 这里仍是 function head，不能当成同基座普通 JSON AR baseline。本轮没有重新完成同基座 AR 的质量＋E2E 对照。16-token 输出长度及全部 P50/P95/P99/均值见 `pure_infra.csv` 与 `RESULTS.json`。

## 4. 三种场景实际输出耗时

tools-ready 从新请求提交开始，包含变化后缀 prefill、公共状态发布、head marker、解码和函数组装；终点由实际 function 对应 schema 的必要 head 决定，不由 gold label 决定。无效函数无法组装时不产生该计时，覆盖数量在表内给出。参数值错误仍可能有可组装调用，所以速度和正确率分开列。

| 模型 | 场景 | B | tools-ready P50/P95 ms | 计时覆盖 | 正确调用子集 P50 ms | 全 head 排空 P50/P95 ms | 调用正确/函数正确 |
|---|---|---|---|---|---|---|---|
| 4B INT4 | 机械臂 | 6 | 42.551/52.787 | 53/60 | 47.638 | 42.533/52.784 | 20/60; 33/60 |
| 4B INT4 | 游戏 NPC | 5 | 40.780/44.156 | 60/60 | 40.788 | 40.785/44.161 | 24/60; 60/60 |
| 4B INT4 | 数字人 | 5 | 40.927/41.089 | 60/60 | 40.920 | 40.937/41.096 | 28/60; 58/60 |
| 9B INT4 | 机械臂 | 6 | 49.242/49.424 | 60/60 | 49.328 | 49.251/49.435 | 10/60; 43/60 |
| 9B INT4 | 游戏 NPC | 5 | 47.570/60.783 | 60/60 | 47.607 | 47.576/60.789 | 33/60; 60/60 |
| 9B INT4 | 数字人 | 5 | 47.697/48.604 | 60/60 | 47.701 | 47.705/48.611 | 35/60; 54/60 |
| 35B-A3B FP8 | 机械臂 | 6 | 53.380/64.676 | 60/60 | 53.352 | 53.384/64.682 | 59/60; 59/60 |
| 35B-A3B FP8 | 游戏 NPC | 5 | 51.712/63.768 | 60/60 | 51.742 | 51.717/63.774 | 47/60; 60/60 |
| 35B-A3B FP8 | 数字人 | 5 | 52.022/56.250 | 60/60 | 52.014 | 52.031/56.255 | 46/60; 55/60 |
| 122B-A10B FP8 | 机械臂 | 6 | 74.285/89.290 | 60/60 | 74.045 | 74.295/89.294 | 42/60; 50/60 |
| 122B-A10B FP8 | 游戏 NPC | 5 | 71.679/88.579 | 60/60 | 71.730 | 71.685/88.583 | 54/60; 60/60 |
| 122B-A10B FP8 | 数字人 | 5 | 71.414/77.672 | 60/60 | 71.414 | 71.422/77.676 | 60/60; 60/60 |
| 397B-A17B FP8 | 机械臂 | 6 | 98.468/120.474 | 60/60 | 96.382 | 98.478/120.479 | 47/60; 48/60 |
| 397B-A17B FP8 | 游戏 NPC | 5 | 94.459/119.558 | 60/60 | 94.443 | 94.464/119.563 | 58/60; 60/60 |
| 397B-A17B FP8 | 数字人 | 5 | 94.091/102.447 | 60/60 | 94.091 | 94.100/102.451 | 60/60; 60/60 |

完整必要参数正确率没有排除模型选错动作、抄错参数或 head 对应错误。每条实际调用、预期调用和必要 head 的最后 token 时间见 `requests.csv`；所有 head 的输出、TPOT 和自己的结束 token 状态见 `heads.csv`。

当前实现一批全部已启动 head 排空后再处理下一逻辑请求。`1000/tools_ready_ms` 是单次决策预算的倒数，不能直接称作整个物理系统已达到的连续控制 Hz。

## 5. 正确性对照与数值路径差异

| 模型 | 真实输入对照数 | 无前缀缓存同切分 head 对照 | 有固定前缀缓存同切分 head 对照 | 不同冷重算路径存在差异的 case 数 |
|---|---|---|---|---|
| 4B INT4 | 12 | 64/64 | 64/64 | 1 |
| 9B INT4 | 12 | 64/64 | 64/64 | 1 |
| 35B-A3B FP8 | 12 | 62/64 | 64/64 | 4 |
| 122B-A10B FP8 | 12 | 64/64 | 64/64 | 1 |
| 397B-A17B FP8 | 12 | 64/64 | 64/64 | 1 |

正式固定前缀缓存路径的批量 head 与独立单 head 必须逐 token 一致，失败会停止本轮。无固定前缀 batch/单头与整段冷重算是独立数值路径，其差异已审计，不声称逐 token 一致。整段冷 prefill 与分段 prefill 的比较单独保留；其形状、算子执行和浮点累加路线不同，本轮不能宣称全路径逐 token 等价。差异的原始 token、前五 logprob、失败运行与重跑来源均保留。原 9B 严格整段冷对照在 npc-00004 的 arg2 上停下：切分为 `44`，整段为 `<|null|>`，二者都不符合期望的 wait_ms；该失败没有被记作完成测试。后续增加了与正式缓存路径相同的前缀切分对照，再进行计时。

| 模型 | 缓存批量/独立单头：必要调用一致 | 缓存/整段冷重算：必要调用一致 |
|---|---|---|
| 4B INT4 | 12/12 | 11/12 |
| 9B INT4 | 12/12 | 11/12 |
| 35B-A3B FP8 | 12/12 | 10/12 |
| 122B-A10B FP8 | 12/12 | 12/12 |
| 397B-A17B FP8 | 12/12 | 11/12 |

TP1 的物理页布局已留直接收据；TP2/TP4 没有采到 driver-side 物理页别名 trace，不能把逻辑命中收据当作各 rank 的直接物理共享证明。

预热阶段另对公共 CUDA graph 与对应 eager forward 的 hidden/GDN 状态做验证；正式计时关闭 trace、logprob 和新图捕获。质量评分与逐 token 实现一致性是两种不同证据。

## 6. 持续变化输入与稳定性

每模型另跑 600 个不同输入，三场景交错，和正式评分集无重复。首尾各 150 条覆盖相同场景比例，但输出长度可能不同；判断稳定性同时查看 TPOT、prefill、图计数与各 rank 显存，不能只凭 E2E 小幅差异归因于缓存。

| 模型 | 首150 tools-ready P50 ms | 尾150 P50 ms | 变化 | 首/尾 TPOT ms | 公共图数/各 rank | 显存 allocated/reserved 是否不变 |
|---|---|---|---|---|---|---|
| 4B INT4 | 40.926 | 40.903 | -0.057% | 2.926/2.917 | 9 | True |
| 9B INT4 | 49.191 | 49.189 | -0.004% | 3.959/3.961 | 9 | True |
| 35B-A3B FP8 | 54.028 | 54.050 | 0.042% | 3.953/3.954 | 9 | False |
| 122B-A10B FP8 | 74.721 | 74.712 | -0.013% | 5.775/5.774 | 9/9 | False |
| 397B-A17B FP8 | 98.891 | 98.797 | -0.095% | 9.815/9.893 | 9/9/9/9 | False |

| 模型 | rank | allocated 增量 MiB | reserved 增量 MiB |
|---|---|---|---|
| 4B INT4 | 0 | 0.000 | 0.000 |
| 9B INT4 | 0 | 0.000 | 0.000 |
| 35B-A3B FP8 | 0 | 0.000 | 50.000 |
| 122B-A10B FP8 | 0 | -0.001 | 50.000 |
| 122B-A10B FP8 | 1 | 0.000 | 50.000 |
| 397B-A17B FP8 | 0 | 0.000 | 26.000 |
| 397B-A17B FP8 | 1 | 0.000 | 26.000 |
| 397B-A17B FP8 | 2 | 0.000 | 26.000 |
| 397B-A17B FP8 | 3 | 0.000 | 26.000 |

部分 reserved pool 有小幅扩张：35B/122B 为 50 MiB/rank，397B 为 26 MiB/rank；不能称为所有显存完全不变。公共图保持 9 个/rank，场景计时期间无新增 eager fallback，持续输入首尾未出现明显减速。本轮 397B 捕获阶段出现 multi-stream symmetric collective 警告，但随后完成全部测试；已保留日志。这不等于已证明任意并发 collective 都安全。397B 还提示缺少特定 MoE tuning config，因此这里是已测部署配置，不能声称穷尽所有 kernel 调优。

所有正式场景＋600 条持续输入期间禁止新捕获，并检查没有新增 eager fallback；rank 收据和 pool 快照保留。仅在本轮形状、长度、并发和 4 GiB KV pool 下得出此结论，不外推为任意动态形状或多租户场景没有退化。

## 7. 三组可重放实例

下例给出各场景前两帧。公共策略和工具定义不变，观测改变导致期望动作或参数改变。所有模型实际输出均列出，包含错误。完整 system、schema、观测与输入 token IDs 在数据和 examples 文件。

### 机械臂

输入 robot-00000：

```json

{
  "frame_id": 10000,
  "task_id": "pick-0000",
  "emergency_stop": true,
  "obstruction": false,
  "object_in_gripper": false,
  "at_pick": false,
  "at_drop": false,
  "placement_verified": false,
  "pick_target": {
    "x_mm": 210,
    "y_mm": -90,
    "z_mm": 105
  },
  "drop_target": {
    "x_mm": 330,
    "y_mm": 70,
    "z_mm": 125
  },
  "requested_speed_mm_s": 35,
  "gripper_close_width_mm": 18,
  "gripper_force_n": 9,
  "gripper_open_width_mm": 70,
  "joint_encoder_deg": [
    11,
    -24,
    37,
    -48,
    59,
    63
  ]
}

```

期望：`{"name": "stop_motion", "arguments": {"reason": "emergency_stop"}}`

| 模型 | 实际组装调用 | 正确 | tools-ready ms |
|---|---|---|---|
| 4B INT4 | {"name": "move_cartesian", "arguments": {"x_mm": 210, "y_mm": -90, "z_mm": 105, "speed_mm_s": 35, "frame": "base"}} | False | 42.734 |
| 9B INT4 | {"name": "stop_motion", "arguments": {"reason": "<|null|>"}} | False | 44.586 |
| 35B-A3B FP8 | {"name": "stop_motion", "arguments": {"reason": "emergency_stop"}} | True | 52.030 |
| 122B-A10B FP8 | {"name": "stop_motion", "arguments": {"reason": "emergency_stop"}} | True | 68.904 |
| 397B-A17B FP8 | {"name": "move_cartesian", "arguments": {"x_mm": 210, "z_mm": 105, "speed_mm_s": 35, "frame": "base"}} | False | 98.615 |

输入 robot-00001：

```json

{
  "frame_id": 10001,
  "task_id": "pick-0001",
  "emergency_stop": false,
  "obstruction": true,
  "object_in_gripper": false,
  "at_pick": false,
  "at_drop": false,
  "placement_verified": false,
  "pick_target": {
    "x_mm": 211,
    "y_mm": -89,
    "z_mm": 106
  },
  "drop_target": {
    "x_mm": 331,
    "y_mm": 71,
    "z_mm": 126
  },
  "requested_speed_mm_s": 36,
  "gripper_close_width_mm": 19,
  "gripper_force_n": 10,
  "gripper_open_width_mm": 71,
  "joint_encoder_deg": [
    12,
    -23,
    38,
    -47,
    60,
    64
  ]
}

```

期望：`{"name": "stop_motion", "arguments": {"reason": "obstruction"}}`

| 模型 | 实际组装调用 | 正确 | tools-ready ms |
|---|---|---|---|
| 4B INT4 | {"name": "stop_motion", "arguments": {"reason": "211"}} | False | 42.823 |
| 9B INT4 | {"name": "stop_motion", "arguments": {"reason": "211"}} | False | 49.309 |
| 35B-A3B FP8 | {"name": "stop_motion", "arguments": {"reason": "obstruction"}} | True | 50.703 |
| 122B-A10B FP8 | {"name": "stop_motion", "arguments": {"reason": "obstruction"}} | True | 67.629 |
| 397B-A17B FP8 | {"name": "stop_motion", "arguments": {"reason": "obstruction"}} | True | 87.833 |

### 游戏 NPC

输入 npc-00000：

```json

{
  "frame_id": 20000,
  "npc_id": "npc-0",
  "hp": 20,
  "hostile_visible": false,
  "hostile_in_range": false,
  "hostile_id": "enemy-0",
  "available_skill": "fire",
  "player_message_pending": false,
  "player_id": "player-0",
  "requested_reply": "Gate 0 is open.",
  "idle_requested": false,
  "wait_ms": 200,
  "safe_point": {
    "x": 10,
    "y": 20
  },
  "hostile_point": {
    "x": 80,
    "y": 90
  },
  "patrol_point": {
    "x": 40,
    "y": 50
  },
  "walk_speed": 2,
  "nearby_entity_ids": [
    "entity-0000",
    "entity-0001",
    "entity-0002"
  ]
}

```

期望：`{"name": "move_to", "arguments": {"npc_id": "npc-0", "x": 10, "y": 20, "speed": 2}}`

| 模型 | 实际组装调用 | 正确 | tools-ready ms |
|---|---|---|---|
| 4B INT4 | {"name": "move_to", "arguments": {"npc_id": "npc-0", "x": 40, "y": 20, "speed": 2}} | False | 40.758 |
| 9B INT4 | {"name": "move_to", "arguments": {"npc_id": "npc-0", "x": 40, "y": 50, "speed": 2}} | False | 47.483 |
| 35B-A3B FP8 | {"name": "move_to", "arguments": {"npc_id": "npc-0", "x": 10, "y": 20, "speed": 2}} | True | 51.742 |
| 122B-A10B FP8 | {"name": "move_to", "arguments": {"npc_id": "npc-0", "x": 10, "y": 20, "speed": 2}} | True | 71.027 |
| 397B-A17B FP8 | {"name": "move_to", "arguments": {"npc_id": "npc-0", "x": 10, "y": 20, "speed": 2}} | True | 94.124 |

输入 npc-00001：

```json

{
  "frame_id": 20001,
  "npc_id": "npc-1",
  "hp": 71,
  "hostile_visible": true,
  "hostile_in_range": true,
  "hostile_id": "enemy-1",
  "available_skill": "ice",
  "player_message_pending": false,
  "player_id": "player-1",
  "requested_reply": "Gate 1 is open.",
  "idle_requested": false,
  "wait_ms": 220,
  "safe_point": {
    "x": 11,
    "y": 21
  },
  "hostile_point": {
    "x": 81,
    "y": 91
  },
  "patrol_point": {
    "x": 41,
    "y": 51
  },
  "walk_speed": 3,
  "nearby_entity_ids": [
    "entity-0001",
    "entity-0002",
    "entity-0003"
  ]
}

```

期望：`{"name": "attack_target", "arguments": {"npc_id": "npc-1", "target_id": "enemy-1", "skill": "ice"}}`

| 模型 | 实际组装调用 | 正确 | tools-ready ms |
|---|---|---|---|
| 4B INT4 | {"name": "attack_target", "arguments": {"npc_id": "npc-1", "target_id": "enemy-1", "skill": "81"}} | False | 40.947 |
| 9B INT4 | {"name": "attack_target", "arguments": {"npc_id": "npc-1", "target_id": "enemy-1", "skill": "ice"}} | True | 47.565 |
| 35B-A3B FP8 | {"name": "attack_target", "arguments": {"npc_id": "npc-1", "target_id": "enemy-1", "skill": "ice"}} | True | 51.510 |
| 122B-A10B FP8 | {"name": "attack_target", "arguments": {"npc_id": "npc-1", "target_id": "enemy-1", "skill": "ice"}} | True | 71.078 |
| 397B-A17B FP8 | {"name": "attack_target", "arguments": {"npc_id": "npc-1", "target_id": "enemy-1", "skill": "ice"}} | True | 95.421 |

### 数字人

输入 avatar-00000：

```json

{
  "frame_id": 30000,
  "avatar_id": "avatar-0",
  "event": "expression_request",
  "requested_emotion": "happy",
  "intensity_pct": 30,
  "transition_ms": 100,
  "target_id": "visitor-0",
  "hold_ms": 400,
  "gesture": "wave",
  "loop": false,
  "visitor_position_mm": [
    110,
    230,
    1500
  ],
  "current_emotion": "calm"
}

```

期望：`{"name": "set_expression", "arguments": {"avatar_id": "avatar-0", "emotion": "happy", "intensity_pct": 30, "transition_ms": 100}}`

| 模型 | 实际组装调用 | 正确 | tools-ready ms |
|---|---|---|---|
| 4B INT4 | {"name": "set_expression", "arguments": {"avatar_id": "avatar-0", "emotion": "happy", "intensity_pct": 30, "transition_ms": 100}} | True | 40.919 |
| 9B INT4 | {"name": "set_expression", "arguments": {"avatar_id": "avatar-0", "emotion": "happy", "intensity_pct": 30, "transition_ms": 100}} | True | 47.478 |
| 35B-A3B FP8 | {"name": "set_expression", "arguments": {"avatar_id": "avatar-0", "emotion": "happy", "intensity_pct": 30, "transition_ms": 100}} | True | 51.837 |
| 122B-A10B FP8 | {"name": "set_expression", "arguments": {"avatar_id": "avatar-0", "emotion": "happy", "intensity_pct": 30, "transition_ms": 100}} | True | 71.554 |
| 397B-A17B FP8 | {"name": "set_expression", "arguments": {"avatar_id": "avatar-0", "emotion": "happy", "intensity_pct": 30, "transition_ms": 100}} | True | 94.211 |

输入 avatar-00001：

```json

{
  "frame_id": 30001,
  "avatar_id": "avatar-1",
  "event": "gaze_request",
  "requested_emotion": "sad",
  "intensity_pct": 31,
  "transition_ms": 120,
  "target_id": "visitor-1",
  "hold_ms": 450,
  "gesture": "nod",
  "loop": true,
  "visitor_position_mm": [
    111,
    231,
    1501
  ],
  "current_emotion": "happy"
}

```

期望：`{"name": "look_at", "arguments": {"avatar_id": "avatar-1", "target_id": "visitor-1", "hold_ms": 450}}`

| 模型 | 实际组装调用 | 正确 | tools-ready ms |
|---|---|---|---|
| 4B INT4 | {"name": "look_at", "arguments": {"avatar_id": "avatar-1", "target_id": "sad", "hold_ms": 31}} | False | 40.952 |
| 9B INT4 | {"name": "set_expression", "arguments": {"avatar_id": "avatar-1", "emotion": "sad", "intensity_pct": 120, "transition_ms": 120}} | False | 47.617 |
| 35B-A3B FP8 | {"name": "look_at", "arguments": {"avatar_id": "avatar-1", "target_id": "visitor-1", "hold_ms": 450}} | True | 52.166 |
| 122B-A10B FP8 | {"name": "look_at", "arguments": {"avatar_id": "avatar-1", "target_id": "visitor-1", "hold_ms": 450}} | True | 71.769 |
| 397B-A17B FP8 | {"name": "look_at", "arguments": {"avatar_id": "avatar-1", "target_id": "visitor-1", "hold_ms": 450}} | True | 95.601 |

## 8. Infra 实现、配置和代码位置

详尽协议见 `PROTOCOL.md`。执行流程：

```text
固定 system/tools tokens
  → prefill-only（0 输出）
  → CUDA 完成 / 发布精确末端 KV + GDN/conv 状态
  → pin 公共边界
变化 environment/query tokens
  → 复用公共边界，计算新后缀
  → 公共状态物化后提交 function,arg1…argN
  → 原生 batch forward，每 head 独立结束
  → function schema 决定必要参数，最慢必要 head 完成即 tools-ready
  → 多余 head 物理排空另计
```

这是私有 vLLM overlay 和 worker extension 的实现，不是仅设置 enable_prefix_caching 就能保证所有同时到达的 head 都自动复用刚计算的公共状态。`prefix_match_unit=1` 是哈希匹配粒度，不是物理 KV page。该混合架构的普通 page 对齐可为 528 等大小；本实现显式发布精确末端边界以复用部分页和递归状态。

公共完整 KV 页只读共享，非对齐末页按原生 copy-on-write 复制为各 head 的私有页；可变 GDN/conv 状态独立。逻辑 prefill 复用不等于物理零复制，也不意味着每个 KV 字节只从 DRAM 读一次。本轮没有启用 shared_attention.py 的实验性专用 attention kernel；marker/结束 token 来自实际 tokenizer。状态完成由引擎和 CUDA event 同步判断，不轮询磁盘日志触发分支。

公共图上限 24，capture 只在预热；decode 使用原生 FULL_DECODE_ONLY，batch 1…8。缓存模型 forward 的执行图，不缓存旧环境的数值输出。多卡 graph context 处理 all-reduce buffer 注册，捕获 warmup 恢复递归状态。

本机新实验目录：
```text
/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/experiments/simpletool_h200_speed_20261113/scenario_v2
```

云端新实验目录：
```text
/inspire/qb-ilm2/project/new-paradigm-of-manipulation/shixiaoxin-253107030017/simpletool_h200_speed_20261113/scenario_v2
```

本机核心源文件（全部绝对路径）：
```text
/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/experiments/simpletool_h200_speed_20261113/scenario_v2/bench_scenarios.py
/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/experiments/simpletool_h200_speed_20261113/scenario_v2/workload.py.inc
/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/experiments/simpletool_h200_speed_20261113/scenario_v2/make_scenarios.py
/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/experiments/simpletool_h200_speed_20261113/scenario_v2/controller.py
/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/experiments/simpletool_h200_speed_20261113/scenario_v2/launch_scenarios.sh
/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/experiments/simpletool_h200_speed_20261113/scenario_v2/PROTOCOL.md
/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/experiments/simpletool_h200_speed_20261113/payload/tp_worker.py
/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/experiments/simpletool_h200_speed_20261113/payload/fast_prefill.py
/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/experiments/simpletool_h200_speed_20261113/payload/benchmark_lib.py
/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/experiments/simpletool_h200_speed_20261113/payload/runtime/prefill-only-20261110/code/prefill_runtime.py
/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/experiments/simpletool_h200_speed_20261113/payload/runtime/prefill-only-20261110/overlay/scheduler.py
```

云端 GPU Python：
```text
/inspire/hdd/global_user/shixiaoxin-253107030017/sxx/vllm_qwen35_sm90_simpletool/bin/python
```

各模型权重的绝对路径及量化元数据：

4B INT4:

```text

/inspire/qb-ilm2/project/new-paradigm-of-manipulation/shixiaoxin-253107030017/simpletool_eval_20261110/public_release_20261112/models/SimpleTool-Qwen3.5-4B-8Head-5pct-INT4

```

版本：`{"torch": "2.14.0+cu130", "vllm": "0.33.0", "cuda": "13.0"}`；quantization：

```json

{
  "quant_method": "awq",
  "bits": 4,
  "group_size": 64,
  "zero_point": true,
  "version": "gemm",
  "lm_head": false,
  "modules_to_not_convert": [
    "visual",
    "lm_head",
    "in_proj_ba",
    "in_proj_a",
    "in_proj_b",
    "mtp"
  ]
}

```

9B INT4:

```text

/inspire/qb-ilm2/project/new-paradigm-of-manipulation/shixiaoxin-253107030017/simpletool_eval_20261110/public_release_20261112/models/SimpleTool-Qwen3.5-9B-8Head-5pct-INT4

```

版本：`{"torch": "2.14.0+cu130", "vllm": "0.33.0", "cuda": "13.0"}`；quantization：

```json

{
  "quant_method": "awq",
  "bits": 4,
  "group_size": 64,
  "zero_point": true,
  "version": "gemm",
  "lm_head": false,
  "modules_to_not_convert": [
    "visual",
    "lm_head",
    "in_proj_ba",
    "in_proj_a",
    "in_proj_b",
    "mtp"
  ]
}

```

35B-A3B FP8:

```text

/inspire/qb-ilm2/project/new-paradigm-of-manipulation/shixiaoxin-253107030017/simpletool_eval_20261110/eval_views/p005-35b-fp8

```

版本：`{"torch": "2.14.0+cu130", "vllm": "0.33.0", "cuda": "13.0"}`；quantization：

```json

{
  "activation_scheme": "dynamic",
  "fmt": "e4m3",
  "quant_method": "fp8",
  "modules_to_not_convert": [
    "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.mlp.linear_fc1",
    "model.visual.blocks.0.mlp.linear_fc2",
    "visual.blocks.0.attn.proj",
    "visual.blocks.0.attn.qkv_proj",
    "visual.blocks.0.mlp.linear_fc1",
    "visual.blocks.0.mlp.linear_fc2",
    "model.visual.blocks.1.attn.proj",
    "model.visual.blocks.1.attn.qkv",
    "model.visual.blocks.1.mlp.linear_fc1",
    "model.visual.blocks.1.mlp.linear_fc2",
    "visual.blocks.1.attn.proj",
    "visual.blocks.1.attn.qkv_proj",
    "visual.blocks.1.mlp.linear_fc1",
    "visual.blocks.1.mlp.linear_fc2",
    "model.visual.blocks.2.attn.proj",
    "model.visual.blocks.2.attn.qkv",
    "model.visual.blocks.2.mlp.linear_fc1",
    "model.visual.blocks.2.mlp.linear_fc2",
    "visual.blocks.2.attn.proj",
    "visual.blocks.2.attn.qkv_proj",
    "visual.blocks.2.mlp.linear_fc1",
    "visual.blocks.2.mlp.linear_fc2",
    "model.visual.blocks.3.attn.proj",
    "model.visual.blocks.3.attn.qkv",
    "model.visual.blocks.3.mlp.linear_fc1",
    "model.visual.blocks.3.mlp.linear_fc2",
    "visual.blocks.3.attn.proj",
    "visual.blocks.3.attn.qkv_proj",
    "visual.blocks.3.mlp.linear_fc1",
    "visual.blocks.3.mlp.linear_fc2",
    "model.visual.blocks.4.attn.proj",
    "model.visual.blocks.4.attn.qkv",
    "model.visual.blocks.4.mlp.linear_fc1",
    "model.visual.blocks.4.mlp.linear_fc2",
    "visual.blocks.4.attn.proj",
    "visual.blocks.4.attn.qkv_proj",
    "visual.blocks.4.mlp.linear_fc1",
    "visual.blocks.4.mlp.linear_fc2",
    "model.visual.blocks.5.attn.proj",
    "model.visual.blocks.5.attn.qkv",
    "model.visual.blocks.5.mlp.linear_fc1",
    "model.visual.blocks.5.mlp.linear_fc2",
    "visual.blocks.5.attn.proj",
    "visual.blocks.5.attn.qkv_proj",
    "visual.blocks.5.mlp.linear_fc1",
    "visual.blocks.5.mlp.linear_fc2",
    "model.visual.blocks.6.attn.proj",
    "model.visual.blocks.6.attn.qkv",
    "model.visual.blocks.6.mlp.linear_fc1",
    "model.visual.blocks.6.mlp.linear_fc2",
    "visual.blocks.6.attn.proj",
    "visual.blocks.6.attn.qkv_proj",
    "visual.blocks.6.mlp.linear_fc1",
    "visual.blocks.6.mlp.linear_fc2",
    "model.visual.blocks.7.attn.proj",
    "model.visual.blocks.7.attn.qkv",
    "model.visual.blocks.7.mlp.linear_fc1",
    "model.visual.blocks.7.mlp.linear_fc2",
    "visual.blocks.7.attn.proj",
    "visual.blocks.7.attn.qkv_proj",
    "visual.blocks.7.mlp.linear_fc1",
    "visual.blocks.7.mlp.linear_fc2",
    "model.visual.blocks.8.attn.proj",
    "model.visual.blocks.8.attn.qkv",
    "model.visual.blocks.8.mlp.linear_fc1",
    "model.visual.blocks.8.mlp.linear_fc2",
    "visual.blocks.8.attn.proj",
    "visual.blocks.8.attn.qkv_proj",
    "visual.blocks.8.mlp.linear_fc1",
    "visual.blocks.8.mlp.linear_fc2",
    "model.visual.blocks.9.attn.proj",
    "model.visual.blocks.9.attn.qkv",
    "model.visual.blocks.9.mlp.linear_fc1",
    "model.visual.blocks.9.mlp.linear_fc2",
    "visual.blocks.9.attn.proj",
    "visual.blocks.9.attn.qkv_proj",
    "visual.blocks.9.mlp.linear_fc1",
    "visual.blocks.9.mlp.linear_fc2",
    "model.visual.blocks.10.attn.proj",
    "model.visual.blocks.10.attn.qkv",
    "model.visual.blocks.10.mlp.linear_fc1",
    "model.visual.blocks.10.mlp.linear_fc2",
    "visual.blocks.10.attn.proj",
    "visual.blocks.10.attn.qkv_proj",
    "visual.blocks.10.mlp.linear_fc1",
    "visual.blocks.10.mlp.linear_fc2",
    "model.visual.blocks.11.attn.proj",
    "model.visual.blocks.11.attn.qkv",
    "model.visual.blocks.11.mlp.linear_fc1",
    "model.visual.blocks.11.mlp.linear_fc2",
    "visual.blocks.11.attn.proj",
    "visual.blocks.11.attn.qkv_proj",
    "visual.blocks.11.mlp.linear_fc1",
    "visual.blocks.11.mlp.linear_fc2",
    "model.visual.blocks.12.attn.proj",
    "model.visual.blocks.12.attn.qkv",
    "model.visual.blocks.12.mlp.linear_fc1",
    "model.visual.blocks.12.mlp.linear_fc2",
    "visual.blocks.12.attn.proj",
    "visual.blocks.12.attn.qkv_proj",
    "visual.blocks.12.mlp.linear_fc1",
    "visual.blocks.12.mlp.linear_fc2",
    "model.visual.blocks.13.attn.proj",
    "model.visual.blocks.13.attn.qkv",
    "model.visual.blocks.13.mlp.linear_fc1",
    "model.visual.blocks.13.mlp.linear_fc2",
    "visual.blocks.13.attn.proj",
    "visual.blocks.13.attn.qkv_proj",
    "visual.blocks.13.mlp.linear_fc1",
    "visual.blocks.13.mlp.linear_fc2",
    "model.visual.blocks.14.attn.proj",
    "model.visual.blocks.14.attn.qkv",
    "model.visual.blocks.14.mlp.linear_fc1",
    "model.visual.blocks.14.mlp.linear_fc2",
    "visual.blocks.14.attn.proj",
    "visual.blocks.14.attn.qkv_proj",
    "visual.blocks.14.mlp.linear_fc1",
    "visual.blocks.14.mlp.linear_fc2",
    "model.visual.blocks.15.attn.proj",
    "model.visual.blocks.15.attn.qkv",
    "model.visual.blocks.15.mlp.linear_fc1",
    "model.visual.blocks.15.mlp.linear_fc2",
    "visual.blocks.15.attn.proj",
    "visual.blocks.15.attn.qkv_proj",
    "visual.blocks.15.mlp.linear_fc1",
    "visual.blocks.15.mlp.linear_fc2",
    "model.visual.blocks.16.attn.proj",
    "model.visual.blocks.16.attn.qkv",
    "model.visual.blocks.16.mlp.linear_fc1",
    "model.visual.blocks.16.mlp.linear_fc2",
    "visual.blocks.16.attn.proj",
    "visual.blocks.16.attn.qkv_proj",
    "visual.blocks.16.mlp.linear_fc1",
    "visual.blocks.16.mlp.linear_fc2",
    "model.visual.blocks.17.attn.proj",
    "model.visual.blocks.17.attn.qkv",
    "model.visual.blocks.17.mlp.linear_fc1",
    "model.visual.blocks.17.mlp.linear_fc2",
    "visual.blocks.17.attn.proj",
    "visual.blocks.17.attn.qkv_proj",
    "visual.blocks.17.mlp.linear_fc1",
    "visual.blocks.17.mlp.linear_fc2",
    "model.visual.blocks.18.attn.proj",
    "model.visual.blocks.18.attn.qkv",
    "model.visual.blocks.18.mlp.linear_fc1",
    "model.visual.blocks.18.mlp.linear_fc2",
    "visual.blocks.18.attn.proj",
    "visual.blocks.18.attn.qkv_proj",
    "visual.blocks.18.mlp.linear_fc1",
    "visual.blocks.18.mlp.linear_fc2",
    "model.visual.blocks.19.attn.proj",
    "model.visual.blocks.19.attn.qkv",
    "model.visual.blocks.19.mlp.linear_fc1",
    "model.visual.blocks.19.mlp.linear_fc2",
    "visual.blocks.19.attn.proj",
    "visual.blocks.19.attn.qkv_proj",
    "visual.blocks.19.mlp.linear_fc1",
    "visual.blocks.19.mlp.linear_fc2",
    "model.visual.blocks.20.attn.proj",
    "model.visual.blocks.20.attn.qkv",
    "model.visual.blocks.20.mlp.linear_fc1",
    "model.visual.blocks.20.mlp.linear_fc2",
    "visual.blocks.20.attn.proj",
    "visual.blocks.20.attn.qkv_proj",
    "visual.blocks.20.mlp.linear_fc1",
    "visual.blocks.20.mlp.linear_fc2",
    "model.visual.blocks.21.attn.proj",
    "model.visual.blocks.21.attn.qkv",
    "model.visual.blocks.21.mlp.linear_fc1",
    "model.visual.blocks.21.mlp.linear_fc2",
    "visual.blocks.21.attn.proj",
    "visual.blocks.21.attn.qkv_proj",
    "visual.blocks.21.mlp.linear_fc1",
    "visual.blocks.21.mlp.linear_fc2",
    "model.visual.blocks.22.attn.proj",
    "model.visual.blocks.22.attn.qkv",
    "model.visual.blocks.22.mlp.linear_fc1",
    "model.visual.blocks.22.mlp.linear_fc2",
    "visual.blocks.22.attn.proj",
    "visual.blocks.22.attn.qkv_proj",
    "visual.blocks.22.mlp.linear_fc1",
    "visual.blocks.22.mlp.linear_fc2",
    "model.visual.blocks.23.attn.proj",
    "model.visual.blocks.23.attn.qkv",
    "model.visual.blocks.23.mlp.linear_fc1",
    "model.visual.blocks.23.mlp.linear_fc2",
    "visual.blocks.23.attn.proj",
    "visual.blocks.23.attn.qkv_proj",
    "visual.blocks.23.mlp.linear_fc1",
    "visual.blocks.23.mlp.linear_fc2",
    "model.visual.blocks.24.attn.proj",
    "model.visual.blocks.24.attn.qkv",
    "model.visual.blocks.24.mlp.linear_fc1",
    "model.visual.blocks.24.mlp.linear_fc2",
    "visual.blocks.24.attn.proj",
    "visual.blocks.24.attn.qkv_proj",
    "visual.blocks.24.mlp.linear_fc1",
    "visual.blocks.24.mlp.linear_fc2",
    "model.visual.blocks.25.attn.proj",
    "model.visual.blocks.25.attn.qkv",
    "model.visual.blocks.25.mlp.linear_fc1",
    "model.visual.blocks.25.mlp.linear_fc2",
    "visual.blocks.25.attn.proj",
    "visual.blocks.25.attn.qkv_proj",
    "visual.blocks.25.mlp.linear_fc1",
    "visual.blocks.25.mlp.linear_fc2",
    "model.visual.blocks.26.attn.proj",
    "model.visual.blocks.26.attn.qkv",
    "model.visual.blocks.26.mlp.linear_fc1",
    "model.visual.blocks.26.mlp.linear_fc2",
    "visual.blocks.26.attn.proj",
    "visual.blocks.26.attn.qkv_proj",
    "visual.blocks.26.mlp.linear_fc1",
    "visual.blocks.26.mlp.linear_fc2",
    "model.visual.deepstack_merger_list.0.linear_fc1",
    "model.visual.deepstack_merger_list.0.linear_fc2",
    "model.visual.deepstack_merger_list.0.norm",
    "visual.deepstack_merger_list.0.linear_fc1",
    "visual.deepstack_merger_list.0.linear_fc2",
    "visual.deepstack_merger_list.0.norm",
    "model.visual.deepstack_merger_list.1.linear_fc1",
    "model.visual.deepstack_merger_list.1.linear_fc2",
    "model.visual.deepstack_merger_list.1.norm",
    "visual.deepstack_merger_list.1.linear_fc1",
    "visual.deepstack_merger_list.1.linear_fc2",
    "visual.deepstack_merger_list.1.norm",
    "model.visual.deepstack_merger_list.2.linear_fc1",
    "model.visual.deepstack_merger_list.2.linear_fc2",
    "model.visual.deepstack_merger_list.2.norm",
    "visual.deepstack_merger_list.2.linear_fc1",
    "visual.deepstack_merger_list.2.linear_fc2",
    "visual.deepstack_merger_list.2.norm",
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
    "model.visual.merger.norm",
    "model.visual.patch_embed.proj",
    "model.visual.pos_embed",
    "visual.merger.linear_fc1",
    "visual.merger.linear_fc2",
    "visual.merger.norm",
    "visual.patch_embed.proj",
    "visual.pos_embed",
    "visual",
    "model.visual",
    "lm_head",
    "model.embed_tokens",
    "model.language_model.layers.0.input_layernorm",
    "model.language_model.layers.0.mlp.shared_expert_gate",
    "model.language_model.layers.0.post_attention_layernorm",
    "model.language_model.layers.0.mlp.gate",
    "model.language_model.layers.0.linear_attn.A_log",
    "model.language_model.layers.0.linear_attn.conv1d",
    "model.language_model.layers.0.linear_attn.dt_bias",
    "model.language_model.layers.0.linear_attn.in_proj_ba",
    "model.language_model.layers.0.linear_attn.in_proj_b",
    "model.language_model.layers.0.linear_attn.in_proj_a",
    "model.language_model.layers.0.linear_attn.norm",
    "model.language_model.layers.1.input_layernorm",
    "model.language_model.layers.1.mlp.shared_expert_gate",
    "model.language_model.layers.1.post_attention_layernorm",
    "model.language_model.layers.1.mlp.gate",
    "model.language_model.layers.1.linear_attn.A_log",
    "model.language_model.layers.1.linear_attn.conv1d",
    "model.language_model.layers.1.linear_attn.dt_bias",
    "model.language_model.layers.1.linear_attn.in_proj_ba",
    "model.language_model.layers.1.linear_attn.in_proj_b",
    "model.language_model.layers.1.linear_attn.in_proj_a",
    "model.language_model.layers.1.linear_attn.norm",
    "model.language_model.layers.2.input_layernorm",
    "model.language_model.layers.2.mlp.shared_expert_gate",
    "model.language_model.layers.2.post_attention_layernorm",
    "model.language_model.layers.2.mlp.gate",
    "model.language_model.layers.2.linear_attn.A_log",
    "model.language_model.layers.2.linear_attn.conv1d",
    "model.language_model.layers.2.linear_attn.dt_bias",
    "model.language_model.layers.2.linear_attn.in_proj_ba",
    "model.language_model.layers.2.linear_attn.in_proj_b",
    "model.language_model.layers.2.linear_attn.in_proj_a",
    "model.language_model.layers.2.linear_attn.norm",
    "model.language_model.layers.3.input_layernorm",
    "model.language_model.layers.3.mlp.shared_expert_gate",
    "model.language_model.layers.3.post_attention_layernorm",
    "model.language_model.layers.3.mlp.gate",
    "model.language_model.layers.3.self_attn.k_norm",
    "model.language_model.layers.3.self_attn.q_norm",
    "model.language_model.layers.4.input_layernorm",
    "model.language_model.layers.4.mlp.shared_expert_gate",
    "model.language_model.layers.4.post_attention_layernorm",
    "model.language_model.layers.4.mlp.gate",
    "model.language_model.layers.4.linear_attn.A_log",
    "model.language_model.layers.4.linear_attn.conv1d",
    "model.language_model.layers.4.linear_attn.dt_bias",
    "model.language_model.layers.4.linear_attn.in_proj_ba",
    "model.language_model.layers.4.linear_attn.in_proj_b",
    "model.language_model.layers.4.linear_attn.in_proj_a",
    "model.language_model.layers.4.linear_attn.norm",
    "model.language_model.layers.5.input_layernorm",
    "model.language_model.layers.5.mlp.shared_expert_gate",
    "model.language_model.layers.5.post_attention_layernorm",
    "model.language_model.layers.5.mlp.gate",
    "model.language_model.layers.5.linear_attn.A_log",
    "model.language_model.layers.5.linear_attn.conv1d",
    "model.language_model.layers.5.linear_attn.dt_bias",
    "model.language_model.layers.5.linear_attn.in_proj_ba",
    "model.language_model.layers.5.linear_attn.in_proj_b",
    "model.language_model.layers.5.linear_attn.in_proj_a",
    "model.language_model.layers.5.linear_attn.norm",
    "model.language_model.layers.6.input_layernorm",
    "model.language_model.layers.6.mlp.shared_expert_gate",
    "model.language_model.layers.6.post_attention_layernorm",
    "model.language_model.layers.6.mlp.gate",
    "model.language_model.layers.6.linear_attn.A_log",
    "model.language_model.layers.6.linear_attn.conv1d",
    "model.language_model.layers.6.linear_attn.dt_bias",
    "model.language_model.layers.6.linear_attn.in_proj_ba",
    "model.language_model.layers.6.linear_attn.in_proj_b",
    "model.language_model.layers.6.linear_attn.in_proj_a",
    "model.language_model.layers.6.linear_attn.norm",
    "model.language_model.layers.7.input_layernorm",
    "model.language_model.layers.7.mlp.shared_expert_gate",
    "model.language_model.layers.7.post_attention_layernorm",
    "model.language_model.layers.7.mlp.gate",
    "model.language_model.layers.7.self_attn.k_norm",
    "model.language_model.layers.7.self_attn.q_norm",
    "model.language_model.layers.8.input_layernorm",
    "model.language_model.layers.8.mlp.shared_expert_gate",
    "model.language_model.layers.8.post_attention_layernorm",
    "model.language_model.layers.8.mlp.gate",
    "model.language_model.layers.8.linear_attn.A_log",
    "model.language_model.layers.8.linear_attn.conv1d",
    "model.language_model.layers.8.linear_attn.dt_bias",
    "model.language_model.layers.8.linear_attn.in_proj_ba",
    "model.language_model.layers.8.linear_attn.in_proj_b",
    "model.language_model.layers.8.linear_attn.in_proj_a",
    "model.language_model.layers.8.linear_attn.norm",
    "model.language_model.layers.9.input_layernorm",
    "model.language_model.layers.9.mlp.shared_expert_gate",
    "model.language_model.layers.9.post_attention_layernorm",
    "model.language_model.layers.9.mlp.gate",
    "model.language_model.layers.9.linear_attn.A_log",
    "model.language_model.layers.9.linear_attn.conv1d",
    "model.language_model.layers.9.linear_attn.dt_bias",
    "model.language_model.layers.9.linear_attn.in_proj_ba",
    "model.language_model.layers.9.linear_attn.in_proj_b",
    "model.language_model.layers.9.linear_attn.in_proj_a",
    "model.language_model.layers.9.linear_attn.norm",
    "model.language_model.layers.10.input_layernorm",
    "model.language_model.layers.10.mlp.shared_expert_gate",
    "model.language_model.layers.10.post_attention_layernorm",
    "model.language_model.layers.10.mlp.gate",
    "model.language_model.layers.10.linear_attn.A_log",
    "model.language_model.layers.10.linear_attn.conv1d",
    "model.language_model.layers.10.linear_attn.dt_bias",
    "model.language_model.layers.10.linear_attn.in_proj_ba",
    "model.language_model.layers.10.linear_attn.in_proj_b",
    "model.language_model.layers.10.linear_attn.in_proj_a",
    "model.language_model.layers.10.linear_attn.norm",
    "model.language_model.layers.11.input_layernorm",
    "model.language_model.layers.11.mlp.shared_expert_gate",
    "model.language_model.layers.11.post_attention_layernorm",
    "model.language_model.layers.11.mlp.gate",
    "model.language_model.layers.11.self_attn.k_norm",
    "model.language_model.layers.11.self_attn.q_norm",
    "model.language_model.layers.12.input_layernorm",
    "model.language_model.layers.12.mlp.shared_expert_gate",
    "model.language_model.layers.12.post_attention_layernorm",
    "model.language_model.layers.12.mlp.gate",
    "model.language_model.layers.12.linear_attn.A_log",
    "model.language_model.layers.12.linear_attn.conv1d",
    "model.language_model.layers.12.linear_attn.dt_bias",
    "model.language_model.layers.12.linear_attn.in_proj_ba",
    "model.language_model.layers.12.linear_attn.in_proj_b",
    "model.language_model.layers.12.linear_attn.in_proj_a",
    "model.language_model.layers.12.linear_attn.norm",
    "model.language_model.layers.13.input_layernorm",
    "model.language_model.layers.13.mlp.shared_expert_gate",
    "model.language_model.layers.13.post_attention_layernorm",
    "model.language_model.layers.13.mlp.gate",
    "model.language_model.layers.13.linear_attn.A_log",
    "model.language_model.layers.13.linear_attn.conv1d",
    "model.language_model.layers.13.linear_attn.dt_bias",
    "model.language_model.layers.13.linear_attn.in_proj_ba",
    "model.language_model.layers.13.linear_attn.in_proj_b",
    "model.language_model.layers.13.linear_attn.in_proj_a",
    "model.language_model.layers.13.linear_attn.norm",
    "model.language_model.layers.14.input_layernorm",
    "model.language_model.layers.14.mlp.shared_expert_gate",
    "model.language_model.layers.14.post_attention_layernorm",
    "model.language_model.layers.14.mlp.gate",
    "model.language_model.layers.14.linear_attn.A_log",
    "model.language_model.layers.14.linear_attn.conv1d",
    "model.language_model.layers.14.linear_attn.dt_bias",
    "model.language_model.layers.14.linear_attn.in_proj_ba",
    "model.language_model.layers.14.linear_attn.in_proj_b",
    "model.language_model.layers.14.linear_attn.in_proj_a",
    "model.language_model.layers.14.linear_attn.norm",
    "model.language_model.layers.15.input_layernorm",
    "model.language_model.layers.15.mlp.shared_expert_gate",
    "model.language_model.layers.15.post_attention_layernorm",
    "model.language_model.layers.15.mlp.gate",
    "model.language_model.layers.15.self_attn.k_norm",
    "model.language_model.layers.15.self_attn.q_norm",
    "model.language_model.layers.16.input_layernorm",
    "model.language_model.layers.16.mlp.shared_expert_gate",
    "model.language_model.layers.16.post_attention_layernorm",
    "model.language_model.layers.16.mlp.gate",
    "model.language_model.layers.16.linear_attn.A_log",
    "model.language_model.layers.16.linear_attn.conv1d",
    "model.language_model.layers.16.linear_attn.dt_bias",
    "model.language_model.layers.16.linear_attn.in_proj_ba",
    "model.language_model.layers.16.linear_attn.in_proj_b",
    "model.language_model.layers.16.linear_attn.in_proj_a",
    "model.language_model.layers.16.linear_attn.norm",
    "model.language_model.layers.17.input_layernorm",
    "model.language_model.layers.17.mlp.shared_expert_gate",
    "model.language_model.layers.17.post_attention_layernorm",
    "model.language_model.layers.17.mlp.gate",
    "model.language_model.layers.17.linear_attn.A_log",
    "model.language_model.layers.17.linear_attn.conv1d",
    "model.language_model.layers.17.linear_attn.dt_bias",
    "model.language_model.layers.17.linear_attn.in_proj_ba",
    "model.language_model.layers.17.linear_attn.in_proj_b",
    "model.language_model.layers.17.linear_attn.in_proj_a",
    "model.language_model.layers.17.linear_attn.norm",
    "model.language_model.layers.18.input_layernorm",
    "model.language_model.layers.18.mlp.shared_expert_gate",
    "model.language_model.layers.18.post_attention_layernorm",
    "model.language_model.layers.18.mlp.gate",
    "model.language_model.layers.18.linear_attn.A_log",
    "model.language_model.layers.18.linear_attn.conv1d",
    "model.language_model.layers.18.linear_attn.dt_bias",
    "model.language_model.layers.18.linear_attn.in_proj_ba",
    "model.language_model.layers.18.linear_attn.in_proj_b",
    "model.language_model.layers.18.linear_attn.in_proj_a",
    "model.language_model.layers.18.linear_attn.norm",
    "model.language_model.layers.19.input_layernorm",
    "model.language_model.layers.19.mlp.shared_expert_gate",
    "model.language_model.layers.19.post_attention_layernorm",
    "model.language_model.layers.19.mlp.gate",
    "model.language_model.layers.19.self_attn.k_norm",
    "model.language_model.layers.19.self_attn.q_norm",
    "model.language_model.layers.20.input_layernorm",
    "model.language_model.layers.20.mlp.shared_expert_gate",
    "model.language_model.layers.20.post_attention_layernorm",
    "model.language_model.layers.20.mlp.gate",
    "model.language_model.layers.20.linear_attn.A_log",
    "model.language_model.layers.20.linear_attn.conv1d",
    "model.language_model.layers.20.linear_attn.dt_bias",
    "model.language_model.layers.20.linear_attn.in_proj_ba",
    "model.language_model.layers.20.linear_attn.in_proj_b",
    "model.language_model.layers.20.linear_attn.in_proj_a",
    "model.language_model.layers.20.linear_attn.norm",
    "model.language_model.layers.21.input_layernorm",
    "model.language_model.layers.21.mlp.shared_expert_gate",
    "model.language_model.layers.21.post_attention_layernorm",
    "model.language_model.layers.21.mlp.gate",
    "model.language_model.layers.21.linear_attn.A_log",
    "model.language_model.layers.21.linear_attn.conv1d",
    "model.language_model.layers.21.linear_attn.dt_bias",
    "model.language_model.layers.21.linear_attn.in_proj_ba",
    "model.language_model.layers.21.linear_attn.in_proj_b",
    "model.language_model.layers.21.linear_attn.in_proj_a",
    "model.language_model.layers.21.linear_attn.norm",
    "model.language_model.layers.22.input_layernorm",
    "model.language_model.layers.22.mlp.shared_expert_gate",
    "model.language_model.layers.22.post_attention_layernorm",
    "model.language_model.layers.22.mlp.gate",
    "model.language_model.layers.22.linear_attn.A_log",
    "model.language_model.layers.22.linear_attn.conv1d",
    "model.language_model.layers.22.linear_attn.dt_bias",
    "model.language_model.layers.22.linear_attn.in_proj_ba",
    "model.language_model.layers.22.linear_attn.in_proj_b",
    "model.language_model.layers.22.linear_attn.in_proj_a",
    "model.language_model.layers.22.linear_attn.norm",
    "model.language_model.layers.23.input_layernorm",
    "model.language_model.layers.23.mlp.shared_expert_gate",
    "model.language_model.layers.23.post_attention_layernorm",
    "model.language_model.layers.23.mlp.gate",
    "model.language_model.layers.23.self_attn.k_norm",
    "model.language_model.layers.23.self_attn.q_norm",
    "model.language_model.layers.24.input_layernorm",
    "model.language_model.layers.24.mlp.shared_expert_gate",
    "model.language_model.layers.24.post_attention_layernorm",
    "model.language_model.layers.24.mlp.gate",
    "model.language_model.layers.24.linear_attn.A_log",
    "model.language_model.layers.24.linear_attn.conv1d",
    "model.language_model.layers.24.linear_attn.dt_bias",
    "model.language_model.layers.24.linear_attn.in_proj_ba",
    "model.language_model.layers.24.linear_attn.in_proj_b",
    "model.language_model.layers.24.linear_attn.in_proj_a",
    "model.language_model.layers.24.linear_attn.norm",
    "model.language_model.layers.25.input_layernorm",
    "model.language_model.layers.25.mlp.shared_expert_gate",
    "model.language_model.layers.25.post_attention_layernorm",
    "model.language_model.layers.25.mlp.gate",
    "model.language_model.layers.25.linear_attn.A_log",
    "model.language_model.layers.25.linear_attn.conv1d",
    "model.language_model.layers.25.linear_attn.dt_bias",
    "model.language_model.layers.25.linear_attn.in_proj_ba",
    "model.language_model.layers.25.linear_attn.in_proj_b",
    "model.language_model.layers.25.linear_attn.in_proj_a",
    "model.language_model.layers.25.linear_attn.norm",
    "model.language_model.layers.26.input_layernorm",
    "model.language_model.layers.26.mlp.shared_expert_gate",
    "model.language_model.layers.26.post_attention_layernorm",
    "model.language_model.layers.26.mlp.gate",
    "model.language_model.layers.26.linear_attn.A_log",
    "model.language_model.layers.26.linear_attn.conv1d",
    "model.language_model.layers.26.linear_attn.dt_bias",
    "model.language_model.layers.26.linear_attn.in_proj_ba",
    "model.language_model.layers.26.linear_attn.in_proj_b",
    "model.language_model.layers.26.linear_attn.in_proj_a",
    "model.language_model.layers.26.linear_attn.norm",
    "model.language_model.layers.27.input_layernorm",
    "model.language_model.layers.27.mlp.shared_expert_gate",
    "model.language_model.layers.27.post_attention_layernorm",
    "model.language_model.layers.27.mlp.gate",
    "model.language_model.layers.27.self_attn.k_norm",
    "model.language_model.layers.27.self_attn.q_norm",
    "model.language_model.layers.28.input_layernorm",
    "model.language_model.layers.28.mlp.shared_expert_gate",
    "model.language_model.layers.28.post_attention_layernorm",
    "model.language_model.layers.28.mlp.gate",
    "model.language_model.layers.28.linear_attn.A_log",
    "model.language_model.layers.28.linear_attn.conv1d",
    "model.language_model.layers.28.linear_attn.dt_bias",
    "model.language_model.layers.28.linear_attn.in_proj_ba",
    "model.language_model.layers.28.linear_attn.in_proj_b",
    "model.language_model.layers.28.linear_attn.in_proj_a",
    "model.language_model.layers.28.linear_attn.norm",
    "model.language_model.layers.29.input_layernorm",
    "model.language_model.layers.29.mlp.shared_expert_gate",
    "model.language_model.layers.29.post_attention_layernorm",
    "model.language_model.layers.29.mlp.gate",
    "model.language_model.layers.29.linear_attn.A_log",
    "model.language_model.layers.29.linear_attn.conv1d",
    "model.language_model.layers.29.linear_attn.dt_bias",
    "model.language_model.layers.29.linear_attn.in_proj_ba",
    "model.language_model.layers.29.linear_attn.in_proj_b",
    "model.language_model.layers.29.linear_attn.in_proj_a",
    "model.language_model.layers.29.linear_attn.norm",
    "model.language_model.layers.30.input_layernorm",
    "model.language_model.layers.30.mlp.shared_expert_gate",
    "model.language_model.layers.30.post_attention_layernorm",
    "model.language_model.layers.30.mlp.gate",
    "model.language_model.layers.30.linear_attn.A_log",
    "model.language_model.layers.30.linear_attn.conv1d",
    "model.language_model.layers.30.linear_attn.dt_bias",
    "model.language_model.layers.30.linear_attn.in_proj_ba",
    "model.language_model.layers.30.linear_attn.in_proj_b",
    "model.language_model.layers.30.linear_attn.in_proj_a",
    "model.language_model.layers.30.linear_attn.norm",
    "model.language_model.layers.31.input_layernorm",
    "model.language_model.layers.31.mlp.shared_expert_gate",
    "model.language_model.layers.31.post_attention_layernorm",
    "model.language_model.layers.31.mlp.gate",
    "model.language_model.layers.31.self_attn.k_norm",
    "model.language_model.layers.31.self_attn.q_norm",
    "model.language_model.layers.32.input_layernorm",
    "model.language_model.layers.32.mlp.shared_expert_gate",
    "model.language_model.layers.32.post_attention_layernorm",
    "model.language_model.layers.32.mlp.gate",
    "model.language_model.layers.32.linear_attn.A_log",
    "model.language_model.layers.32.linear_attn.conv1d",
    "model.language_model.layers.32.linear_attn.dt_bias",
    "model.language_model.layers.32.linear_attn.in_proj_ba",
    "model.language_model.layers.32.linear_attn.in_proj_b",
    "model.language_model.layers.32.linear_attn.in_proj_a",
    "model.language_model.layers.32.linear_attn.norm",
    "model.language_model.layers.33.input_layernorm",
    "model.language_model.layers.33.mlp.shared_expert_gate",
    "model.language_model.layers.33.post_attention_layernorm",
    "model.language_model.layers.33.mlp.gate",
    "model.language_model.layers.33.linear_attn.A_log",
    "model.language_model.layers.33.linear_attn.conv1d",
    "model.language_model.layers.33.linear_attn.dt_bias",
    "model.language_model.layers.33.linear_attn.in_proj_ba",
    "model.language_model.layers.33.linear_attn.in_proj_b",
    "model.language_model.layers.33.linear_attn.in_proj_a",
    "model.language_model.layers.33.linear_attn.norm",
    "model.language_model.layers.34.input_layernorm",
    "model.language_model.layers.34.mlp.shared_expert_gate",
    "model.language_model.layers.34.post_attention_layernorm",
    "model.language_model.layers.34.mlp.gate",
    "model.language_model.layers.34.linear_attn.A_log",
    "model.language_model.layers.34.linear_attn.conv1d",
    "model.language_model.layers.34.linear_attn.dt_bias",
    "model.language_model.layers.34.linear_attn.in_proj_ba",
    "model.language_model.layers.34.linear_attn.in_proj_b",
    "model.language_model.layers.34.linear_attn.in_proj_a",
    "model.language_model.layers.34.linear_attn.norm",
    "model.language_model.layers.35.input_layernorm",
    "model.language_model.layers.35.mlp.shared_expert_gate",
    "model.language_model.layers.35.post_attention_layernorm",
    "model.language_model.layers.35.mlp.gate",
    "model.language_model.layers.35.self_attn.k_norm",
    "model.language_model.layers.35.self_attn.q_norm",
    "model.language_model.layers.36.input_layernorm",
    "model.language_model.layers.36.mlp.shared_expert_gate",
    "model.language_model.layers.36.post_attention_layernorm",
    "model.language_model.layers.36.mlp.gate",
    "model.language_model.layers.36.linear_attn.A_log",
    "model.language_model.layers.36.linear_attn.conv1d",
    "model.language_model.layers.36.linear_attn.dt_bias",
    "model.language_model.layers.36.linear_attn.in_proj_ba",
    "model.language_model.layers.36.linear_attn.in_proj_b",
    "model.language_model.layers.36.linear_attn.in_proj_a",
    "model.language_model.layers.36.linear_attn.norm",
    "model.language_model.layers.37.input_layernorm",
    "model.language_model.layers.37.mlp.shared_expert_gate",
    "model.language_model.layers.37.post_attention_layernorm",
    "model.language_model.layers.37.mlp.gate",
    "model.language_model.layers.37.linear_attn.A_log",
    "model.language_model.layers.37.linear_attn.conv1d",
    "model.language_model.layers.37.linear_attn.dt_bias",
    "model.language_model.layers.37.linear_attn.in_proj_ba",
    "model.language_model.layers.37.linear_attn.in_proj_b",
    "model.language_model.layers.37.linear_attn.in_proj_a",
    "model.language_model.layers.37.linear_attn.norm",
    "model.language_model.layers.38.input_layernorm",
    "model.language_model.layers.38.mlp.shared_expert_gate",
    "model.language_model.layers.38.post_attention_layernorm",
    "model.language_model.layers.38.mlp.gate",
    "model.language_model.layers.38.linear_attn.A_log",
    "model.language_model.layers.38.linear_attn.conv1d",
    "model.language_model.layers.38.linear_attn.dt_bias",
    "model.language_model.layers.38.linear_attn.in_proj_ba",
    "model.language_model.layers.38.linear_attn.in_proj_b",
    "model.language_model.layers.38.linear_attn.in_proj_a",
    "model.language_model.layers.38.linear_attn.norm",
    "model.language_model.layers.39.input_layernorm",
    "model.language_model.layers.39.mlp.shared_expert_gate",
    "model.language_model.layers.39.post_attention_layernorm",
    "model.language_model.layers.39.mlp.gate",
    "model.language_model.layers.39.self_attn.k_norm",
    "model.language_model.layers.39.self_attn.q_norm",
    "mtp.layers.0.input_layernorm",
    "mtp.layers.0.mlp.gate",
    "mtp.layers.0.mlp.shared_expert_gate",
    "mtp.layers.0.post_attention_layernorm",
    "mtp.layers.0.self_attn.k_norm",
    "mtp.layers.0.self_attn.q_norm",
    "mtp.fc",
    "mtp.norm",
    "mtp.pre_fc_norm_embedding",
    "mtp.pre_fc_norm_hidden"
  ],
  "weight_block_size": [
    128,
    128
  ]
}

```

122B-A10B FP8:

```text

/inspire/qb-ilm2/project/new-paradigm-of-manipulation/shixiaoxin-253107030017/simpletool_eval_20261110/eval_views/p005-122b-fp8

```

版本：`{"torch": "2.14.0+cu130", "vllm": "0.33.0", "cuda": "13.0"}`；quantization：

```json

{
  "quant_method": "fp8",
  "activation_scheme": "dynamic",
  "weight_per_tensor": false,
  "act_per_tensor": false,
  "weight_block_size": [
    128,
    128
  ],
  "modules_to_not_convert": [
    "lm_head",
    "model.language_model.embed_tokens",
    "model.language_model.layers.0.linear_attn.conv1d",
    "model.language_model.layers.0.linear_attn.in_proj_a",
    "model.language_model.layers.0.linear_attn.in_proj_b",
    "model.language_model.layers.0.mlp.gate",
    "model.language_model.layers.0.mlp.shared_expert_gate",
    "model.language_model.layers.1.linear_attn.conv1d",
    "model.language_model.layers.1.linear_attn.in_proj_a",
    "model.language_model.layers.1.linear_attn.in_proj_b",
    "model.language_model.layers.1.mlp.gate",
    "model.language_model.layers.1.mlp.shared_expert_gate",
    "model.language_model.layers.10.linear_attn.conv1d",
    "model.language_model.layers.10.linear_attn.in_proj_a",
    "model.language_model.layers.10.linear_attn.in_proj_b",
    "model.language_model.layers.10.mlp.gate",
    "model.language_model.layers.10.mlp.shared_expert_gate",
    "model.language_model.layers.11.mlp.gate",
    "model.language_model.layers.11.mlp.shared_expert_gate",
    "model.language_model.layers.12.linear_attn.conv1d",
    "model.language_model.layers.12.linear_attn.in_proj_a",
    "model.language_model.layers.12.linear_attn.in_proj_b",
    "model.language_model.layers.12.mlp.gate",
    "model.language_model.layers.12.mlp.shared_expert_gate",
    "model.language_model.layers.13.linear_attn.conv1d",
    "model.language_model.layers.13.linear_attn.in_proj_a",
    "model.language_model.layers.13.linear_attn.in_proj_b",
    "model.language_model.layers.13.mlp.gate",
    "model.language_model.layers.13.mlp.shared_expert_gate",
    "model.language_model.layers.14.linear_attn.conv1d",
    "model.language_model.layers.14.linear_attn.in_proj_a",
    "model.language_model.layers.14.linear_attn.in_proj_b",
    "model.language_model.layers.14.mlp.gate",
    "model.language_model.layers.14.mlp.shared_expert_gate",
    "model.language_model.layers.15.mlp.gate",
    "model.language_model.layers.15.mlp.shared_expert_gate",
    "model.language_model.layers.16.linear_attn.conv1d",
    "model.language_model.layers.16.linear_attn.in_proj_a",
    "model.language_model.layers.16.linear_attn.in_proj_b",
    "model.language_model.layers.16.mlp.gate",
    "model.language_model.layers.16.mlp.shared_expert_gate",
    "model.language_model.layers.17.linear_attn.conv1d",
    "model.language_model.layers.17.linear_attn.in_proj_a",
    "model.language_model.layers.17.linear_attn.in_proj_b",
    "model.language_model.layers.17.mlp.gate",
    "model.language_model.layers.17.mlp.shared_expert_gate",
    "model.language_model.layers.18.linear_attn.conv1d",
    "model.language_model.layers.18.linear_attn.in_proj_a",
    "model.language_model.layers.18.linear_attn.in_proj_b",
    "model.language_model.layers.18.mlp.gate",
    "model.language_model.layers.18.mlp.shared_expert_gate",
    "model.language_model.layers.19.mlp.gate",
    "model.language_model.layers.19.mlp.shared_expert_gate",
    "model.language_model.layers.2.linear_attn.conv1d",
    "model.language_model.layers.2.linear_attn.in_proj_a",
    "model.language_model.layers.2.linear_attn.in_proj_b",
    "model.language_model.layers.2.mlp.gate",
    "model.language_model.layers.2.mlp.shared_expert_gate",
    "model.language_model.layers.20.linear_attn.conv1d",
    "model.language_model.layers.20.linear_attn.in_proj_a",
    "model.language_model.layers.20.linear_attn.in_proj_b",
    "model.language_model.layers.20.mlp.gate",
    "model.language_model.layers.20.mlp.shared_expert_gate",
    "model.language_model.layers.21.linear_attn.conv1d",
    "model.language_model.layers.21.linear_attn.in_proj_a",
    "model.language_model.layers.21.linear_attn.in_proj_b",
    "model.language_model.layers.21.mlp.gate",
    "model.language_model.layers.21.mlp.shared_expert_gate",
    "model.language_model.layers.22.linear_attn.conv1d",
    "model.language_model.layers.22.linear_attn.in_proj_a",
    "model.language_model.layers.22.linear_attn.in_proj_b",
    "model.language_model.layers.22.mlp.gate",
    "model.language_model.layers.22.mlp.shared_expert_gate",
    "model.language_model.layers.23.mlp.gate",
    "model.language_model.layers.23.mlp.shared_expert_gate",
    "model.language_model.layers.24.linear_attn.conv1d",
    "model.language_model.layers.24.linear_attn.in_proj_a",
    "model.language_model.layers.24.linear_attn.in_proj_b",
    "model.language_model.layers.24.mlp.gate",
    "model.language_model.layers.24.mlp.shared_expert_gate",
    "model.language_model.layers.25.linear_attn.conv1d",
    "model.language_model.layers.25.linear_attn.in_proj_a",
    "model.language_model.layers.25.linear_attn.in_proj_b",
    "model.language_model.layers.25.mlp.gate",
    "model.language_model.layers.25.mlp.shared_expert_gate",
    "model.language_model.layers.26.linear_attn.conv1d",
    "model.language_model.layers.26.linear_attn.in_proj_a",
    "model.language_model.layers.26.linear_attn.in_proj_b",
    "model.language_model.layers.26.mlp.gate",
    "model.language_model.layers.26.mlp.shared_expert_gate",
    "model.language_model.layers.27.mlp.gate",
    "model.language_model.layers.27.mlp.shared_expert_gate",
    "model.language_model.layers.28.linear_attn.conv1d",
    "model.language_model.layers.28.linear_attn.in_proj_a",
    "model.language_model.layers.28.linear_attn.in_proj_b",
    "model.language_model.layers.28.mlp.gate",
    "model.language_model.layers.28.mlp.shared_expert_gate",
    "model.language_model.layers.29.linear_attn.conv1d",
    "model.language_model.layers.29.linear_attn.in_proj_a",
    "model.language_model.layers.29.linear_attn.in_proj_b",
    "model.language_model.layers.29.mlp.gate",
    "model.language_model.layers.29.mlp.shared_expert_gate",
    "model.language_model.layers.3.mlp.gate",
    "model.language_model.layers.3.mlp.shared_expert_gate",
    "model.language_model.layers.30.linear_attn.conv1d",
    "model.language_model.layers.30.linear_attn.in_proj_a",
    "model.language_model.layers.30.linear_attn.in_proj_b",
    "model.language_model.layers.30.mlp.gate",
    "model.language_model.layers.30.mlp.shared_expert_gate",
    "model.language_model.layers.31.mlp.gate",
    "model.language_model.layers.31.mlp.shared_expert_gate",
    "model.language_model.layers.32.linear_attn.conv1d",
    "model.language_model.layers.32.linear_attn.in_proj_a",
    "model.language_model.layers.32.linear_attn.in_proj_b",
    "model.language_model.layers.32.mlp.gate",
    "model.language_model.layers.32.mlp.shared_expert_gate",
    "model.language_model.layers.33.linear_attn.conv1d",
    "model.language_model.layers.33.linear_attn.in_proj_a",
    "model.language_model.layers.33.linear_attn.in_proj_b",
    "model.language_model.layers.33.mlp.gate",
    "model.language_model.layers.33.mlp.shared_expert_gate",
    "model.language_model.layers.34.linear_attn.conv1d",
    "model.language_model.layers.34.linear_attn.in_proj_a",
    "model.language_model.layers.34.linear_attn.in_proj_b",
    "model.language_model.layers.34.mlp.gate",
    "model.language_model.layers.34.mlp.shared_expert_gate",
    "model.language_model.layers.35.mlp.gate",
    "model.language_model.layers.35.mlp.shared_expert_gate",
    "model.language_model.layers.36.linear_attn.conv1d",
    "model.language_model.layers.36.linear_attn.in_proj_a",
    "model.language_model.layers.36.linear_attn.in_proj_b",
    "model.language_model.layers.36.mlp.gate",
    "model.language_model.layers.36.mlp.shared_expert_gate",
    "model.language_model.layers.37.linear_attn.conv1d",
    "model.language_model.layers.37.linear_attn.in_proj_a",
    "model.language_model.layers.37.linear_attn.in_proj_b",
    "model.language_model.layers.37.mlp.gate",
    "model.language_model.layers.37.mlp.shared_expert_gate",
    "model.language_model.layers.38.linear_attn.conv1d",
    "model.language_model.layers.38.linear_attn.in_proj_a",
    "model.language_model.layers.38.linear_attn.in_proj_b",
    "model.language_model.layers.38.mlp.gate",
    "model.language_model.layers.38.mlp.shared_expert_gate",
    "model.language_model.layers.39.mlp.gate",
    "model.language_model.layers.39.mlp.shared_expert_gate",
    "model.language_model.layers.4.linear_attn.conv1d",
    "model.language_model.layers.4.linear_attn.in_proj_a",
    "model.language_model.layers.4.linear_attn.in_proj_b",
    "model.language_model.layers.4.mlp.gate",
    "model.language_model.layers.4.mlp.shared_expert_gate",
    "model.language_model.layers.40.linear_attn.conv1d",
    "model.language_model.layers.40.linear_attn.in_proj_a",
    "model.language_model.layers.40.linear_attn.in_proj_b",
    "model.language_model.layers.40.mlp.gate",
    "model.language_model.layers.40.mlp.shared_expert_gate",
    "model.language_model.layers.41.linear_attn.conv1d",
    "model.language_model.layers.41.linear_attn.in_proj_a",
    "model.language_model.layers.41.linear_attn.in_proj_b",
    "model.language_model.layers.41.mlp.gate",
    "model.language_model.layers.41.mlp.shared_expert_gate",
    "model.language_model.layers.42.linear_attn.conv1d",
    "model.language_model.layers.42.linear_attn.in_proj_a",
    "model.language_model.layers.42.linear_attn.in_proj_b",
    "model.language_model.layers.42.mlp.gate",
    "model.language_model.layers.42.mlp.shared_expert_gate",
    "model.language_model.layers.43.mlp.gate",
    "model.language_model.layers.43.mlp.shared_expert_gate",
    "model.language_model.layers.44.linear_attn.conv1d",
    "model.language_model.layers.44.linear_attn.in_proj_a",
    "model.language_model.layers.44.linear_attn.in_proj_b",
    "model.language_model.layers.44.mlp.gate",
    "model.language_model.layers.44.mlp.shared_expert_gate",
    "model.language_model.layers.45.linear_attn.conv1d",
    "model.language_model.layers.45.linear_attn.in_proj_a",
    "model.language_model.layers.45.linear_attn.in_proj_b",
    "model.language_model.layers.45.mlp.gate",
    "model.language_model.layers.45.mlp.shared_expert_gate",
    "model.language_model.layers.46.linear_attn.conv1d",
    "model.language_model.layers.46.linear_attn.in_proj_a",
    "model.language_model.layers.46.linear_attn.in_proj_b",
    "model.language_model.layers.46.mlp.gate",
    "model.language_model.layers.46.mlp.shared_expert_gate",
    "model.language_model.layers.47.mlp.gate",
    "model.language_model.layers.47.mlp.shared_expert_gate",
    "model.language_model.layers.5.linear_attn.conv1d",
    "model.language_model.layers.5.linear_attn.in_proj_a",
    "model.language_model.layers.5.linear_attn.in_proj_b",
    "model.language_model.layers.5.mlp.gate",
    "model.language_model.layers.5.mlp.shared_expert_gate",
    "model.language_model.layers.6.linear_attn.conv1d",
    "model.language_model.layers.6.linear_attn.in_proj_a",
    "model.language_model.layers.6.linear_attn.in_proj_b",
    "model.language_model.layers.6.mlp.gate",
    "model.language_model.layers.6.mlp.shared_expert_gate",
    "model.language_model.layers.7.mlp.gate",
    "model.language_model.layers.7.mlp.shared_expert_gate",
    "model.language_model.layers.8.linear_attn.conv1d",
    "model.language_model.layers.8.linear_attn.in_proj_a",
    "model.language_model.layers.8.linear_attn.in_proj_b",
    "model.language_model.layers.8.mlp.gate",
    "model.language_model.layers.8.mlp.shared_expert_gate",
    "model.language_model.layers.9.linear_attn.conv1d",
    "model.language_model.layers.9.linear_attn.in_proj_a",
    "model.language_model.layers.9.linear_attn.in_proj_b",
    "model.language_model.layers.9.mlp.gate",
    "model.language_model.layers.9.mlp.shared_expert_gate",
    "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.mlp.linear_fc1",
    "model.visual.blocks.0.mlp.linear_fc2",
    "model.visual.blocks.1.attn.proj",
    "model.visual.blocks.1.attn.qkv",
    "model.visual.blocks.1.mlp.linear_fc1",
    "model.visual.blocks.1.mlp.linear_fc2",
    "model.visual.blocks.10.attn.proj",
    "model.visual.blocks.10.attn.qkv",
    "model.visual.blocks.10.mlp.linear_fc1",
    "model.visual.blocks.10.mlp.linear_fc2",
    "model.visual.blocks.11.attn.proj",
    "model.visual.blocks.11.attn.qkv",
    "model.visual.blocks.11.mlp.linear_fc1",
    "model.visual.blocks.11.mlp.linear_fc2",
    "model.visual.blocks.12.attn.proj",
    "model.visual.blocks.12.attn.qkv",
    "model.visual.blocks.12.mlp.linear_fc1",
    "model.visual.blocks.12.mlp.linear_fc2",
    "model.visual.blocks.13.attn.proj",
    "model.visual.blocks.13.attn.qkv",
    "model.visual.blocks.13.mlp.linear_fc1",
    "model.visual.blocks.13.mlp.linear_fc2",
    "model.visual.blocks.14.attn.proj",
    "model.visual.blocks.14.attn.qkv",
    "model.visual.blocks.14.mlp.linear_fc1",
    "model.visual.blocks.14.mlp.linear_fc2",
    "model.visual.blocks.15.attn.proj",
    "model.visual.blocks.15.attn.qkv",
    "model.visual.blocks.15.mlp.linear_fc1",
    "model.visual.blocks.15.mlp.linear_fc2",
    "model.visual.blocks.16.attn.proj",
    "model.visual.blocks.16.attn.qkv",
    "model.visual.blocks.16.mlp.linear_fc1",
    "model.visual.blocks.16.mlp.linear_fc2",
    "model.visual.blocks.17.attn.proj",
    "model.visual.blocks.17.attn.qkv",
    "model.visual.blocks.17.mlp.linear_fc1",
    "model.visual.blocks.17.mlp.linear_fc2",
    "model.visual.blocks.18.attn.proj",
    "model.visual.blocks.18.attn.qkv",
    "model.visual.blocks.18.mlp.linear_fc1",
    "model.visual.blocks.18.mlp.linear_fc2",
    "model.visual.blocks.19.attn.proj",
    "model.visual.blocks.19.attn.qkv",
    "model.visual.blocks.19.mlp.linear_fc1",
    "model.visual.blocks.19.mlp.linear_fc2",
    "model.visual.blocks.2.attn.proj",
    "model.visual.blocks.2.attn.qkv",
    "model.visual.blocks.2.mlp.linear_fc1",
    "model.visual.blocks.2.mlp.linear_fc2",
    "model.visual.blocks.20.attn.proj",
    "model.visual.blocks.20.attn.qkv",
    "model.visual.blocks.20.mlp.linear_fc1",
    "model.visual.blocks.20.mlp.linear_fc2",
    "model.visual.blocks.21.attn.proj",
    "model.visual.blocks.21.attn.qkv",
    "model.visual.blocks.21.mlp.linear_fc1",
    "model.visual.blocks.21.mlp.linear_fc2",
    "model.visual.blocks.22.attn.proj",
    "model.visual.blocks.22.attn.qkv",
    "model.visual.blocks.22.mlp.linear_fc1",
    "model.visual.blocks.22.mlp.linear_fc2",
    "model.visual.blocks.23.attn.proj",
    "model.visual.blocks.23.attn.qkv",
    "model.visual.blocks.23.mlp.linear_fc1",
    "model.visual.blocks.23.mlp.linear_fc2",
    "model.visual.blocks.24.attn.proj",
    "model.visual.blocks.24.attn.qkv",
    "model.visual.blocks.24.mlp.linear_fc1",
    "model.visual.blocks.24.mlp.linear_fc2",
    "model.visual.blocks.25.attn.proj",
    "model.visual.blocks.25.attn.qkv",
    "model.visual.blocks.25.mlp.linear_fc1",
    "model.visual.blocks.25.mlp.linear_fc2",
    "model.visual.blocks.26.attn.proj",
    "model.visual.blocks.26.attn.qkv",
    "model.visual.blocks.26.mlp.linear_fc1",
    "model.visual.blocks.26.mlp.linear_fc2",
    "model.visual.blocks.3.attn.proj",
    "model.visual.blocks.3.attn.qkv",
    "model.visual.blocks.3.mlp.linear_fc1",
    "model.visual.blocks.3.mlp.linear_fc2",
    "model.visual.blocks.4.attn.proj",
    "model.visual.blocks.4.attn.qkv",
    "model.visual.blocks.4.mlp.linear_fc1",
    "model.visual.blocks.4.mlp.linear_fc2",
    "model.visual.blocks.5.attn.proj",
    "model.visual.blocks.5.attn.qkv",
    "model.visual.blocks.5.mlp.linear_fc1",
    "model.visual.blocks.5.mlp.linear_fc2",
    "model.visual.blocks.6.attn.proj",
    "model.visual.blocks.6.attn.qkv",
    "model.visual.blocks.6.mlp.linear_fc1",
    "model.visual.blocks.6.mlp.linear_fc2",
    "model.visual.blocks.7.attn.proj",
    "model.visual.blocks.7.attn.qkv",
    "model.visual.blocks.7.mlp.linear_fc1",
    "model.visual.blocks.7.mlp.linear_fc2",
    "model.visual.blocks.8.attn.proj",
    "model.visual.blocks.8.attn.qkv",
    "model.visual.blocks.8.mlp.linear_fc1",
    "model.visual.blocks.8.mlp.linear_fc2",
    "model.visual.blocks.9.attn.proj",
    "model.visual.blocks.9.attn.qkv",
    "model.visual.blocks.9.mlp.linear_fc1",
    "model.visual.blocks.9.mlp.linear_fc2",
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
    "model.visual.patch_embed.proj",
    "model.visual.pos_embed",
    "mtp.fc",
    "mtp.layers.0.mlp.gate",
    "mtp.layers.0.mlp.shared_expert_gate"
  ]
}

```

397B-A17B FP8:

```text

/inspire/qb-ilm2/project/new-paradigm-of-manipulation/shixiaoxin-253107030017/simpletool_eval_20261110/eval_views/p005-397b-fp8

```

版本：`{"torch": "2.14.0+cu130", "vllm": "0.33.0", "cuda": "13.0"}`；quantization：

```json

{
  "quant_method": "fp8",
  "activation_scheme": "dynamic",
  "weight_per_tensor": false,
  "act_per_tensor": false,
  "weight_block_size": [
    128,
    128
  ],
  "modules_to_not_convert": [
    "lm_head",
    "model.language_model.embed_tokens",
    "model.language_model.layers.0.linear_attn.conv1d",
    "model.language_model.layers.0.linear_attn.in_proj_a",
    "model.language_model.layers.0.linear_attn.in_proj_b",
    "model.language_model.layers.0.mlp.gate",
    "model.language_model.layers.0.mlp.shared_expert_gate",
    "model.language_model.layers.1.linear_attn.conv1d",
    "model.language_model.layers.1.linear_attn.in_proj_a",
    "model.language_model.layers.1.linear_attn.in_proj_b",
    "model.language_model.layers.1.mlp.gate",
    "model.language_model.layers.1.mlp.shared_expert_gate",
    "model.language_model.layers.10.linear_attn.conv1d",
    "model.language_model.layers.10.linear_attn.in_proj_a",
    "model.language_model.layers.10.linear_attn.in_proj_b",
    "model.language_model.layers.10.mlp.gate",
    "model.language_model.layers.10.mlp.shared_expert_gate",
    "model.language_model.layers.11.mlp.gate",
    "model.language_model.layers.11.mlp.shared_expert_gate",
    "model.language_model.layers.12.linear_attn.conv1d",
    "model.language_model.layers.12.linear_attn.in_proj_a",
    "model.language_model.layers.12.linear_attn.in_proj_b",
    "model.language_model.layers.12.mlp.gate",
    "model.language_model.layers.12.mlp.shared_expert_gate",
    "model.language_model.layers.13.linear_attn.conv1d",
    "model.language_model.layers.13.linear_attn.in_proj_a",
    "model.language_model.layers.13.linear_attn.in_proj_b",
    "model.language_model.layers.13.mlp.gate",
    "model.language_model.layers.13.mlp.shared_expert_gate",
    "model.language_model.layers.14.linear_attn.conv1d",
    "model.language_model.layers.14.linear_attn.in_proj_a",
    "model.language_model.layers.14.linear_attn.in_proj_b",
    "model.language_model.layers.14.mlp.gate",
    "model.language_model.layers.14.mlp.shared_expert_gate",
    "model.language_model.layers.15.mlp.gate",
    "model.language_model.layers.15.mlp.shared_expert_gate",
    "model.language_model.layers.16.linear_attn.conv1d",
    "model.language_model.layers.16.linear_attn.in_proj_a",
    "model.language_model.layers.16.linear_attn.in_proj_b",
    "model.language_model.layers.16.mlp.gate",
    "model.language_model.layers.16.mlp.shared_expert_gate",
    "model.language_model.layers.17.linear_attn.conv1d",
    "model.language_model.layers.17.linear_attn.in_proj_a",
    "model.language_model.layers.17.linear_attn.in_proj_b",
    "model.language_model.layers.17.mlp.gate",
    "model.language_model.layers.17.mlp.shared_expert_gate",
    "model.language_model.layers.18.linear_attn.conv1d",
    "model.language_model.layers.18.linear_attn.in_proj_a",
    "model.language_model.layers.18.linear_attn.in_proj_b",
    "model.language_model.layers.18.mlp.gate",
    "model.language_model.layers.18.mlp.shared_expert_gate",
    "model.language_model.layers.19.mlp.gate",
    "model.language_model.layers.19.mlp.shared_expert_gate",
    "model.language_model.layers.2.linear_attn.conv1d",
    "model.language_model.layers.2.linear_attn.in_proj_a",
    "model.language_model.layers.2.linear_attn.in_proj_b",
    "model.language_model.layers.2.mlp.gate",
    "model.language_model.layers.2.mlp.shared_expert_gate",
    "model.language_model.layers.20.linear_attn.conv1d",
    "model.language_model.layers.20.linear_attn.in_proj_a",
    "model.language_model.layers.20.linear_attn.in_proj_b",
    "model.language_model.layers.20.mlp.gate",
    "model.language_model.layers.20.mlp.shared_expert_gate",
    "model.language_model.layers.21.linear_attn.conv1d",
    "model.language_model.layers.21.linear_attn.in_proj_a",
    "model.language_model.layers.21.linear_attn.in_proj_b",
    "model.language_model.layers.21.mlp.gate",
    "model.language_model.layers.21.mlp.shared_expert_gate",
    "model.language_model.layers.22.linear_attn.conv1d",
    "model.language_model.layers.22.linear_attn.in_proj_a",
    "model.language_model.layers.22.linear_attn.in_proj_b",
    "model.language_model.layers.22.mlp.gate",
    "model.language_model.layers.22.mlp.shared_expert_gate",
    "model.language_model.layers.23.mlp.gate",
    "model.language_model.layers.23.mlp.shared_expert_gate",
    "model.language_model.layers.24.linear_attn.conv1d",
    "model.language_model.layers.24.linear_attn.in_proj_a",
    "model.language_model.layers.24.linear_attn.in_proj_b",
    "model.language_model.layers.24.mlp.gate",
    "model.language_model.layers.24.mlp.shared_expert_gate",
    "model.language_model.layers.25.linear_attn.conv1d",
    "model.language_model.layers.25.linear_attn.in_proj_a",
    "model.language_model.layers.25.linear_attn.in_proj_b",
    "model.language_model.layers.25.mlp.gate",
    "model.language_model.layers.25.mlp.shared_expert_gate",
    "model.language_model.layers.26.linear_attn.conv1d",
    "model.language_model.layers.26.linear_attn.in_proj_a",
    "model.language_model.layers.26.linear_attn.in_proj_b",
    "model.language_model.layers.26.mlp.gate",
    "model.language_model.layers.26.mlp.shared_expert_gate",
    "model.language_model.layers.27.mlp.gate",
    "model.language_model.layers.27.mlp.shared_expert_gate",
    "model.language_model.layers.28.linear_attn.conv1d",
    "model.language_model.layers.28.linear_attn.in_proj_a",
    "model.language_model.layers.28.linear_attn.in_proj_b",
    "model.language_model.layers.28.mlp.gate",
    "model.language_model.layers.28.mlp.shared_expert_gate",
    "model.language_model.layers.29.linear_attn.conv1d",
    "model.language_model.layers.29.linear_attn.in_proj_a",
    "model.language_model.layers.29.linear_attn.in_proj_b",
    "model.language_model.layers.29.mlp.gate",
    "model.language_model.layers.29.mlp.shared_expert_gate",
    "model.language_model.layers.3.mlp.gate",
    "model.language_model.layers.3.mlp.shared_expert_gate",
    "model.language_model.layers.30.linear_attn.conv1d",
    "model.language_model.layers.30.linear_attn.in_proj_a",
    "model.language_model.layers.30.linear_attn.in_proj_b",
    "model.language_model.layers.30.mlp.gate",
    "model.language_model.layers.30.mlp.shared_expert_gate",
    "model.language_model.layers.31.mlp.gate",
    "model.language_model.layers.31.mlp.shared_expert_gate",
    "model.language_model.layers.32.linear_attn.conv1d",
    "model.language_model.layers.32.linear_attn.in_proj_a",
    "model.language_model.layers.32.linear_attn.in_proj_b",
    "model.language_model.layers.32.mlp.gate",
    "model.language_model.layers.32.mlp.shared_expert_gate",
    "model.language_model.layers.33.linear_attn.conv1d",
    "model.language_model.layers.33.linear_attn.in_proj_a",
    "model.language_model.layers.33.linear_attn.in_proj_b",
    "model.language_model.layers.33.mlp.gate",
    "model.language_model.layers.33.mlp.shared_expert_gate",
    "model.language_model.layers.34.linear_attn.conv1d",
    "model.language_model.layers.34.linear_attn.in_proj_a",
    "model.language_model.layers.34.linear_attn.in_proj_b",
    "model.language_model.layers.34.mlp.gate",
    "model.language_model.layers.34.mlp.shared_expert_gate",
    "model.language_model.layers.35.mlp.gate",
    "model.language_model.layers.35.mlp.shared_expert_gate",
    "model.language_model.layers.36.linear_attn.conv1d",
    "model.language_model.layers.36.linear_attn.in_proj_a",
    "model.language_model.layers.36.linear_attn.in_proj_b",
    "model.language_model.layers.36.mlp.gate",
    "model.language_model.layers.36.mlp.shared_expert_gate",
    "model.language_model.layers.37.linear_attn.conv1d",
    "model.language_model.layers.37.linear_attn.in_proj_a",
    "model.language_model.layers.37.linear_attn.in_proj_b",
    "model.language_model.layers.37.mlp.gate",
    "model.language_model.layers.37.mlp.shared_expert_gate",
    "model.language_model.layers.38.linear_attn.conv1d",
    "model.language_model.layers.38.linear_attn.in_proj_a",
    "model.language_model.layers.38.linear_attn.in_proj_b",
    "model.language_model.layers.38.mlp.gate",
    "model.language_model.layers.38.mlp.shared_expert_gate",
    "model.language_model.layers.39.mlp.gate",
    "model.language_model.layers.39.mlp.shared_expert_gate",
    "model.language_model.layers.4.linear_attn.conv1d",
    "model.language_model.layers.4.linear_attn.in_proj_a",
    "model.language_model.layers.4.linear_attn.in_proj_b",
    "model.language_model.layers.4.mlp.gate",
    "model.language_model.layers.4.mlp.shared_expert_gate",
    "model.language_model.layers.40.linear_attn.conv1d",
    "model.language_model.layers.40.linear_attn.in_proj_a",
    "model.language_model.layers.40.linear_attn.in_proj_b",
    "model.language_model.layers.40.mlp.gate",
    "model.language_model.layers.40.mlp.shared_expert_gate",
    "model.language_model.layers.41.linear_attn.conv1d",
    "model.language_model.layers.41.linear_attn.in_proj_a",
    "model.language_model.layers.41.linear_attn.in_proj_b",
    "model.language_model.layers.41.mlp.gate",
    "model.language_model.layers.41.mlp.shared_expert_gate",
    "model.language_model.layers.42.linear_attn.conv1d",
    "model.language_model.layers.42.linear_attn.in_proj_a",
    "model.language_model.layers.42.linear_attn.in_proj_b",
    "model.language_model.layers.42.mlp.gate",
    "model.language_model.layers.42.mlp.shared_expert_gate",
    "model.language_model.layers.43.mlp.gate",
    "model.language_model.layers.43.mlp.shared_expert_gate",
    "model.language_model.layers.44.linear_attn.conv1d",
    "model.language_model.layers.44.linear_attn.in_proj_a",
    "model.language_model.layers.44.linear_attn.in_proj_b",
    "model.language_model.layers.44.mlp.gate",
    "model.language_model.layers.44.mlp.shared_expert_gate",
    "model.language_model.layers.45.linear_attn.conv1d",
    "model.language_model.layers.45.linear_attn.in_proj_a",
    "model.language_model.layers.45.linear_attn.in_proj_b",
    "model.language_model.layers.45.mlp.gate",
    "model.language_model.layers.45.mlp.shared_expert_gate",
    "model.language_model.layers.46.linear_attn.conv1d",
    "model.language_model.layers.46.linear_attn.in_proj_a",
    "model.language_model.layers.46.linear_attn.in_proj_b",
    "model.language_model.layers.46.mlp.gate",
    "model.language_model.layers.46.mlp.shared_expert_gate",
    "model.language_model.layers.47.mlp.gate",
    "model.language_model.layers.47.mlp.shared_expert_gate",
    "model.language_model.layers.48.linear_attn.conv1d",
    "model.language_model.layers.48.linear_attn.in_proj_a",
    "model.language_model.layers.48.linear_attn.in_proj_b",
    "model.language_model.layers.48.mlp.gate",
    "model.language_model.layers.48.mlp.shared_expert_gate",
    "model.language_model.layers.49.linear_attn.conv1d",
    "model.language_model.layers.49.linear_attn.in_proj_a",
    "model.language_model.layers.49.linear_attn.in_proj_b",
    "model.language_model.layers.49.mlp.gate",
    "model.language_model.layers.49.mlp.shared_expert_gate",
    "model.language_model.layers.5.linear_attn.conv1d",
    "model.language_model.layers.5.linear_attn.in_proj_a",
    "model.language_model.layers.5.linear_attn.in_proj_b",
    "model.language_model.layers.5.mlp.gate",
    "model.language_model.layers.5.mlp.shared_expert_gate",
    "model.language_model.layers.50.linear_attn.conv1d",
    "model.language_model.layers.50.linear_attn.in_proj_a",
    "model.language_model.layers.50.linear_attn.in_proj_b",
    "model.language_model.layers.50.mlp.gate",
    "model.language_model.layers.50.mlp.shared_expert_gate",
    "model.language_model.layers.51.mlp.gate",
    "model.language_model.layers.51.mlp.shared_expert_gate",
    "model.language_model.layers.52.linear_attn.conv1d",
    "model.language_model.layers.52.linear_attn.in_proj_a",
    "model.language_model.layers.52.linear_attn.in_proj_b",
    "model.language_model.layers.52.mlp.gate",
    "model.language_model.layers.52.mlp.shared_expert_gate",
    "model.language_model.layers.53.linear_attn.conv1d",
    "model.language_model.layers.53.linear_attn.in_proj_a",
    "model.language_model.layers.53.linear_attn.in_proj_b",
    "model.language_model.layers.53.mlp.gate",
    "model.language_model.layers.53.mlp.shared_expert_gate",
    "model.language_model.layers.54.linear_attn.conv1d",
    "model.language_model.layers.54.linear_attn.in_proj_a",
    "model.language_model.layers.54.linear_attn.in_proj_b",
    "model.language_model.layers.54.mlp.gate",
    "model.language_model.layers.54.mlp.shared_expert_gate",
    "model.language_model.layers.55.mlp.gate",
    "model.language_model.layers.55.mlp.shared_expert_gate",
    "model.language_model.layers.56.linear_attn.conv1d",
    "model.language_model.layers.56.linear_attn.in_proj_a",
    "model.language_model.layers.56.linear_attn.in_proj_b",
    "model.language_model.layers.56.mlp.gate",
    "model.language_model.layers.56.mlp.shared_expert_gate",
    "model.language_model.layers.57.linear_attn.conv1d",
    "model.language_model.layers.57.linear_attn.in_proj_a",
    "model.language_model.layers.57.linear_attn.in_proj_b",
    "model.language_model.layers.57.mlp.gate",
    "model.language_model.layers.57.mlp.shared_expert_gate",
    "model.language_model.layers.58.linear_attn.conv1d",
    "model.language_model.layers.58.linear_attn.in_proj_a",
    "model.language_model.layers.58.linear_attn.in_proj_b",
    "model.language_model.layers.58.mlp.gate",
    "model.language_model.layers.58.mlp.shared_expert_gate",
    "model.language_model.layers.59.mlp.gate",
    "model.language_model.layers.59.mlp.shared_expert_gate",
    "model.language_model.layers.6.linear_attn.conv1d",
    "model.language_model.layers.6.linear_attn.in_proj_a",
    "model.language_model.layers.6.linear_attn.in_proj_b",
    "model.language_model.layers.6.mlp.gate",
    "model.language_model.layers.6.mlp.shared_expert_gate",
    "model.language_model.layers.7.mlp.gate",
    "model.language_model.layers.7.mlp.shared_expert_gate",
    "model.language_model.layers.8.linear_attn.conv1d",
    "model.language_model.layers.8.linear_attn.in_proj_a",
    "model.language_model.layers.8.linear_attn.in_proj_b",
    "model.language_model.layers.8.mlp.gate",
    "model.language_model.layers.8.mlp.shared_expert_gate",
    "model.language_model.layers.9.linear_attn.conv1d",
    "model.language_model.layers.9.linear_attn.in_proj_a",
    "model.language_model.layers.9.linear_attn.in_proj_b",
    "model.language_model.layers.9.mlp.gate",
    "model.language_model.layers.9.mlp.shared_expert_gate",
    "model.visual.blocks.0.attn.proj",
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.0.mlp.linear_fc1",
    "model.visual.blocks.0.mlp.linear_fc2",
    "model.visual.blocks.1.attn.proj",
    "model.visual.blocks.1.attn.qkv",
    "model.visual.blocks.1.mlp.linear_fc1",
    "model.visual.blocks.1.mlp.linear_fc2",
    "model.visual.blocks.10.attn.proj",
    "model.visual.blocks.10.attn.qkv",
    "model.visual.blocks.10.mlp.linear_fc1",
    "model.visual.blocks.10.mlp.linear_fc2",
    "model.visual.blocks.11.attn.proj",
    "model.visual.blocks.11.attn.qkv",
    "model.visual.blocks.11.mlp.linear_fc1",
    "model.visual.blocks.11.mlp.linear_fc2",
    "model.visual.blocks.12.attn.proj",
    "model.visual.blocks.12.attn.qkv",
    "model.visual.blocks.12.mlp.linear_fc1",
    "model.visual.blocks.12.mlp.linear_fc2",
    "model.visual.blocks.13.attn.proj",
    "model.visual.blocks.13.attn.qkv",
    "model.visual.blocks.13.mlp.linear_fc1",
    "model.visual.blocks.13.mlp.linear_fc2",
    "model.visual.blocks.14.attn.proj",
    "model.visual.blocks.14.attn.qkv",
    "model.visual.blocks.14.mlp.linear_fc1",
    "model.visual.blocks.14.mlp.linear_fc2",
    "model.visual.blocks.15.attn.proj",
    "model.visual.blocks.15.attn.qkv",
    "model.visual.blocks.15.mlp.linear_fc1",
    "model.visual.blocks.15.mlp.linear_fc2",
    "model.visual.blocks.16.attn.proj",
    "model.visual.blocks.16.attn.qkv",
    "model.visual.blocks.16.mlp.linear_fc1",
    "model.visual.blocks.16.mlp.linear_fc2",
    "model.visual.blocks.17.attn.proj",
    "model.visual.blocks.17.attn.qkv",
    "model.visual.blocks.17.mlp.linear_fc1",
    "model.visual.blocks.17.mlp.linear_fc2",
    "model.visual.blocks.18.attn.proj",
    "model.visual.blocks.18.attn.qkv",
    "model.visual.blocks.18.mlp.linear_fc1",
    "model.visual.blocks.18.mlp.linear_fc2",
    "model.visual.blocks.19.attn.proj",
    "model.visual.blocks.19.attn.qkv",
    "model.visual.blocks.19.mlp.linear_fc1",
    "model.visual.blocks.19.mlp.linear_fc2",
    "model.visual.blocks.2.attn.proj",
    "model.visual.blocks.2.attn.qkv",
    "model.visual.blocks.2.mlp.linear_fc1",
    "model.visual.blocks.2.mlp.linear_fc2",
    "model.visual.blocks.20.attn.proj",
    "model.visual.blocks.20.attn.qkv",
    "model.visual.blocks.20.mlp.linear_fc1",
    "model.visual.blocks.20.mlp.linear_fc2",
    "model.visual.blocks.21.attn.proj",
    "model.visual.blocks.21.attn.qkv",
    "model.visual.blocks.21.mlp.linear_fc1",
    "model.visual.blocks.21.mlp.linear_fc2",
    "model.visual.blocks.22.attn.proj",
    "model.visual.blocks.22.attn.qkv",
    "model.visual.blocks.22.mlp.linear_fc1",
    "model.visual.blocks.22.mlp.linear_fc2",
    "model.visual.blocks.23.attn.proj",
    "model.visual.blocks.23.attn.qkv",
    "model.visual.blocks.23.mlp.linear_fc1",
    "model.visual.blocks.23.mlp.linear_fc2",
    "model.visual.blocks.24.attn.proj",
    "model.visual.blocks.24.attn.qkv",
    "model.visual.blocks.24.mlp.linear_fc1",
    "model.visual.blocks.24.mlp.linear_fc2",
    "model.visual.blocks.25.attn.proj",
    "model.visual.blocks.25.attn.qkv",
    "model.visual.blocks.25.mlp.linear_fc1",
    "model.visual.blocks.25.mlp.linear_fc2",
    "model.visual.blocks.26.attn.proj",
    "model.visual.blocks.26.attn.qkv",
    "model.visual.blocks.26.mlp.linear_fc1",
    "model.visual.blocks.26.mlp.linear_fc2",
    "model.visual.blocks.3.attn.proj",
    "model.visual.blocks.3.attn.qkv",
    "model.visual.blocks.3.mlp.linear_fc1",
    "model.visual.blocks.3.mlp.linear_fc2",
    "model.visual.blocks.4.attn.proj",
    "model.visual.blocks.4.attn.qkv",
    "model.visual.blocks.4.mlp.linear_fc1",
    "model.visual.blocks.4.mlp.linear_fc2",
    "model.visual.blocks.5.attn.proj",
    "model.visual.blocks.5.attn.qkv",
    "model.visual.blocks.5.mlp.linear_fc1",
    "model.visual.blocks.5.mlp.linear_fc2",
    "model.visual.blocks.6.attn.proj",
    "model.visual.blocks.6.attn.qkv",
    "model.visual.blocks.6.mlp.linear_fc1",
    "model.visual.blocks.6.mlp.linear_fc2",
    "model.visual.blocks.7.attn.proj",
    "model.visual.blocks.7.attn.qkv",
    "model.visual.blocks.7.mlp.linear_fc1",
    "model.visual.blocks.7.mlp.linear_fc2",
    "model.visual.blocks.8.attn.proj",
    "model.visual.blocks.8.attn.qkv",
    "model.visual.blocks.8.mlp.linear_fc1",
    "model.visual.blocks.8.mlp.linear_fc2",
    "model.visual.blocks.9.attn.proj",
    "model.visual.blocks.9.attn.qkv",
    "model.visual.blocks.9.mlp.linear_fc1",
    "model.visual.blocks.9.mlp.linear_fc2",
    "model.visual.merger.linear_fc1",
    "model.visual.merger.linear_fc2",
    "model.visual.patch_embed.proj",
    "model.visual.pos_embed",
    "mtp.fc",
    "mtp.layers.0.mlp.gate",
    "mtp.layers.0.mlp.shared_expert_gate"
  ]
}

```

## 9. 交付与重算

`RESULTS.json` 保留全部分位数、模型配置和 rank 快照。`pure_infra.csv` 是冷/热×B×T 的 80 格结果；`scenarios.csv` 是 15 组场景统计；`requests.csv` 是 900 条正式请求；`heads.csv` 是各 head 的逐条数据。`data/` 内固定 180 条题目及 tags/expected；`make_scenarios.py` 可重新生成额外观测。

`evidence/<run>/requests.jsonl` 包含纯 infra、预热、验证、正式与 soak 原始时序；`quality.jsonl` 是正式 180 条评分；`examples.jsonl` 保留输入 token IDs。每运行 archive 和 FILE_MANIFEST 验证文件 SHA256。失败运行不删除，不计入正式统计。

本地重新统计：
```bash
cd /Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/experiments/simpletool_h200_speed_20261113/scenario_v2
python3 summarize.py
python3 write_report.py
```

可搬迁源码目录：`infra_bundle/`；源码压缩包：`infra_bundle.tar.gz`。依赖版本与原始 overlay 哈希见 `ENVIRONMENT.json`。`DELIVERY_MANIFEST.json` 核对交付文件；`verify_delivery.py` 从模型原始文本重新评分并检查归档哈希。完整交付压缩包为 `simpletool_h200_scenario_v2_delivery.tar.gz`，含原始运行归档，不含模型权重。

资源释放回执见 `RELEASE.json`（存在且已确认 stopped 状态才表示已释放）。云端结果位于持久共享存储，释放 GPU 容器不删除这些文件。

## 10. 资源释放确认

任务 `job-6f0c3466-9641-4739-b9a1-edac94941f41` 已确认 `job_stopped`，node_count=0，node_infos 为空。确认时间：2026-11-13T14:37:44.587718+00:00。本次 GPU 容器已释放，独立 CPU 服务未改动。
