# iPhone build43 A16＋KV8 推理优化验收

> 后续真实对话发现：build43在特定三轮历史下循环。其固定步数速度结果仍有效，但原八项测试不能代表完整多轮生成质量。build46的定位和聊天策略修复见 `IPHONE_REPEAT_FIX.md`；原始greedy的跨平台/跨精度数值分歧尚未消除。

2026-11-07，iPhone 17 Pro Max，MLX Metal GPU，Release。packed affine INT4/group64权重保持原文件；FP16激活/I-O，自定义投影内部FP32反量化/累加，GDN递归矩阵/gates FP32，attention KV为INT8/group64。不是W4A8或NPU。

最终按架构选择：0.8B统一B1–B4投影算术及融合MLP；2B/4B保留单头compact QMV与B2–B4权重复用/融合SwiGLU。保留4B stride4/16MiB allocator cache、辅助引擎分阶段卸载。未启用实验性Q8融合attention、GDN prepare融合、stride8或微测tile表。普通路径关闭算子profile，不构造profile标签。

## 三模型完整速度复测

精确512 tokens系统缓存命中＋512新增prefill。每档预热一次、正式三次，交替B4/B1先后顺序，取全部正式样本中位数，不删除慢样本。B4固定每头32步共128 tokens；单头speech固定32步；普通无通道marker AR固定30步。固定长度忽略EOS，仅衡量实际运算吞吐。

| 模型 | prefill秒 | B1 speech t/s | 普通AR t/s | B4每头 t/s | B4合计 t/s | B4 decode秒 | 理想4m达成 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.8B | 0.1893 | 91.26 | 90.84 | 65.54 | 262.15 | 0.4883 | 71.8% |
| 2B | 0.4218 | 44.89 | 37.42 | 26.64 | 106.55 | 1.2013 | 59.3% |
| 4B | 1.5625 | 19.49 | 18.68 | 11.01 | 44.02 | 2.9075 | 56.5% |

prefill包含增量prefill及generation suffix，不包含权重加载/system首次预热。decode包含量化/反量化、marker、CPU构图提交及GPU求值等待；不是纯GPU kernel时间。理想4m达成=T_B1/T_B4，不是硬件利用率。

## 同包旧路径对照（4B）

对照实际build43、同一组packed权重、同一进程内轮换legacy与production；legacy复现build35的A16通用矩阵路径。这是独立于上表完整速度任务的匹配对照，保留它的实际build。不能把历史不同热状态的49→36直接归因为精度损失。

| 路径 | B4 decode秒 | B4总吞吐t/s | B1 speech t/s | 自然B4 decode秒 |
|---|---:|---:|---:|---:|
| legacy | 4.3483 | 29.44 | 16.12 | 2.2181 |
| production | 3.1339 | 40.84 | 18.55 | 1.7981 |

## 小模型同包选择对照

以下是build42同进程轮换测量，每档正式3次，不是把最终包和历史冷状态直接相减。0.8B选择consistent以恢复长中文fixture的B4/B1一致性；2B保留production。

| 模型 | legacy B4 t/s | 选择后B4 t/s | 四头变化 | legacy B1 t/s | 选择后B1 t/s | 单头变化 |
|---|---:|---:|---:|---:|---:|---:|
| 0.8b | 217.70 | 259.68 | +19.3% | 94.14 | 89.53 | -4.9% |
| 2b | 85.60 | 121.43 | +41.9% | 42.90 | 43.05 | +0.3% |

0.8B并非所有指标都变快：匹配对照中单头约慢5%，换取四头吞吐提升和该长聊天对拍恢复。2B单头差异很小，不主张有稳定单头收益。


4B上述匹配对照：四头吞吐变化+38.8%；单头speech吞吐变化+15.1%。两组按实际热状态和顺序测量，不锁频；小幅差异不作因果结论。

## 稳定性与范围

自然EOS、真实中文聊天、同模式重复及B4/B1 speech对拍均包含在三模型速度任务。另有三模型all-head串行对拍、512/1024新增prefill、取消恢复、单头memory通路与卸载检查，以及Lab新对话和纳西妲4B测试。内容质量、跨精度逐token相等和任意输入长热稳态不在本轮承诺范围。

- llm-0.8b: completed / all_passed=true
- llm-2b: completed / all_passed=true
- llm-4b: completed / all_passed=true
- lab-new-chat: completed / all_passed=true
- lab-nahida: completed / all_passed=true
- 0.8B: thermal=[0]，采样进程峰=881.3 MiB，采样最小可分配余量=2494.7 MiB，卸载MLX active=0，计时段外温控等待=0.0秒
- 2B: thermal=[0]，采样进程峰=1555.1 MiB，采样最小可分配余量=1820.9 MiB，卸载MLX active=0，计时段外温控等待=0.0秒
- 4B: thermal=[0]，采样进程峰=3044.8 MiB，采样最小可分配余量=331.2 MiB，卸载MLX active=0，计时段外温控等待=0.0秒

## 自然停止及分头可用时间

相同512命中＋512新增fixture；以下是自然EOS/通道停止，不把固定32步诊断当作业务长度。每头时间从decode开始计，speech可逐token流式；结构化头到stop后才发布。

| 模型 | prefill中位秒 | 自然B4 decode中位秒 | speech stop秒/各次token数 | expression stop秒/各次token数 | inner stop秒/各次token数 | command stop秒/各次token数 |
|---|---:|---:|---|---|---|---|
| 0.8b | 0.1893 | 0.1600 | 0.1038/[8, 8, 8] | 0.0155/[1, 1, 1] | 0.1597/[13, 13, 13] | 0.0155/[1, 1, 1] |
| 2b | 0.4218 | 0.4505 | 0.4502/[16, 16, 16] | 0.0298/[1, 1, 1] | 0.0298/[1, 1, 1] | 0.0298/[1, 1, 1] |
| 4b | 1.5625 | 1.7241 | 0.9732/[15, 15, 15] | 0.0910/[1, 1, 1] | 1.7237/[29, 29, 29] | 0.0910/[1, 1, 1] |

调优过程保留build36/37/38/39全部收据和源码快照；Q8融合未见稳定收益，stride8提升很小，tile微测不代表整轮显著提速，因此未默认启用。build35源码与二进制归属错误已单独更正，不删除旧测量。算子profile有强制同步，stage含其内部linear，不能相加或视作纯GPU计数器。

## 波动与历史结果

本轮build43完整速度任务的4B四头中位44.02 token/s；随后同包匹配对照的production中位40.84，原始三次耗时2.2383/3.1801/3.1339秒。历史build40曾测得65.00，但不能将其承诺为当前稳定速度。所有样本保留，无异常值剔除。thermal标记没有锁定频率或全部系统竞争，差异尚未完整归因；这些结果不是硬件理论上限。

