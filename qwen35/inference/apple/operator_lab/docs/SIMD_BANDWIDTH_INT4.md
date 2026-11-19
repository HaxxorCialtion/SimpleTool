# int4 SIMD 布局与原生带宽校准：2026-10-08

本轮实现新的多通道 SIMD 布局，验证其数值并轮换真实模型各层权重测量。候选整体更慢，已拒绝接入默认路径。另完成两个独立进程的原生 Metal 带宽校准，得到真实4B packed payload的有效读取吞吐约249/256GB/s。本轮没有新完整模型提速，4B/2B主线仍为int4，没有新增FP16语言模型推理。

## 新 kernel 与拒绝依据

`simpletool/simd_branch_qmv.py` 为每个通道分配一个SIMD group，降低每线程的分支累加器数量。`split`直接读packed权重；`shared`让不同分支协作加载同一块packed ushort4及metadata到threadgroup memory，再同步计算。输入、累加、输出均FP32，half仅用于既有量化metadata存储。保持已验证compact单向量的算术与归约顺序，公共权重不修改。

两版都支持B1..4、rows1/2/4/8、K为512的倍数、group64、FP32或half metadata。shared版本每个512输入块需要两次barrier；每线程累加器减少并不保证整核更快，分支拆开也增加了执行组数量及重复计算。

256项结构验证通过，含非整齐N79、B1..4、两种metadata、signed zero、非连续输入，比较FP32输出的uint32位模式，输入SHA不变。真实2B/4B参数的2736项候选输出同样逐位等于独立逐向量gold。

性能使用真实所有MLP down/gate层轮换：4B各32层、2B各24层；每个配置五轮，同步wall time包含Python/MLX调用开销。vocabulary只有一个矩阵，重复测量，不能称为跨层冷读。没有强制缓存清空，也没有测DRAM计数器。

以下为B4每个算子的平均层时延中位数，各形状中最快新候选均为split rows4：

| 模型/矩阵 | 既有row-reuse | 最快新候选 | 时延变化 |
| --- | ---: | ---: | ---: |
| 4B down | 0.212 ms | 0.241 ms | 慢13.4% |
| 4B gate | 0.196 ms | 0.214 ms | 慢9.0% |
| 4B vocabulary | 2.197 ms | 3.064 ms | 慢39.5% |
| 2B down | 0.161 ms | 0.172 ms | 慢6.6% |
| 2B gate | 0.156 ms | 0.165 ms | 慢6.1% |
| 2B vocabulary | 1.800 ms | 2.471 ms | 慢37.3% |

完整样本含B2/B3/B4与全部八种新配置，见 `reports/simd_branch_qmv_{2b,4b}_bitwise_20261008.json`。候选未加入完整Python模型、Swift模型或默认策略，也不声称真实指令对拍已经验证这个新布局。

原生iOS Metal编译48个候选specialization通过，另有校准Swift程序的iOS arm64类型检查；未在iPhone执行。此处仅证明编译，不证明手机性能或完整SDK接入。

## 原生硬件校准

`native/bench_payload_bandwidth.swift` 将本项目4B int4 safetensors的真实language payload作为原始uint32位读取，不做语言推理或浮点反量化。payload精确2,369,040,384字节，拆为64MiB buffers，完整数据轮换；读取kernel采用不同每线程工作量，并以CPU总和mod32核对全部输出。copy路径使用blit并逐字节核对目标buffer。所有校验通过。

每条路径预热一次，七轮轮换计时，记录command buffer的GPU时间以及编码/等待wall time。两个独立进程结果：

| 进程 | 最快完整payload读取 | 有效读取吞吐 | copy读写吞吐 |
| --- | ---: | ---: | ---: |
| 第一批 | 9.524 ms | 248.74 GB/s | 233.50 GB/s |
| 独立复测 | 9.265 ms | 255.71 GB/s | 236.57 GB/s |

读取吞吐计算payload/GPU时间；copy计算2×payload/GPU时间。小工作量IPT4只有约170–171GB/s，IPT16/64/256可达约246–256GB/s。证明实际线程工作量会影响即使简单读取的吞吐。

现场设备为Apple M4 Pro、24GiB RAM；Apple给出的M4 Pro标称统一内存带宽为最高273GB/s。[Apple官方规格](https://www.apple.com.cn/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/)。校准最好结果约为标称的91–94%，但这是有效payload吞吐，不是DRAM counter、int4算术吞吐或完整推理利用率。输入及完整copy目标总buffer约4.757GB，仅用于校准，不代表部署RAM。外部负载和热状态未受控。

## 对优化方向的影响

之前同权重Swift组合候选四头固定96 token观测1.914s，见 `reports/swift_projection_4b_ab_repeat_20261008.json`。这不能直接与9.265ms的简单读取当作同一工作量对比：完整推理还包括解包、缩放、FP32计算、GDN、attention、状态读写与调度。

基于既有tensor header清点，作如下明确条件假设：每步全部权重读一次；每个分支全部FP32 GDN矩阵读写一次；忽略缓存命中、额外KV/conv/activation流量及计算成本。4B条件字节数为单头2.470GB、四头2.772GB。用273GB/s除，96步四头的仅流量估算约0.975s；其单头/四头比例约89.1%。

这不是严格时延下界、效率上界或可实现的提速承诺。模型实际DRAM流量未测，假设也可能不成立。它提供下一步假设：若仍完整读写四份dense递推矩阵，仅调权重读取无法解释如何达到98%效率；应继续尝试状态表示与写入策略，同时保留FP32算术、共同前缀只读和独立分支语义。改变表示可能改变舍入，必须重新对拍，不能以代数等价替代实际数值证据。

本轮没有证明距离完整推理硬件极限的精确比例，也没有突破98%或iPhone验收。它排除了当前SIMD拆分候选，并将原先的标称带宽讨论推进到真实payload的原生校准。

## 证据与复现

汇总 `reports/simd_bandwidth_summary_20261008.json` 验证2992项逐位案例、两组原生校准、iOS编译，保存全部源码、binary及checkpoint SHA。版本快照在 `artifacts/simd_bandwidth_snapshot_20261008/`，不含大模型权重。

- `reports/simd_branch_qmv_regression_20261008.json`及日志。
- `reports/simd_branch_qmv_{2b,4b}_bitwise_20261008.json`及日志。
- `reports/payload_bandwidth_{4b,4b_repeat}_20261008.json`及日志。
- `reports/simd_branch_ios_build_20261008.{json,log}`、`artifacts/simd_branch_ios_20261008.metallib`。

初轮计时脚本因变量round遮蔽内置函数而在打印时失败，已保留 `simd_branch_qmv_4b_first_failed_20261008.*`；修正后重跑。初步数值相等记录也保留，正式结论使用uint32位模式检查。第一版输入构造被where变为连续，已保留contiguous_first记录并修正为先where再切片，正式非连续测试在修正后完成。

项目根目录运行：

```sh
/Users/cialtionshi/mlx_env/bin/python check_simd_branch_qmv.py
/Users/cialtionshi/mlx_env/bin/python bench_simd_branch_qmv.py --size 4b --output reports/simd_branch_qmv_4b_bitwise_20261008.json
/Users/cialtionshi/mlx_env/bin/python bench_simd_branch_qmv.py --size 2b --output reports/simd_branch_qmv_2b_bitwise_20261008.json
xcrun swiftc -O native/bench_payload_bandwidth.swift -o artifacts/bench_payload_bandwidth
artifacts/bench_payload_bandwidth artifacts/4b_int4_compact/model.safetensors reports/payload_bandwidth_4b_20261008.json
/Users/cialtionshi/mlx_env/bin/python build_simd_branch_ios.py
/Users/cialtionshi/mlx_env/bin/python make_simd_bandwidth_report.py
```

汇总脚本只读取证据、核对SHA并保存快照，不执行GPU推理。完整手机运行、Swift原生prompt编码/图像/多轮、owned GDN完整接入、量化质量、llama.cpp一致性和98%目标仍未验收。
