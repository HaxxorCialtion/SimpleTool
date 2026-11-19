# H200 35B-A3B FP8：继续扩大batch至吞吐下降

checkpoint标识：SimpleTool-Qwen3.6-35B-A3B-8Head-5pct-FP8，架构配置qwen3_5_moe。模型与算法沿用2026-11-16的B1–32实验：同一35B-A3B block-FP8八头checkpoint，公共N512 tokens一次prefill（零输出），发布末端KV/GDN/卷积状态，B路marker与后续生成token批量单tokendecode。各头固定生成64 tokens、ignore_eos。

主机观察TPOT包含引擎调度、采样和测量程序记账；排除prefill、marker首步、网络及冷启动。合计吞吐=B×63/所有分支首token至最后token的窗口。它不是决策Hz。

B>8复用八种已训练marker；同一公共prompt下重复分支可能提高MoE expert路由重合，不能外推成独立多用户批处理结果或新增语义头。

## 结果

停止原因：`confirmed_throughput_decline`。所测档位中最高候选汇总吞吐位于B=704，P50=15869.4 token/s。各档样本来源不同（control/screen/confirm），拐点判定以成对复测为准。

| B | 请求数 | 每头TPOT P50/P95 ms | 每头速率保留 | 合计decode TPS P50/P95 | prime P50 ms | marker阶段 P50 ms | 整批完成 P50 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20 | 3.844/5.813 | 100.0% | 260.2/260.4 | 22.522 | 7.246 | 272.022 |
| 32 | 20 | 5.984/6.134 | 64.2% | 5347.5/5479.0 | 22.644 | 35.115 | 436.314 |
| 256 | 20 | 17.553/17.711 | 21.9% | 14584.7/14888.0 | 23.134 | 237.651 | 1372.847 |
| 384 | 20 | 25.433/25.585 | 15.1% | 15098.5/15151.9 | 23.016 | 353.820 | 1988.729 |
| 512 | 20 | 32.566/32.985 | 11.8% | 15722.0/15915.1 | 23.054 | 470.166 | 2558.544 |
| 576 | 12 | 36.304/36.778 | 10.6% | 15866.1/15960.7 | 22.358 | 526.642 | 2856.728 |
| 640 | 24 | 41.394/43.677 | 9.3% | 15461.2/15675.4 | 22.904 | 593.098 | 3239.938 |
| 704 | 12 | 44.362/44.600 | 8.7% | 15869.4/15951.8 | 22.425 | 643.148 | 3483.145 |
| 768 | 12 | 49.284/50.112 | 7.8% | 15583.2/15948.4 | 22.865 | 725.216 | 3883.293 |
| 896 | 24 | 59.707/63.253 | 6.4% | 15006.7/15973.6 | 23.807 | 844.477 | 4670.982 |
| 1024 | 24 | 69.091/73.322 | 5.6% | 14821.1/15913.7 | 23.794 | 961.737 | 5384.909 |

## 成对对照与停止判定

每个新batch与此前最佳batch在相同输入上进行12组随机AB/BA成对测试；出现中位吞吐下降时，再进行24组确认。以成对吞吐比值中位数的bootstrap 95%区间上界<1作为确认条件，若预设档位中仍有更大的batch，则再补测该档位；若下降发生在最大档1024，则没有额外更高档验证。bootstrap重复4000次。该自适应探索没有作多重比较校正，不证明连续整数batch或所有配置的全局最优。

| 阶段 | 对照B | 新B | 配对数 | 新/对照吞吐比中位数 | bootstrap 95%区间 |
|---|---:|---:|---:|---:|---:|
| screen | 512 | 576 | 12 | 1.00723 | 1.00191 – 1.01190 |
| screen | 576 | 640 | 12 | 0.98752 | 0.97383 – 1.00210 |
| confirm | 576 | 640 | 24 | 1.00046 | 0.98858 – 1.00605 |
| screen | 576 | 704 | 12 | 1.01215 | 0.99976 – 1.01517 |
| screen | 704 | 768 | 12 | 1.00002 | 0.98983 – 1.01897 |
| screen | 768 | 896 | 12 | 0.99967 | 0.98325 – 1.02056 |
| confirm | 768 | 896 | 24 | 1.00121 | 0.99222 – 1.00541 |
| screen | 768 | 1024 | 12 | 0.98325 | 0.96939 – 0.99471 |
| confirm | 768 | 1024 | 24 | 0.99401 | 0.97776 – 0.99858 |
## 最终同轮成对确认

B1024 / B768吞吐比中位数=0.994014，下降0.60%；bootstrap 95%区间=[0.9777630349722025, 0.9985810496393065]。

| 同轮确认档位 | TPOT P50 ms | 合计TPS P50 |
|---:|---:|---:|
| 768 | 51.055 | 15043.9 |
| 1024 | 69.091 | 14821.1 |

成对比值中位数与两个分组中位数之比不必完全相同。若B1024为首次确认下降档，则本轮没有测B>1024；结论是已观察到下降，不是证明峰值恰好位于某一整数batch。


## 配置、验证与限制

- 单卡H200，TP1；FP8 E4M3 block128×128，动态激活量化；model dtype BF16，KV/GDN auto；不是W4A16/KV8。
- max_num_seqs=1024、max_model_len=4096、max_num_batched_tokens=4096；异步调度；原生FULL_DECODE_ONLY CUDA graph捕获到1024。公共prefill使用既有专用graph路径。
- 缓存预算88 GiB、GDN逻辑块128；前两次block16粗扫/加密为80/96 GiB，并分别在B768/B704发生抢占重算。本轮重测B1/32/256/384/512对照，拐点用本轮同进程成对比较。复用之前已完成任务的compiler-keyed缓存，启动不是全冷编译。
- inherited machine.json中的80GiB备注来自原始脚本；本轮缓存以config.json args.kv_gib=88、launch.sh和日志为准。原始证据保留。
- 不执行任何生成的工具；没有视觉输入、自然EOS裁头、HTTP或外部执行延迟。
- 所有请求运行时断言：一次完整512-token prime forward；所有head命中512公共tokens；每次decode调度B个单token输入，完整64步。原始完整timeline按batch保留首条，全部正式请求保留薄指标。
- 每个实际测试batch用4-token短序列对同batch公共prefill graph/eager生成token作严格一致比较；正式64-token段逐请求检查布局，未逐条重跑eager。并记录重复alias间的一致性。该验证不等于跨batch生成一致性或真实工具质量通过。
- TPOT包含按head逐token收集的host测量开销；高batch结果是该引擎与测量程序的观测吞吐，不是GPU裸kernel极限。

Python3.12.12；FlashAttention3、FlashInfer GDN、FP8 DeepGEMM线性层、TRITON FP8 MoE（auto解析）。GPU为单H200、TP1；遥测SM频率1980MHz。运行时版本：`{"torch": "2.14.0+cu130", "vllm": "0.33.0", "cuda": "13.0"}`。
运行时检查443次请求，正式计时388次，验证11个batch。

独立样例与薄指标审计：

```json
{
  "metrics_requests": 388,
  "checked_runtime": 443,
  "timed_phase_counts": {
    "control": 100,
    "screen": 144,
    "confirm": 144
  },
  "full_timeline_batches_audited": [
    1,
    32,
    256,
    384,
    512,
    576,
    640,
    704,
    768,
    896,
    1024
  ],
  "timeline_failures": [],
  "same_batch_public_graph_eager_validations": 11,
  "alias_mismatch_count": 0,
  "prefill_graph_modes": {
    "capture-replay": 1,
    "replay": 431
  },
  "gpu_memory_final": {
    "allocated": 133869072896,
    "reserved": 147794690048,
    "max_allocated": 136066904576,
    "free": 1477574656
  },
  "metrics_sha256": "799bf377738289de0b0461d43b71e32814c929326f99dfb45bdf9c9b9e52b6e9"
}
```

## 文件与复现

本地目录：`/Users/cialtionshi/SimpleLove_AIOC/simpletool_codex_engine/experiments/simpletool_h200_throughput_peak_20261116`。
云端目录：`/inspire/hdd/global_user/shixiaoxin-253107030017/sxx/simpletool_h200_throughput_peak_20261116/block128/`。
模型只读复用：`/inspire/hdd/global_user/shixiaoxin-253107030017/sxx/simpletool_h200_batch32_20261116/model/`。
bench.py/launch.sh、配置、GPU遥测、提交/退出回执、profiles.json、comparisons.json、metrics.jsonl和分batch timeline保留。复现需要同一H200环境与私有runtime快照，并为输出选择新目录。

## block16失败与配置变更

最初80GiB、GDN逻辑块16的粗扫在B768预热失败：64个生成token经历76次调度。96GiB重试在B704失败：64个token经历73次调度；完整timeline显示16/32/48步边界少量分支被抢占并重新计算公共512token及后缀。两者均未作为有效B768/B704性能数据。

本轮改为GDN逻辑块128，并用88GiB缓存；每head私有GDN/卷积状态和公共末端512的精确缓存匹配保持。64生成token测量段不跨128token状态块，减少中途分配/双状态并存压力。这是当前N512/T64专用优化，其他prompt长度/生成长度可能跨块，不能外推为任意长序列没有换块。

## block16/80GiB粗扫（同日另一张H200）

下列为先前配置，拐点判定不混合两配置的请求。

| B | 每头TPOT P50 ms | 合计decode TPS P50 |
|---:|---:|---:|
| 1 | 3.845 | 260.1 |
| 8 | 4.908 | 1629.9 |
| 16 | 5.240 | 3053.5 |
| 32 | 6.028 | 5308.8 |
| 48 | 6.830 | 7027.8 |
| 64 | 7.451 | 8589.9 |
| 96 | 9.225 | 10406.3 |
| 128 | 10.622 | 12050.5 |
| 192 | 14.901 | 12885.3 |
| 256 | 18.235 | 14039.1 |
| 384 | 26.563 | 14456.4 |
| 512 | 34.194 | 14973.5 |

配置及平台最终回执分别保存在job-request*.json与final-*status*.json；失败记录保留在evidence/、evidence-near/，最终block128记录在evidence-block128/。

## 资源回收

最终平台回执：job_succeeded；实例instance_success。临时H200任务已经结束，未停止其他工作负载。回执文件：final-block128-status.json、final-block128-instances.json。
