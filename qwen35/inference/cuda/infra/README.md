# SimpleTool 固定公共状态推理实验源代码包

本包不包含模型、凭据、共享 vLLM 安装或编译缓存。它是已测云端实现的独立源码快照；GPU 环境必须匹配已测 API，不能当作任意 pip vLLM 版本可直接使用的通用插件。

已测：H200 / SM90，vLLM 0.33.0，Torch 2.14.0+cu130，CUDA 13.0。原 GPU Python 为 `/inspire/hdd/global_user/shixiaoxin-253107030017/sxx/vllm_qwen35_sm90_simpletool/bin/python`。不改共享安装；`rebase.py` 仅重写本包私有 manifest 路径并核对 overlay 哈希。

```bash
export SL_GPU_PYTHON=/inspire/hdd/global_user/shixiaoxin-253107030017/sxx/vllm_qwen35_sm90_simpletool/bin/python
bash launch.sh --model /absolute/path/to/model --tp 1 --output /absolute/path/to/new-run --mode full --prefill-graphs on --repeats 20 --soak 600 --async-scheduling --allow-whole-cold-drift
```

397B 已选配置为 TP4、同步调度，省略 `--async-scheduling`；122B 为 TP2。权重、量化格式、特殊 token 和 tokenizer 必须配套。模型权重路径和每次实际配置在上一级 `REPORT.md` 及 `evidence/*/config.json`，没有自动下载或转换模型。

`scenario_v2/bench_scenarios.py` 是入口；`tp_worker.py` 是 worker extension；`fast_prefill.py` 管理有界公共 prefill 图；`runtime/prefill-only-20261110/code/prefill_runtime.py` 负责 0-token prime、末端状态发布和分支门控；`runtime/overlay/` 是私有导入源。

只生成 function、arg1…arg6，不生成 content。先物化公共前缀边界，再从新环境后缀继续计算，最后 batched decode。完整公共 KV 页共享，非对齐末页按 native copy-on-write 复制；GDN/conv 状态独立。各头自己的终止 token 来自 tokenizer，工具输出不执行。当前单逻辑请求串行，不是生产级多租户 PD 服务。

固定前缀缓存路径的独立单头对照保持严格；无缓存 batch/单头与整段冷重算的数值差异显式审计，`--allow-whole-cold-drift` 允许记录这些路线差异后继续，不表示全路径逐 token 等价。原始输出和失败运行见上级 evidence。应结合任务质量和数值审计使用本包，不能只用速度表判断模型能力。
