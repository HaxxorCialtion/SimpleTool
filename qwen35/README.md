# SimpleTool Qwen3.5 / Qwen3.6 eight-branch release

First source integration: 2026-11-19. [中文交付核对](RELEASE_STATUS_zh.md). The original paper reproduction remains in the repository root and `simpletool_reproduction_release/`. This directory contains the newer checkpoint family and deployment research.

Each logical branch uses the same backbone and projection weights. One common prompt is followed by `<function>`, `<arg1>` … `<arg6>` and, optionally, `<content>`. Each branch is autoregressive and stops at its own closing marker. Arguments do not consume the function branch's newly generated output. A call's critical path includes common prefill, state publication/fork, branch markers and the longest required argument/function branch.

## Start here

1. [Models: 7 sizes / 11 public formats](models/README.md), pinned revisions and inventories.
2. [Reference HTTP client](reference/INFERENCE.md): readable prompt/position protocol, no custom CUDA patch required.
3. [CUDA exact-prefix backend](inference/cuda/README.md): shared KV, independent GDN/conv, zero-output prime, bounded prefill/decode graphs and optional content.
4. [Apple / iPhone backend](inference/apple/README.md): packed INT4, A16/KV8 option, FP32 GDN, batched heads and memory ownership guidance.
5. [Training code and selection manifests](training/README.md).
6. [Frozen evaluation suites](evaluation/README.md) and [recorded quality results](results/all_model_scores.csv).
7. [Fixed-prefix performance protocol and results](benchmarks/fixed_prefix/REPORT.md).
8. [Release status and limitations](RELEASE_STATUS.md), [third-party notices](NOTICE.md).

## Minimal reference inference

Install a Qwen3.5-capable vLLM environment separately. For example, start a small model (hardware memory requirements still apply):

```bash
vllm serve Cialtion/SimpleTool-Qwen3.5-0.8B-8Head-5pct-BF16 \
  --dtype bfloat16 --enable-prefix-caching --max-model-len 4096
python qwen35/reference/infer_heads.py \
  --model Cialtion/SimpleTool-Qwen3.5-0.8B-8Head-5pct-BF16 --protocol direct
```

This is an illustrative BF16 serving command, not a newly measured BF16 score. The reference client uses concurrent independent completion requests; stock vLLM is not guaranteed to deduplicate their initial prefill. Use the exact-prefix backend for explicit common-state materialization and fork.

Required properties must be first and `required` must preserve their property insertion order. Arg slots follow the selected tool's property order. Maximum six properties per tool. Ignore outputs beyond the chosen tool's arity. Zero-argument tools retain a null `arg1` in the deployment protocol. Tool execution belongs to the application; these clients only print proposals.

## CPU checks

```bash
python qwen35/check_release.py
```

This checks reference assembly, service admission/content contracts, frozen data and scoring tests. It does not allocate a GPU, execute a tool, download model weights, or claim a fresh iPhone/GPU inference result.

The release contains training source, preprocessing, merge/export code and exact selection metadata. **The original training corpus is not included or publicly downloadable through this release yet.** Diagnostic-800 is evaluation data and must not be used as its substitute. This is a first source/weights/evaluation release, not a claim of complete training-data availability.
