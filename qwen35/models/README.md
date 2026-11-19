# Released 5% checkpoint registry

All 11 entries were anonymously visible on both hubs on 2026-11-19. The pinned revisions and artifact sizes are in [registry.json](registry.json). These are full merged checkpoints; no separate LoRA is needed.

| Artifact | Hugging Face | ModelScope | Weight GiB |
|---|---|---|---|
| Qwen3.5-0.8B-8Head-5pct-BF16 | [download](https://huggingface.co/Cialtion/SimpleTool-Qwen3.5-0.8B-8Head-5pct-BF16) | [mirror](https://modelscope.cn/models/cialtion/SimpleTool-Qwen3.5-0.8B-8Head-5pct-BF16) | 1.63 |
| Qwen3.5-0.8B-8Head-5pct-INT4 | [download](https://huggingface.co/Cialtion/SimpleTool-Qwen3.5-0.8B-8Head-5pct-INT4) | [mirror](https://modelscope.cn/models/cialtion/SimpleTool-Qwen3.5-0.8B-8Head-5pct-INT4) | 0.95 |
| Qwen3.5-122B-A10B-8Head-5pct-FP8 | [download](https://huggingface.co/Cialtion/SimpleTool-Qwen3.5-122B-A10B-8Head-5pct-FP8) | [mirror](https://modelscope.cn/models/cialtion/SimpleTool-Qwen3.5-122B-A10B-8Head-5pct-FP8) | 118.43 |
| Qwen3.5-2B-8Head-5pct-BF16 | [download](https://huggingface.co/Cialtion/SimpleTool-Qwen3.5-2B-8Head-5pct-BF16) | [mirror](https://modelscope.cn/models/cialtion/SimpleTool-Qwen3.5-2B-8Head-5pct-BF16) | 4.24 |
| Qwen3.5-2B-8Head-5pct-INT4 | [download](https://huggingface.co/Cialtion/SimpleTool-Qwen3.5-2B-8Head-5pct-INT4) | [mirror](https://modelscope.cn/models/cialtion/SimpleTool-Qwen3.5-2B-8Head-5pct-INT4) | 2.37 |
| Qwen3.5-397B-A17B-8Head-5pct-FP8 | [download](https://huggingface.co/Cialtion/SimpleTool-Qwen3.5-397B-A17B-8Head-5pct-FP8) | [mirror](https://modelscope.cn/models/cialtion/SimpleTool-Qwen3.5-397B-A17B-8Head-5pct-FP8) | 378.26 |
| Qwen3.5-4B-8Head-5pct-INT4 | [download](https://huggingface.co/Cialtion/SimpleTool-Qwen3.5-4B-8Head-5pct-INT4) | [mirror](https://modelscope.cn/models/cialtion/SimpleTool-Qwen3.5-4B-8Head-5pct-INT4) | 3.83 |
| Qwen3.5-9B-8Head-5pct-INT4 | [download](https://huggingface.co/Cialtion/SimpleTool-Qwen3.5-9B-8Head-5pct-INT4) | [mirror](https://modelscope.cn/models/cialtion/SimpleTool-Qwen3.5-9B-8Head-5pct-INT4) | 8.58 |
| Qwen3.5-4B-8Head-5pct-BF16 | [download](https://huggingface.co/Cialtion/SimpleTool-Qwen3.5-4B-8Head-5pct-BF16) | [mirror](https://modelscope.cn/models/cialtion/SimpleTool-Qwen3.5-4B-8Head-5pct-BF16) | 8.68 |
| Qwen3.5-9B-8Head-5pct-BF16 | [download](https://huggingface.co/Cialtion/SimpleTool-Qwen3.5-9B-8Head-5pct-BF16) | [mirror](https://modelscope.cn/models/cialtion/SimpleTool-Qwen3.5-9B-8Head-5pct-BF16) | 17.98 |
| Qwen3.6-35B-A3B-8Head-5pct-FP8 | [download](https://huggingface.co/Cialtion/SimpleTool-Qwen3.6-35B-A3B-8Head-5pct-FP8) | [mirror](https://modelscope.cn/models/cialtion/SimpleTool-Qwen3.6-35B-A3B-8Head-5pct-FP8) | 34.89 |

Small checkpoints: BF16 and affine RTN/group64 INT4 in AWQ layout, without activation calibration. Large checkpoints: block FP8 with retained higher-precision tensors. **Use BF16 compute for FP8, especially 397B.** Do not apply INT4 scores to the BF16 artifact as an independently measured result. The 35B base is Qwen3.6; the other bases are Qwen3.5.

Hub INT4 is not the Apple MLX packed format. Convert from the corresponding full-precision checkpoint with the Apple exporter; validate conversion and inference for that exact revision. The Apple source snapshot has historical four/eight-head device evidence, not a new device validation of every public 5% weight.
