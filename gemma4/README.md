# SimpleTool Gemma4 eight-branch adapters

Five completed 5% training checkpoints: E2B, E4B, 12B, 26B-A4B and 31B. **These are complete adapter packages, not merged standalone models.** Each requires the corresponding original `google/gemma-4-<size>-it` base and the custom reconstruction loader.

| Size | Hugging Face | ModelScope | Required-call matches | Function matches |
|---|---|---|---|---|
| E2B | [HF](https://huggingface.co/Cialtion/SimpleTool-Gemma4-E2B-8Head-5pct-LoRA) | [ModelScope](https://modelscope.cn/models/cialtion/SimpleTool-Gemma4-E2B-8Head-5pct-LoRA) | 469/800 | 752/800 |
| E4B | [HF](https://huggingface.co/Cialtion/SimpleTool-Gemma4-E4B-8Head-5pct-LoRA) | [ModelScope](https://modelscope.cn/models/cialtion/SimpleTool-Gemma4-E4B-8Head-5pct-LoRA) | 565/800 | 775/800 |
| 12B | [HF](https://huggingface.co/Cialtion/SimpleTool-Gemma4-12B-8Head-5pct-LoRA) | [ModelScope](https://modelscope.cn/models/cialtion/SimpleTool-Gemma4-12B-8Head-5pct-LoRA) | diagnostic only | not reported |
| 26B-A4B | [HF](https://huggingface.co/Cialtion/SimpleTool-Gemma4-26B-A4B-8Head-5pct-LoRA) | [ModelScope](https://modelscope.cn/models/cialtion/SimpleTool-Gemma4-26B-A4B-8Head-5pct-LoRA) | 577/800 | 781/800 |
| 31B | [HF](https://huggingface.co/Cialtion/SimpleTool-Gemma4-31B-8Head-5pct-LoRA) | [ModelScope](https://modelscope.cn/models/cialtion/SimpleTool-Gemma4-31B-8Head-5pct-LoRA) | 643/800 | 784/800 |

All five complete adapter packages are public on both Hugging Face and ModelScope as of 2026-11-19. Anonymous file access and matching weight sizes were checked; no new full inference run is implied.

Scores are historical 2026-11-17 measurements on frozen [Diagnostic-800](../qwen35/evaluation/diagnostic800/README.md), using its typed literal matcher and schema validity gate, not an official BFCL score. 12B produced repeated malformed outputs and evaluation was halted; no 0/800 ability score is assigned. Vision, content and a parallel Gemma speed backend have not been accepted.

## Artifact contract

Download the whole model repository, including `adapter_model.safetensors`, `head_tokens.safetensors`, `tokenizer/`, configuration and loader. E-series restores per-layer embeddings; 26B-A4B additionally requires `grouped_expert_lora.safetensors`. PEFT loading alone omits trained parts. Base and adapters retain their respective upstream licensing; published base metadata and package LICENSE specify Apache2.0.

## Inference

The original environment used Python3.12, Torch2.11.0 CUDA12.8, transformers5.17.0, PEFT0.21.3, safetensors and accelerate. Obtain the matching base and complete adapter package locally:

```bash
python /path/to/adapter-package/infer.py --base /path/to/base \
  --artifact /path/to/adapter-package --messages messages.json --args 2
```

`messages.json` is a list of system/user messages with tool definitions and the request. The reference sequentially decodes function and selected args; `--content` opts in to content. It prints proposals without executing tools. Args follow the trained position mapping; maximum six top-level properties. Required-list order comes first, optional properties use sorted order, and prompt definitions and targets must be transformed together. See the actual `source/prepare_data.py` implementation.

## Training source snapshot

`source/` contains the original training, data preparation, custom text model loading, PLE and grouped-expert LoRA logic. Publication changes only replace machine-specific source/config paths. Edit each config's `model_dir`, `data_dir`, `output_dir`; set `SIMPLETOOL_TRAINING_DATA` to the original three-JSONL pool. Preparation writes caches under `source/data/`; align the training config's data directory with that result. `prepare_31b_reuse.py` requires independently verified 26B/31B tokenizer and template equality.

This release has syntax checks, not a fresh clean-machine training or Hub weight reload acceptance. The full training corpus is being published separately by the cloud data agent; evaluation data is not a substitute.

## Frozen Diagnostic-800 evaluation

`source/evaluate_gemma.py` is the actual native batched evaluator with only machine-path/CLI adaptation. It preserves frozen bodies, replaces the ChatML role shell with the trained native Gemma shell, decodes no content, stops each branch independently and uses the frozen local scorers. Example from the repository root:

```bash
python gemma4/source/evaluate_gemma.py --model e4b \
  --base /path/to/gemma-4-E4B-it --artifact /path/to/complete-adapter \
  --datasets qwen35/evaluation/diagnostic800/frozen --output gemma-evaluation
```

The historical minimum batches are E2B/E4B=32, 12B/26B=16 and 31B=8, requiring suitable GPU memory. Do not use 12B for a reported ability result before resolving its documented abnormal output. This reference is not the optimized Qwen shared-state CUDA path. Publication changes have syntax/contract checks; no new full Gemma run is claimed.
