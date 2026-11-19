# Training source and reconstruction boundary

The three directories are the actual 5% training lineages, fetched read-only from retained cloud sources on 2026-11-19:

| Directory | Models | Data treatment |
|---|---|---|
| qwen35_8head_scale_20261108 | 0.8B, 2B, 4B | original prompts, up to 16K packed segments |
| qwen35_9b_scale_20261108 | 9B | 8K repack, masked empty-think insertion, separate input/output rows |
| qwen35_moe_handoff_20261109 | Qwen3.6 35B-A3B, Qwen3.5 122B-A10B / 397B-A17B | 8K source lineage, DDP/layer/EP placement and optional frozen FP8 experts |

`common8.py`, `train_8head.py`, `packed_store.py`, preprocessing and merge/export programs are included. A few inherited docstrings still refer to four-head ancestors; actual `HEADS`/special-token constants and saved configuration govern the eight-head runs. There are 17 start/end/null tokens. LoRA r64/alpha128 targets trainable text projections; routed experts/router remain frozen in the supplied MoE setup. Read actual per-model configs for differences.

**The raw training corpus has not been published in this release.** Manifests identify source-file SHA256, exact zero-based selected lines, prompt-group splits, pack membership and token/mask hashes. They are not the corpus. Original sources were `09_train_ready.jsonl`, `09b_train_ready.jsonl` and `final_simplelove.jsonl`. Do not replace them with Diagnostic-800 or claim a full retraining reproduction from this Git checkout alone.

Portable path changes are limited to configs and preprocessing input/output locations. `MODIFICATIONS.md` describes packaging changes. No optimizer or enormous token caches are bundled.

## Data contract

Each raw row contains `prompt`, `tools`, `head_0_content`, `head_1_function`, `head_2_arg1` … `head_7_arg6`. The original raw prompt ends in `<|im_start|>assistant\n`. Branch targets contain their opening marker, body, matching close and `<|im_end|>`. Preserve tool/property order, mask prompt/branch-start positions, isolate packed segments and reset positional/recurrence boundaries. Split by prompt hash so corresponding branches do not cross train/validation.

If you have the three hash-matching raw sources and the trained tokenizer:

```bash
python qwen35/training/qwen35_8head_scale_20261108/prepare_scale.py \
  --fraction 0.05 --source-root /absolute/path/to/corpus \
  --tokenizer /absolute/path/to/trained-tokenizer --output /absolute/path/to/p005
export SFT_MODEL_DIR=/absolute/path/to/original-base
export SFT_DATA_DIR=/absolute/path/to/p005
export SFT_OUTPUT_DIR=/absolute/path/to/new-run
export SFT_CONFIG=$PWD/qwen35/training/qwen35_8head_scale_20261108/config_0p8b.py
torchrun --standalone --nproc_per_node=2 \
  qwen35/training/qwen35_8head_scale_20261108/train_8head.py
```

The trainer needs CUDA Torch, transformers with the Qwen3.5/3.6 implementations, FLA/FA2 varlen support, Triton/TileLang as selected by `common8.py`, accelerate, safetensors and numpy. This release has not performed a clean-machine retraining smoke; inference environment versions are not a training dependency lock.

The 9B repacker reads `SFT_SOURCE_PACKS_DIR` pointing at the verified small-model packs; the MoE data lineage uses that 8K result. Preserve filtered-length and empty-think differences, rather than pretending all models saw exactly identical token sequences.

## Checkpoint/export contract

A deliverable LoRA consists of `adapter_model.safetensors`, `adapter_config.json`, **`head_tokens.safetensors` and `tokenizer/`** plus run configuration and source base identity. Untied models have distinct trained input/output rows. PEFT adapter alone is incomplete. `merge_8head.py` / `merge_helpers.py` and `export_fp16.py` restore these rows, merge deltas and verify saved tensors. Some legacy exporters use the original run-directory convention and require that layout. They are supplied as research source, not as a universal export CLI.

`merge_8head.py --selftest` exercises block-FP8 quantize/dequantize roundtrip with Torch. Merging into block FP8 and loading with BF16 compute are separate operations. Never silently substitute a random base revision or omit the tokenizer/new token rows.

## Quantization research source

`quantize_rtn_awq.py SOURCE OUTPUT` is the actual group64 affine RTN converter with integer packing roundtrip checks. `build_fp8_view.py` preserves the original project-specific FP8 merge-view recipe; its source paths and hard links are historical, so adapt its locations before reuse. It is not an automatic download/build entrypoint. The public Hub artifacts are self-contained exports, unlike that internal linked evaluation view.
