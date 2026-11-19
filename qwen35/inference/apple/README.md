# Apple Silicon / iPhone source snapshot

`SimpleToolSwift/` contains the actual four-channel and eight-branch engines, tokenizer, static KV tail, quantized attention and Metal resources. `Vendor/mlx-swift/` is the matching complete source dependency, including C/C++ MLX, with its upstream license. `native/` preserves standalone Metal operator probes. `operator_lab/` preserves the Python operator exploration and recorded result documents; it is a separate historical four-channel research interface.

## Runtime

- One common prefill, batched forward, a single stored full-attention prefix and independent branch GDN/conv state.
- Packed affine INT4/group64 weights. Explicit A16 mode narrows supported floating tensors; GDN recurrent matrices/gates retain FP32. KV8 means quantized attention KV, not INT8 activation arithmetic across the model.
- `EightHeadSession` takes a complete raw prompt; the four-channel `ChatSession` is a different ABI. Trained tokenizer and markers must match the model. A 5% public CUDA AWQ checkpoint cannot be loaded as a Swift packed artifact directly.
- Independent stop/active-batch compaction, single-owner inference lane, cancellation at operation boundaries, explicit unload and compile-cache clearing.

Read [docs/PITFALLS.md](docs/PITFALLS.md) for memory-limit waiting, compiled tapes retaining weights, autorelease pools, ordinary iOS signing budgets, RoPE batch issues, norm/conv conversion, tokenizer order and numerical differences. Older absolute paths/build numbers in historical documents identify original evidence, not portable install paths.

## Build and conversion

Requires Apple Silicon, compatible Xcode/Swift and Metal toolchain. Build via Xcode so the MLX metallib and kernel resources are bundled:

```bash
bash qwen35/inference/apple/Scripts/build_smoke.sh /tmp/simpletool-infra-build
```

Run the printed `InfraSmoke` command with an existing converted model directory and a new receipt path. The conversion entry point is `Scripts/export_int4.py`; its source-download receipt requires original revisions/file hashes and is intentionally not fabricated. Conversion checks source finite values, offset RMSNorm (+1), conv layout, shard reload identity and tokenizer copying. Base shape/tied-embedding assumptions are checked; a new model family needs an explicit adaptation.

Historical device results for the three dense sizes and eight-branch prototype are retained in the original project. This release does not claim a fresh device test or build of every public checkpoint. No weights, IPA, signing profile, device ID, audio/embedding models or App assets are bundled here. The source tree is intended for an application's own integration.

## Operator lab

Python modules need compatible MLX and MLX-LM (recorded locally: see `operator_lab/requirements.txt`). `operator_lab/docs/` describes accepted and rejected candidates: row-reused/grouped INT4 projection, SwiGLU fusion, GDN gate/conv/prepare fusion, deferred/fixed-history recurrence, state sharing, static graph scheduling and shader compiler experiments. Presence does not imply enabled by default or a speedup. Preserve each document's precision/device/test limits.
