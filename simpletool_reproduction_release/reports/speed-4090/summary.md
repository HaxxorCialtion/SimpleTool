# RTX 4090 W4A16 latency results

Measured 2026-09-25. Model `Cialtion/SimpleTool/RT-Qwen3-4B-AWQ-v2`, revision `aca7673dd2d90a919159613261ea374332e9d32f`.

Hardware: RTX 4090 48 GB (49140 MiB), driver 570.195.03, 450 W power limit. vLLM 0.11.0, torch 2.8.0, compressed-tensors 0.11.0; W4A16 Marlin, bfloat16 activations, Flash Attention, CUDA graphs and prefix caching enabled. Two null config fields adapted as documented in `compatibility.json`; weights unchanged.

Timing is complete synchronous local `LLM.generate` for all heads in one call, no HTTP. One logical call in flight. Function + maximum candidate-tool arity heads; no content head. See [protocol](../../SPEED_BENCHMARK.md).

| Scenario | Cache regime | N | Mean ms | p50 ms | p95 ms | Function smoke matches |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Game — Tower Defense | first_seen | 1 | 98.65 | 98.65 | 98.65 | 1/1 |
| Game — Tower Defense | warm_identical | 30 | 31.70 | 31.22 | 33.06 | 30/30 |
| Game — Tower Defense | cache_reset | 30 | 77.70 | 77.36 | 78.92 | 30/30 |
| Game — Tower Defense | shared_prefix_new_query | 30 | 42.19 | 41.45 | 46.64 | 30/30 |
| Robotic Arm — Assembly | first_seen | 1 | 95.84 | 95.84 | 95.84 | 1/1 |
| Robotic Arm — Assembly | warm_identical | 30 | 58.25 | 58.27 | 64.66 | 30/30 |
| Robotic Arm — Assembly | cache_reset | 30 | 93.67 | 92.99 | 98.17 | 30/30 |
| Robotic Arm — Assembly | shared_prefix_new_query | 30 | 59.00 | 58.51 | 64.87 | 30/30 |
| Digital Human — Streamer | first_seen | 1 | 91.94 | 91.94 | 91.94 | 1/1 |
| Digital Human — Streamer | warm_identical | 30 | 41.53 | 41.44 | 42.87 | 30/30 |
| Digital Human — Streamer | cache_reset | 30 | 89.03 | 88.91 | 90.51 | 30/30 |
| Digital Human — Streamer | shared_prefix_new_query | 30 | 53.47 | 53.02 | 56.99 | 30/30 |

Three scenarios × (one first-seen + 30 × three regimes) = 273 recorded calls; additionally 15 untimed warmups. No length-capped heads. All function names matched, but the check does not validate all arguments. All 90 cache resets were confirmed successful in `gpu.log`. Medians recomputed from `samples.jsonl`, source/prompt hashes checked.

Game: 3 heads, 917 prompt tokens/head, longest output 4 tokens. Arm: 5 heads, 920 prompt tokens/head, longest output 5 tokens. Avatar: 3 heads, 965 prompt tokens/head, longest output 6 tokens (original prompts). Changed-query trials append synthetic identifiers and may change lengths.

Warm-identical mean across the three equal-sized scenarios: 43.83 ms. Cache-reset mean: 86.80 ms. These short calls support tens-of-milliseconds warm-cache latency; they do not establish universal 60 ms latency or a paper speedup factor. No matched autoregressive baseline was run.

The unchanged original demo separately measured 32.8 / 62.5 / 42.9 ms (game/arm/avatar), mean 46.1 ms, on another 48 GB RTX 4090 node with the same driver. Its raw log is included. Do not combine those single measurements with the repeated run.

## Startup and attempts

Final repeat-run engine initialization: 203.18 seconds, excluding Python imports. Runtime CUDA compilation is excluded from request latency. `first_seen` is measured after engine initialization.

- Attempt a: CUDA 13 image incompatible with CUDA 12.8 host; stopped before inference.
- Attempt b: unmodified model config rejected by old compressed-tensors parser.
- Attempt c: original demo succeeded; repeat-run cache reset succeeded, but the script misread the V1 wrapper’s `None` return as failure. Its partial measurements were excluded.
- Attempt d: corrected reset check, all 273 measurements completed, platform `job_succeeded`.

The platform released the successful worker; no retained running benchmark instance was requested. Platform metrics returned four coarse samples spanning startup; these do not resolve individual timed requests. `run.json` contains before/after GPU readings, not continuous per-request telemetry. The tokenizer emits a regex warning in the pinned environment; the released tokenizer was retained unchanged and outputs/token lengths are published.
