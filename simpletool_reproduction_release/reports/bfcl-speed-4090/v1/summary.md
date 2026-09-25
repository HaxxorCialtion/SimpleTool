# BFCL-v3 single-question latency results

**Paper context:** the author states the reported 16 Hz was measured directly on BFCL-v3. The earlier game/arm/avatar measurements are separate supplementary tests. The present public-checkpoint run is reported as observed, not fitted to 16 Hz.

Checkpoint: `RT-Qwen3-4B-AWQ`, prompt `v1`. See `run.json` for full model, data and source hashes.

Stops: `<|null|>` or any defined head closing tag, plus ChatML end/EOS. Per-head stop reasons are retained. Special-token skipping is disabled during matching.

```text
name, memory.total [MiB], driver_version, power.limit [W], clocks.max.sm [MHz], clocks.max.memory [MHz]
NVIDIA GeForce RTX 4090, 49140 MiB, 570.195.03, 450.00 W, 3105 MHz, 10501 MHz
```

Engine initialization: 255.11 s (outside request timing). Question concurrency: **1**. Head policy: function + max candidate schema arity; no content.

Dataset input: 2061; capacity exclusions: 181; timed questions per mode: 1880. Input order and excluded IDs are in `run.json`.

| Mode | Calls | Mean ms | p50 ms | p95 ms | Serial Hz | Phase wall Hz | Capped calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sequential | 1880 | 74.61 | 64.09 | 133.60 | 13.40 | 13.28 | 1 |
| cache_reset | 1880 | 99.74 | 90.34 | 167.76 | 10.03 | 9.37 | 1 |
| warm_identical | 1880 | 65.57 | 56.81 | 118.69 | 15.25 | 6.94 | 1 |

`serial_hz` = calls / summed generation seconds, not mean of per-call Hz. `phase_wall_hz` includes scoring/writing/reset overhead and all priming in warm-identical mode. Engine initialization and synthetic warmup are outside both rates.

Sequential is a first traversal without individual benchmark priming. Cache-reset removes cross-request prefix reuse. Warm-identical measures the second of two identical requests and excludes the first generation from its latency; it is a favorable diagnostic, not first-time query throughput.

| Mode | Dataset | N | Mean ms | p50 ms | p95 ms | Hz | Local legacy | Local strict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sequential | BFCL_v3_live_simple | 244 | 82.81 | 57.74 | 250.14 | 12.08 | 185/244 | 185/244 |
| sequential | BFCL_v3_simple | 400 | 57.34 | 51.17 | 94.16 | 17.44 | 364/400 | 364/400 |
| sequential | BFCL_v3_live_multiple | 886 | 77.93 | 70.10 | 126.22 | 12.83 | 618/885 | 618/885 |
| sequential | BFCL_v3_multiple | 200 | 73.31 | 69.35 | 113.30 | 13.64 | 179/200 | 179/200 |
| sequential | BFCL_v3_exec_multiple_dataset | 50 | 96.32 | 69.54 | 212.45 | 10.38 | 43/49 | 42/49 |
| sequential | BFCL_v3_exec_simple_dataset | 100 | 85.97 | 57.07 | 173.29 | 11.63 | 91/100 | 89/100 |
| cache_reset | BFCL_v3_live_simple | 244 | 99.11 | 74.05 | 268.58 | 10.09 | 181/244 | 181/244 |
| cache_reset | BFCL_v3_simple | 400 | 70.87 | 64.79 | 114.26 | 14.11 | 364/400 | 364/400 |
| cache_reset | BFCL_v3_live_multiple | 886 | 116.81 | 113.17 | 171.44 | 8.56 | 616/885 | 616/885 |
| cache_reset | BFCL_v3_multiple | 200 | 84.40 | 79.98 | 121.45 | 11.85 | 180/200 | 180/200 |
| cache_reset | BFCL_v3_exec_multiple_dataset | 50 | 107.14 | 79.22 | 224.62 | 9.33 | 42/49 | 41/49 |
| cache_reset | BFCL_v3_exec_simple_dataset | 100 | 92.62 | 65.41 | 175.22 | 10.80 | 91/100 | 89/100 |
| warm_identical | BFCL_v3_live_simple | 244 | 78.11 | 55.19 | 239.86 | 12.80 | 182/244 | 182/244 |
| warm_identical | BFCL_v3_simple | 400 | 54.52 | 48.87 | 96.94 | 18.34 | 364/400 | 364/400 |
| warm_identical | BFCL_v3_live_multiple | 886 | 67.11 | 63.65 | 106.56 | 14.90 | 615/885 | 615/885 |
| warm_identical | BFCL_v3_multiple | 200 | 57.72 | 55.04 | 97.44 | 17.32 | 179/200 | 179/200 |
| warm_identical | BFCL_v3_exec_multiple_dataset | 50 | 80.01 | 54.87 | 198.73 | 12.50 | 43/49 | 42/49 |
| warm_identical | BFCL_v3_exec_simple_dataset | 100 | 74.04 | 50.04 | 153.40 | 13.51 | 91/100 | 89/100 |

Incorrect, unscorable and capped predictions remain in timing aggregates. Local scoring is not official BFCL AST/execution scoring. This covers six recovered single-turn subsets, not every BFCL-v3 category. Quantized v1 latency-run accuracy is distinct from the previously published non-quantized accuracy run.

See [BFCL_SPEED_PROTOCOL.md](../../../BFCL_SPEED_PROTOCOL.md) for cache, head-count, stop, environment and measurement contracts. The 48 GB RTX 4090 differs from a retail 24 GB card. No HTTP overhead, tool execution or matched autoregressive baseline is measured here.

## Validation

All expected IDs occur once per mode in the recorded order. Head sets and positive finite timings were checked. Per-question legacy/strict correctness was replayed from raw outputs, and all reported aggregates were recomputed.

## Interpretation and run notes

The new sequential traversal measured 13.40 Hz, below the paper’s 16 Hz. Repeated-input latency gives 15.25 Hz, but includes an extra priming generation outside its request timer. This is not evidence that first-time questions achieve 16 Hz.

All 1,883 requested cache resets were confirmed by engine success logs. Null and closing-tag stops both occurred in every mode (see `validation.json`). `exec_simple_15` hit the 128-token cap once in every mode and was retained. Data/model/source hashes match the recorded run; model weights and config are unmodified. The tokenizer emits a regex warning in the pinned environment; original tokenizer files were retained.

Attempt a failed before timing because the reset wrapper returned None; attempt b was stopped after the author clarified null stopping. Their partial/invalid results are excluded. Attempt c completed with the final stop protocol and platform status `job_succeeded`; no running worker was retained. No code or run manifest from earlier published measurements was rewritten.
