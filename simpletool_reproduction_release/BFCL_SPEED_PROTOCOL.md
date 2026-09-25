# BFCL-v3 single-turn latency protocol

## Paper claim and supplementary experiments

The author clarifies that the paper's **16 Hz** was measured directly on BFCL-v3. The game, arm and avatar timings in `reports/speed-4090/` are supplementary demo measurements; they are not the source of that paper number. Sixteen calls per second corresponds to 62.5 ms per call when calculated as the reciprocal of mean latency.

This new experiment measures the released v1 W4A16 checkpoint on the recovered BFCL-v3 single-turn snapshot. It reports observed results without fitting the setup to 16 Hz. It is a supplementary reproduction with a pinned public checkpoint and environment; no assertion is made that every original paper measurement detail has been recovered.

## Exactly one question at a time

`python -m simpletool_eval.bfcl_latency` uses a synchronous loop. A single `LLM.generate` receives only the head prompts of **one question**, and finishes all those heads before the next question is submitted. There is no multi-question batching, async dispatch or concurrent request worker. vLLM `max_num_seqs=8` is capacity for streams within a question, not eight questions.

Each question requests `function` plus the maximum property count among its candidate tools (zero to six argument heads). No content head is requested. The head count is determined from candidate schemas, never from the answer or the selected ground-truth function. Properties preserve insertion order. This head policy follows the local speed script's design and is distinct from the fixed eight heads in the earlier batched accuracy run.

v1 uses the exact recovered accuracy prompt builder, and the author-confirmed union of `<|null|>`, all defined head closing tags (`</content>`, `</function>`, `</arg1>`–`</arg6>`) and ChatML end as stops, with normal EOS handling. `</*>` means those actual closing tags, not a literal wildcard string. Special tokens remain visible during stop matching (`skip_special_tokens=False`); matched stop strings are excluded from returned text. An empty argument head stopped by null is treated as absent. Per-head stop reasons are saved. Maximum output is 128 tokens/head, temperature 0, seed 0. Maximum context is 8192; overflow fails rather than truncating input. Source, dataset and model hashes are recorded.

## Dataset coverage

The six published local subsets are `simple`, `multiple`, `live_simple`, `live_multiple`, `exec_simple_dataset` and `exec_multiple_dataset`. This is not the entire official BFCL-v3 suite. There are 2,061 single-turn inputs. Excluding any row with a candidate schema exceeding six properties leaves **1,880** timed questions. These include two inputs without usable local GT: they are timed normally, but not included in accuracy denominators (1,878 scored).

The historical local speed script contains an unused per-head sampling-parameter list and passes only its first entry to vLLM. This new runner uses the explicit union of closing tags plus the author-confirmed null stop, rather than reproducing that accidental first-head-only stop setting. The old speed script also uses a different XML tool prompt; that prompt is not silently substituted for the released v1 multi-head format. These are reasons to label this a supplementary public-checkpoint measurement, not an exact replay of every paper setting.

Rows are visited in the recorded dataset/file order; every mode uses every eligible row exactly once. All incorrect and length-capped outputs remain in latency aggregates. Local legacy and strict matches accompany timing so short but incorrect outputs are visible. They are not official BFCL AST/execution scores.

## Three separately reported cache conditions

1. **sequential**: synthetic warmup, then cleared prefix cache and one pass through all questions. No benchmark question is individually primed. Previously processed questions can share prefixes, as in a running service.
2. **cache_reset**: the prefix cache is cleared before every question, with success verified from engine logs. Reset overhead is outside the request timer. Compiled kernels and model remain warm.
3. **warm_identical**: each question is first generated once as an untimed priming request, then the identical question is measured. This is a favorable repeated-input diagnostic, not first-time user-request throughput. Priming cost is recorded separately and included in phase wall time.

One complete synchronous `LLM.generate` is timed with a monotonic nanosecond clock. This includes vLLM tokenization, scheduling, prefill and all requested head completions. It excludes Python prompt construction, result parsing/scoring, JSON writes, tool execution and HTTP/network overhead. Engine initialization and phase wall times are recorded separately.

Report mean, p50 and nearest-rank p95 in milliseconds. `serial_hz = N / sum(request_seconds) = 1000 / mean_ms`; do not average per-question reciprocal latencies. Also report `phase_wall_hz`, which includes loop overhead, resets and (for the repeated-input diagnostic) priming. A paper comparison must specify which rate is being compared.

## Environment and commands

The v1 checkpoint is `Cialtion/SimpleTool/RT-Qwen3-4B-AWQ`, revision `aca7673dd2d90a919159613261ea374332e9d32f`. Its storage is compressed-tensors asymmetric int4, group size 128, with bfloat16 model dtype. Unlike the symmetric v2 checkpoint, v1 has a meaningful `zp_dtype` and must not have that field removed.

From `simpletool_reproduction_release/`:

```bash
python scripts/download_model.py --variant RT-Qwen3-4B-AWQ --output models
python scripts/verify_model.py --model models/RT-Qwen3-4B-AWQ --output reports/model-verification-awq-v1.json
# The existing environment has vLLM 0.11.0 / torch 2.8.0. Use an isolated
# compressed-tensors overlay; keep checkpoint metadata and base environment intact.
python -m pip install --no-deps --target /absolute/path/ct013 compressed-tensors==0.13.0
export PYTHONPATH=/absolute/path/ct013
export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
python -m simpletool_eval.bfcl_latency --model models/RT-Qwen3-4B-AWQ \
  --version v1 --output reports/bfcl-speed-new
```

Output must be a new directory. The default runs all six subsets and all three modes. `--limit` is for smoke tests only and is recorded in the manifest. `--version v2` exists for separate experiments and must not be silently substituted for v1. The GPU used here is an RTX 4090 with 48 GB, not the standard retail 24 GB configuration; driver, power limit and clock readings are reported with results.

## Offline validation

```bash
python scripts/summarize_bfcl_latency.py --run reports/bfcl-speed-new
```

The validator rejects incomplete runs, duplicate/missing IDs, wrong head sets and invalid timings. It replays local correctness from raw outputs and recomputes all latency aggregates. For the default three-mode run, retain engine logs and confirm 1,883 successful prefix resets (1,880 per-question resets plus one reset at the start of each mode). vLLM 0.11 discards reset Boolean values at EngineCore as well as at the public wrapper, so Python `None` is not positive evidence by itself.

## Stop-protocol correction history

The initial follow-up run omitted null stopping and used default special-token skipping. It was interrupted after the author clarified the intended stop protocol. Its partial 12.23 Hz sequential result is not the final-protocol measurement and must not be presented as the reproduction result. The final run explicitly adds null and preserves special-token visibility for matching. The three older demo measurements retain their original configuration and must not be relabelled as having this newly verified setting.

## Completed measurement

See [full results and raw outputs](reports/bfcl-speed-4090/v1/summary.md). On the available RTX 4090 48 GB, v1 W4A16 completed all 1,880 questions per mode with the author-confirmed null/closing-tag stop protocol. Mean latencies and reciprocal-mean rates were:

| Condition | Mean latency | Serial rate |
| --- | ---: | ---: |
| Sequential traversal | 74.61 ms | 13.40 Hz |
| Prefix cache reset before each question | 99.74 ms | 10.03 Hz |
| Identical question primed once before timing | 65.57 ms | 15.25 Hz |

This run does **not** establish 16 Hz for first-time questions across these six subsets. The favorable repeated-input condition comes close but excludes priming from its request timer; its full phase wall rate including priming is 6.94 Hz. Do not replace the sequential result with this diagnostic. All three modes retained the length-capped `exec_simple_15` and two unscorable GT cases in timing. The tokenizer emitted a regex warning; its files were left unchanged and their hashes are included.

The quantized checkpoint produces different local correctness from the earlier non-quantized accuracy run. Each mode publishes its own local match counts; cache conditions can alter floating-point execution and token choices, so identical greedy outputs across modes are not assumed.
