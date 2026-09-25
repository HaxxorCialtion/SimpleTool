# Reproducing and auditing the SimpleTool release

This guide covers the code, input snapshots, raw outputs, and reports committed on branch **`reproduction/bfcl-v3-4090`**. It is intended for researchers and AI coding agents reproducing the measurements on another machine.

## 1. Checkout and choose the experiment

```bash
git clone --branch reproduction/bfcl-v3-4090 https://github.com/HaxxorCialtion/SimpleTool.git
cd SimpleTool/simpletool_reproduction_release
```

All commands below run from this directory unless explicitly stated otherwise. No author-specific `/inspire/...` path is required for the portable commands. Historical manifests retain those paths as provenance; replace them with your own paths when starting a new run.

| Experiment | Checkpoint | Hardware used | Published evidence |
| --- | --- | --- | --- |
| Full local accuracy | `RT-Qwen3-4B` (v1, non-quantized) | H100 80 GB | `results/RT-Qwen3-4B-eager/`, `reports/RT-Qwen3-4B-eager/` |
| Three demo latency scenarios | `RT-Qwen3-4B-AWQ-v2` | RTX 4090 48 GB | `reports/speed-4090/` |
| BFCL-v3 single-question latency | `RT-Qwen3-4B-AWQ` (v1) | RTX 4090 48 GB | `reports/bfcl-speed-4090/v1/` |

Every checkpoint comes from `Cialtion/SimpleTool`, pinned to revision `aca7673dd2d90a919159613261ea374332e9d32f`. Model weights are downloaded from Hugging Face; they are not duplicated in Git. The code, benchmark fixtures, and measured outputs are included in the branch.

These are separate experiments. Do not combine their model versions, accuracy figures, prompts, head counts, stop settings, or timing claims.

## 2. What the BFCL speed result means

The author clarifies that the paper's **16 Hz** was measured directly on BFCL-v3. The three demo scenarios are supplementary measurements, not the source of the paper number.

The new public-checkpoint BFCL run measures one question at a time across six recovered single-turn subsets. On the available RTX 4090 48 GB:

| Condition | Questions | Mean | Median | p95 | Reciprocal-mean rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sequential traversal | 1,880 | 74.61 ms | 64.09 ms | 133.60 ms | 13.40 Hz |
| Reset prefix cache before each question | 1,880 | 99.74 ms | 90.34 ms | 167.76 ms | 10.03 Hz |
| Prime the identical question once, then time it | 1,880 | 65.57 ms | 56.81 ms | 118.69 ms | 15.25 Hz |

The ordinary traversal did **not** exactly reproduce 16 Hz. The repeated-input result is closer but excludes priming from request latency; including priming and loop overhead gives 6.94 measured questions/s for that phase. It must not replace the ordinary-traversal result.

`Hz = number_of_measured_questions / sum_of_generation_seconds`, equivalently `1000 / mean_latency_ms`. Neither the inverse median nor the average of individual question rates is this metric.

All eligible questions contribute to timing, including incorrect answers, two rows without usable GT, and the length-capped `exec_simple_15` in each mode. Measurements end when generation stops or hits the configured output cap; a capped call is not evidence of a complete successful tool call.

## 3. Input population and files

The full accuracy input snapshot contains 18 datasets and 5,647 rows. Its recovered local scoring rules yield 5,437 scored rows. The BFCL latency subset has:

| Dataset | Input rows | Excluded for capacity | Timed rows | Rows with usable GT |
| --- | ---: | ---: | ---: | ---: |
| `BFCL_v3_live_simple` | 258 | 14 | 244 | 244 |
| `BFCL_v3_simple` | 400 | 0 | 400 | 400 |
| `BFCL_v3_live_multiple` | 1,053 | 167 | 886 | 885 |
| `BFCL_v3_multiple` | 200 | 0 | 200 | 200 |
| `BFCL_v3_exec_multiple_dataset` | 50 | 0 | 50 | 49 |
| `BFCL_v3_exec_simple_dataset` | 100 | 0 | 100 | 100 |
| **Total** | **2,061** | **181** | **1,880** | **1,878** |

A row is excluded when any candidate tool exceeds six properties. This is a model-capacity selection rule; it changes the evaluated population. The complete list of excluded and measured IDs is recorded in the BFCL `run.json`. Missing/invalid GT does not exclude an otherwise supported row from latency.

`data/` and `data/possible_answer/` contain the recovered inputs and answers. Many `.json` files are newline-delimited JSON objects, not a single JSON array. Use the supplied readers. `data/manifest.json` pins input and GT hashes and records the known malformed GT line and ID mismatch. Do not silently repair the snapshot or replace it with current upstream data while retaining the same run label.

See [DATA_SOURCES.md](DATA_SOURCES.md) for source families and preserved license notices. Multilingual strings in original datasets, predictions, logs, and historical code snapshots are retained byte-for-byte; all release-facing explanatory documents are in English.

## 4. Environment

The measured BFCL runtime used Python 3.12, vLLM 0.11.0, torch 2.8.0, transformers 4.57.6, compressed-tensors 0.13.0, and pydantic 2.13.4. The GPU reported 49,140 MiB, driver 570.195.03, and a 450 W power limit. The host group provided CUDA 12.8; the container was `ngc-pytorch:25.02-cuda12.8.0-py3`. Equivalent compatible CUDA environments may work, but latency and generated tokens can vary across software and hardware.

A starting point on a new compatible machine is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The base vLLM environment expects an older compressed-tensors package. The v1 quantized checkpoint uses newer metadata, including a meaningful asymmetric zero-point dtype. The recorded run used an isolated import overlay rather than modifying the base environment or checkpoint:

```bash
CT_OVERLAY="$PWD/.runtime/ct013"
python -m pip install --no-deps --target "$CT_OVERLAY" compressed-tensors==0.13.0
export PYTHONPATH="$CT_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=8 VLLM_WORKER_MULTIPROC_METHOD=spawn
python -c 'import importlib.metadata as m; print({n:m.version(n) for n in ["vllm","torch","transformers","compressed-tensors","pydantic"]})'
```

Use a fresh overlay directory rather than mixing versions. v1 stores asymmetric int4 weights with group size 128 and bfloat16 model dtype. Its folder name includes AWQ, but the serialization/loader is `compressed-tensors` with the Marlin W4A16 kernel. Do not force an AWQ loader based solely on the directory name, and do not remove meaningful zero-point metadata.

The earlier v2 demo used a different, explicitly recorded compatibility view that removed only two null config fields for compressed-tensors 0.11. That procedure is not appropriate for this v1 checkpoint. See [SPEED_BENCHMARK.md](SPEED_BENCHMARK.md).

The measurements are on a 48 GB RTX 4090, not a stock 24 GB card. Memory reservation and available cache capacity are recorded; this release does not claim identical performance on all RTX 4090 configurations.

## 5. Download, verify, and preflight

```bash
python scripts/download_model.py --variant RT-Qwen3-4B-AWQ --output models
python scripts/verify_model.py --model models/RT-Qwen3-4B-AWQ \
  --output reports/my-model-verification.json
python -m unittest discover -s tests -v
python -m simpletool_eval.bfcl_latency \
  --model models/RT-Qwen3-4B-AWQ --version v1 \
  --output reports/my-bfcl-run --preflight-only
```

Preflight validates the pinned data and supported input schemas without importing the GPU runtime. It should report 1,880 eligible questions across six datasets. The model verifier compares downloaded files to the pinned remote revision; directory names alone are not verification.

## 6. Run BFCL latency

First, optionally test a small subset in a new directory:

```bash
python -m simpletool_eval.bfcl_latency \
  --model models/RT-Qwen3-4B-AWQ --version v1 \
  --datasets BFCL_v3_simple --limit 8 --modes sequential \
  --output reports/my-bfcl-smoke
```

Then run the full measurement:

```bash
set -o pipefail
python -m simpletool_eval.bfcl_latency \
  --model models/RT-Qwen3-4B-AWQ --version v1 \
  --output reports/my-bfcl-run 2>&1 | tee reports/my-bfcl-run.log
python scripts/summarize_bfcl_latency.py --run reports/my-bfcl-run
```

Use a new output directory. The runner intentionally rejects an existing one. A smoke run cannot be presented as the full-suite result. For the default full run, each of `sequential.jsonl`, `cache_reset.jsonl`, and `warm_identical.jsonl` must contain 1,880 records. All three modes should appear in `summary.json`, and `run.json` must end with `status: complete`.

### Concurrency and head contract

A synchronous `LLM.generate` call receives only prompts for the heads of one question. It returns before the next question is submitted. There is no multi-question batching or asynchronous request dispatch. `max_num_seqs=8` permits simultaneous head streams within a question; it does not authorize eight questions in flight.

The requested heads are `function` plus the maximum argument count among the candidate schemas. There is no `content` head. Candidate schemas, not GT, determine the head count. Argument names follow the selected schema's property insertion order; unused heads beyond that selected schema's arity have no argument semantics.

### Stops and timing

Each head stops at `<|null|>`, any actual head closing tag (`</content>`, `</function>`, `</arg1>` through `</arg6>`), or normal ChatML/EOS termination. `</*>` is explanatory shorthand, not a literal wildcard. `skip_special_tokens=False` keeps these registered special tokens visible during string matching. Matched stop strings are excluded from returned text, and the stop reason is recorded separately.

Timing covers a complete synchronous generation call: input processing, vLLM scheduling, prefill, and all requested heads through stopping or the 128-token cap. It excludes prompt-string construction, result parsing/scoring, disk writes, network/HTTP, tool execution, and engine initialization. Maximum context is 8,192; overlong input fails instead of being truncated. Temperature and seed are zero.

The three cache modes and wall-time interpretation are specified in [BFCL_SPEED_PROTOCOL.md](BFCL_SPEED_PROTOCOL.md). Retain the engine log: the default run requests 1,883 resets. vLLM 0.11 discards Boolean reset results in wrappers, so the saved validation checks the engine's successful-reset log entries rather than treating Python `None` as confirmation.

## 7. Raw output schema

| File/field | Meaning |
| --- | --- |
| `run.json` | Status, settings, model/source/data hashes, runtime versions, GPU information, ordered IDs and exclusions |
| `sequential.jsonl` | One first-traversal measurement per supported question |
| `cache_reset.jsonl` | One measurement per question after explicit prefix-cache reset |
| `warm_identical.jsonl` | One measured second generation per question; `prime_ms` records the preceding generation separately |
| `dataset`, `id`, `order`, `mode` | Stable input identity, visit order, and cache condition |
| `latency_ms` | Wall time of the measured synchronous generation call |
| `head_count`, `raw_heads` | Number of requested streams and their returned text |
| `prompt_tokens`, `token_counts` | Per-head prompt and generated token counts |
| `finish_reasons`, `stop_reasons` | Whether a head stopped or hit its cap, and which marker stopped it |
| `legacy_correct`, `strict_correct`, `function_correct` | Local GT checks; `null` means GT is unavailable, not an excluded timing |
| `summary.json` | Overall/per-subset aggregates, priming cost, phase wall time and rates |
| `summary.md` | Human-readable tables and interpretation |
| `validation.json` | Recorded coverage, replay, hash and stop/reset audit findings |

Null stop text is not required to appear inside `raw_heads`: it is excluded by the sampling API and retained in `stop_reasons`. A capped head usually has no stop marker; do not reinterpret this as a successful null or closing-tag termination.

## 8. Audit the committed artifacts without a GPU

Validate the release checksums from this directory:

```bash
sha256sum --check ARTIFACT_SHA256SUMS.txt
```

The checksum list covers the other committed release files and excludes itself. It is a consistency check, not an independent signature. Compare Git revision and the remote model verification report when checking provenance.

Recompute the BFCL summaries in a scratch copy, preserving the published reports:

```bash
AUDIT_DIR=$(mktemp -d)
cp -R reports/bfcl-speed-4090/v1 "$AUDIT_DIR/run"
python scripts/summarize_bfcl_latency.py --run "$AUDIT_DIR/run" --data data
```

This validates status, input coverage/order, head sets, positive finite timings, local correctness replay, and summary arithmetic. The separate published `validation.json` additionally records model/source hash checks and log-based reset counts performed for the committed run. The summarizer rewrites its destination's report/validation files, which is why the example uses a copy.

All 5,640 recorded BFCL measurements were checked. The published raw measurement files match the validated originals byte-for-byte. The initial incomplete stopping-protocol run is excluded; its provisional 12.23 Hz must not be quoted as the final result.

## 9. Reproduce the separate accuracy experiment

Use a separate shell/environment without the quantized-model overlay if you want to match the earlier recorded accuracy environment. From this directory:

```bash
python scripts/download_model.py --variant RT-Qwen3-4B --output models
python scripts/verify_model.py --model models/RT-Qwen3-4B \
  --output reports/my-fp-model-verification.json
RESULT_DIR=results/my-accuracy REPORT_DIR=reports/my-accuracy \
  bash scripts/run_accuracy.sh --enforce-eager --disable-cascade-attn \
  --batch-size 64 --max-num-seqs 128
python -m simpletool_eval.strict --predictions results/my-accuracy \
  --data data --output reports/my-accuracy/strict
```

This accuracy runner batches questions and requests eight heads. It is intentionally distinct from the single-question speed runner. Its historical stop settings are preserved to reproduce its saved run; it should not be described as using the subsequently clarified BFCL latency protocol.

For CPU-only rescoring of the included accuracy outputs:

```bash
python -m simpletool_eval.evaluate --predictions results/RT-Qwen3-4B-eager \
  --output reports/my-accuracy-legacy-recheck
python -m simpletool_eval.strict --predictions results/RT-Qwen3-4B-eager \
  --data data --output reports/my-accuracy-strict-recheck
```

Read [AI_AGENT_EVAL_PROTOCOL.md](AI_AGENT_EVAL_PROTOCOL.md) and [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md) before changing scoring. The historical scorer tolerates extra decoded argument names; the strict local diagnostic rejects them. Mobile Actions evaluates a single current call conditioned on supplied reference history, not a complete autonomous trajectory. BFCL Exec uses local value matching, not official program execution.

## 10. Reporting checklist and limitations

Report the checkpoint/revision, hardware including memory capacity, runtime versions, prompt and stopping contract, head policy, sample counts/exclusions, cache condition, mean/p50/p95, and how Hz was computed. State whether priming, startup, HTTP, and tool execution are included. Preserve incorrect and capped outputs in the declared timing population.

Do not make any of the following claims from these artifacts:

- Exact reproduction of 16 Hz in ordinary traversal: this run measured 13.40 Hz.
- An autoregressive speedup factor: no matched autoregressive baseline was measured here.
- Official BFCL leaderboard accuracy: the supplied scorers are local matchers.
- Complete v2 BFCL accuracy: the v2 measurements cover only the three demo latency scenarios.
- Retail 24 GB RTX 4090 performance: the measured card has 48 GB.
- Bitwise identical output on arbitrary environments: kernels, cache conditions and numerical execution can affect greedy decoding.

The tokenizer emitted a regex warning in the pinned environment. Its original files were retained, with hashes and raw outputs published. Each mode has its own correctness counts; do not substitute the non-quantized accuracy run's scores for the quantized latency checkpoint.

For issue responses, see [SPEED_ISSUE_RESPONSE_DRAFT.md](SPEED_ISSUE_RESPONSE_DRAFT.md) and [ISSUE_RESPONSE_DRAFT.md](ISSUE_RESPONSE_DRAFT.md). These are drafts, not automatically posted comments. Training code and its release schedule are outside this package.
