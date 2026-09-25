# RTX 4090 latency benchmark

The author clarifies that the paper’s 16 Hz was measured directly on BFCL-v3. The three demo scenarios below are supplementary tests, not the source of that number. See [BFCL single-turn protocol](BFCL_SPEED_PROTOCOL.md) for the separate question-concurrency-one experiment.

Completed results: [4090 report](reports/speed-4090/summary.md).

Download the pinned variant from the repository root:

```bash
python simpletool_reproduction_release/scripts/download_model.py --variant RT-Qwen3-4B-AWQ-v2 --output models
python simpletool_reproduction_release/scripts/verify_model.py --model models/RT-Qwen3-4B-AWQ-v2 --output /tmp/speed-model-verification.json
```

For the supplied older environment, first create the compatibility view described below. Run from the repository root using the same environment as `01_benchmark.py`:

```bash
python simpletool_reproduction_release/scripts/benchmark_latency.py \
  --model /path/to/RT-Qwen3-4B-AWQ-v2 \
  --version v2 --repeats 30 --warmup 5 --output /path/to/new-results
```

The output directory must not exist. The script uses the repository's actual prompt builder and scenarios; prompt/source hashes, runtime versions, GPU details, every measured raw output and token count are recorded. Match the model verification manifest as well as the directory name. v1 weights require `--version v1`.

## Timing contract

One logical tool call in flight. Its function and argument streams are generated together by one synchronous `LLM.generate` call, with no HTTP server or network. Wall time includes vLLM input processing, scheduling, prefill and completion of every requested head. It excludes model loading, prompt string construction, JSON reconstruction and tool execution. It is not isolated decode latency or TTFT. Engine initialization is recorded separately.

Heads are `function` plus 1–6 arguments, chosen by the largest arity among available tools, exactly as the public `--n-args auto` demo. The content head is omitted. This differs from the accuracy run's fixed eight heads. No argument-head reduction based on the ground-truth chosen function is used.

Report each scenario separately: output length and head count affect latency. Function-name agreement is a smoke check only; argument correctness is not established by it. Length-capped heads are counted and all text retained, so incomplete outputs cannot silently masquerade as faster successful calls.

## Cache regimes

- `first_seen`: first encounter of a scenario, before per-scenario warmup. This is not engine initialization and is not guaranteed to have no shared prefix with earlier scenarios.
- `warm_identical`: five untimed warmups, then 30 identical prompt executions. This is a favorable repeated-input prefix-cache regime.
- `cache_reset`: prefix cache is explicitly reset before each of 30 calls. Model compilation/graph warmup remains completed. Reset time is excluded; the following generation pays prefill cost.
- `shared_prefix_new_query`: 30 synthetic query variations append distinct observation identifiers. Most of the prefix is shared; a short changed suffix must be processed. These are controlled cache probes, not 30 independent realistic user tasks.

Each row reports mean, median, nearest-rank p95, function-name smoke matches and truncation count. Every trial, including incorrect outputs, contributes to timing. Three curated scenarios are not a workload-wide BFCL speed benchmark, and do not establish a paper speedup ratio: that requires a matched autoregressive baseline, dataset, quantization and output contract.

## Original demo labels

`01_benchmark.py` calls repeated complete generations “Hot prefill” and divides complete call time by longest-head output tokens as “decode bottleneck.” Neither quantity isolates prefill or decode. Its cold pass also excludes model initialization. Preserve its output as historical demo evidence, but use the timing definitions above for new claims.

The available platform advertises NVIDIA 4090 **48 GB**, not a standard retail 24 GB card. Publish observed driver, memory, power and clocks with results; do not assume equivalence with all RTX 4090 systems.

## Compatibility observed in the provided environment

The checkpoint directory is named `RT-Qwen3-4B-AWQ-v2`, but its serialization is `compressed-tensors`, symmetric int4, group size 128, with bfloat16 model dtype. Do not force `quantization=awq` merely because the directory contains AWQ.

The supplied vLLM 0.11.0 environment contains compressed-tensors 0.11.0. The checkpoint was written with compressed-tensors 0.13.0 and includes `scale_dtype: null` and `zp_dtype: null`, which the older config parser rejects. Create an explicitly separate compatibility view:

```bash
python simpletool_reproduction_release/scripts/prepare_speed_compat.py \
  --source /path/to/RT-Qwen3-4B-AWQ-v2 \
  --output /path/to/RT-Qwen3-4B-AWQ-v2-ct011
```

Then benchmark the compatibility directory. Only those two null fields are removed; meaningful non-null values are rejected, and weights/tokenizer files are symlinked unchanged. `compatibility.json` records original and adapted configuration hashes. Do not run the unmodified-HF config verifier against this adapted config; verify the original first. The view depends on the original directory remaining accessible.

Run with `USE_TF=0 USE_FLAX=0 VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=8` to match the measured process environment. The original environment is not modified.

The repeat runner uses vLLM 0.11's synchronous engine utility for cache reset although both the public V1 wrapper and EngineCore discard the scheduler’s Boolean success return; success is verified from the engine log. This internal API is version-specific; use the pinned environment when reproducing it. The recorded GPU log must show successful cache resets (90 for three scenarios × 30 trials).

The later [BFCL-specific runner](BFCL_SPEED_PROTOCOL.md) explicitly uses null plus every defined closing tag with `skip_special_tokens=False` and records stop reasons. The demo measurements above retain the original script’s special-token-skipping default; they are historical supplementary measurements, not a replacement for that verified BFCL stop protocol.
