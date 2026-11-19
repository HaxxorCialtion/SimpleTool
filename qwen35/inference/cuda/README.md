# Qwen3.5 hybrid-state CUDA backend

This is the tested private vLLM source overlay and service snapshot, made portable without changing the numerical kernels. It is tied to internal vLLM APIs: vLLM 0.33.0, Torch 2.14.0+cu130, Triton 3.8.0, transformers 5.20.2, flashinfer-python/cubin 0.7.2.post1, CUDA 13, Hopper SM90. See `infra/ENVIRONMENT.json`. Installing arbitrary latest vLLM is not supported.

## Mechanism

`infra/runtime/prefill-only-20261110/code/prefill_runtime.py` creates a zero-output public prime and publishes its exact end state before head requests become runnable. `infra/runtime/code/fork_gate.py` gates scheduler publication. The overlay scheduler and `prefix_match_unit=1` permit exact token boundaries. Full attention KV pages are shared; nonaligned final pages use copy-on-write. GDN matrices and causal-convolution histories must become independent writable state after fork.

`export_endpoint.py`, the GDN attention overlay and `chunk_delta_h.py` export an endpoint checkpoint plus the short residual recurrence. `fast_prefill.py` caches bounded graphs; `tp_worker.py` controls capture/verification on every TP worker. `rebase.py` validates overlay hashes and writes only this package's private manifest; it does not patch the installed vLLM files.

One owner thread handles one logical request at a time, with a bounded HTTP admission queue. Each logical request batches its active heads, stops them independently and cancels unused argument tails once the selected function is known. `include_content` is decided per request. Content completion has separate timing and must not inflate tools-ready latency.

## Start

Download a full model first. Copy `engine.example.json` to `engine.json`, set the absolute model path and choose TP/cache/context/capture sizes for the target device. `tp=4` is an example from 397B, not the setting for all models. Use a separate installation matching the tested versions.

```bash
cd qwen35/inference/cuda
cp engine.example.json engine.json
# Edit engine.json: model, tp, cache budget and initial scenarios.
export SL_GPU_PYTHON=/absolute/path/to/compatible/python
bash run_server.sh
```

The example binds localhost; expose it through your own authenticated service if needed. The stdlib client defaults to localhost and reads optional `INF_API_KEY` / `SIMPLETOOL_BASE_URL` from the environment.

```bash
python client.py ready
python client.py infer --json-file example_request.json
python client.py infer --json-file example_request_content.json --include-content
```

APIs: `/healthz`, `/readyz`, `/v1/config`, `/v1/metrics`, `/v1/scenarios`, `/v1/infer`. A scenario pins invariant system/tools/strategy text; each request supplies changing environment/history/query. No generated call executes here.

For the original fixed-prefix experiment use `infra/launch.sh`; see `../../benchmarks/fixed_prefix/REPORT.md`. Its launcher accepts explicit model/TP/output parameters and writes fresh results. It excludes content and does not implement a robot/game simulator.

## Boundaries

397B FP8 needs BF16 compute: historical FP16 intermediate overflow caused corrupted text. KV/GDN `auto` here is not Apple KV8. Numerical identity was tested for particular state splits; whole-prompt cold recomputation can diverge and is recorded. Graph/eager parity on a fixed batch is not proof of equality across arbitrary batches. CPU service tests use a fake engine. The source is a research runtime, not a production multi-tenant PD engine.
