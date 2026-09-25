# SimpleTool accuracy and latency reproduction

**Start here:** [Detailed reproduction and artifact guide](REPRODUCIBILITY_GUIDE.md). Branch: `reproduction/bfcl-v3-4090`. The branch includes code, original benchmark fixtures, and all published raw accuracy and latency measurements.

Recovered author-local benchmarks, prompts, conversion, scorers, and complete raw outputs for **RT-Qwen3-4B (v1, non-quantized)**. Model: `Cialtion/SimpleTool`, revision `aca7673dd2d90a919159613261ea374332e9d32f`. This is not an evaluation of RT-Qwen3-4B-v2.

**Agents: read [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) and [AI_AGENT_EVAL_PROTOCOL.md](AI_AGENT_EVAL_PROTOCOL.md) first.** SimpleTool predicts one call per row using schema-ordered argument heads. Unused heads are ignored. Mobile Actions uses supplied prior-call history, not whole-sequence generation.

## Completed run

One H100 80 GB, vLLM 0.11.0, eager execution, cascade attention disabled, batch size 64, max concurrent sequences 128. All 5,647 inputs across 18 datasets have saved outputs; 5,437 are scored after the recorded capacity filter and two missing/invalid GT cases. See `results/RT-Qwen3-4B-eager/run.json` for exact configuration and hashes.

| Category | Historical local matcher | Strict local matcher | Paper ST-Qwen3-4B |
| --- | ---: | ---: | ---: |
| BFCL Non-Live | 92.67% | 92.67% | 92.5% |
| BFCL Live | 76.17% | 76.17% | 76.4% |
| BFCL Exec | 89.93% | 87.25% | 89.9% |
| Mobile Actions | 84.72% | 82.54% | 84.5% |
| Others | 86.60% | 86.03% | 86.6% |

Each category pools correct/scored counts (micro averaging). These are author-local matches, **not official BFCL AST/execution leaderboard scores**. Strict Mobile Actions remains a one-call, history-conditioned evaluation. Neither column measures autonomous trajectory success.

## Reproduce

Run from this directory. Install `requirements.txt` in a compatible CUDA environment or use the author's existing vLLM environment. Download the pinned model and verify it:

```bash
python scripts/download_model.py --output models
python scripts/verify_model.py --model models/RT-Qwen3-4B
python -m simpletool_eval.run --preflight-only
python -m unittest discover -s tests -v
RESULT_DIR=results/new-run REPORT_DIR=reports/new-run \
  bash scripts/run_accuracy.sh --enforce-eager --disable-cascade-attn --batch-size 64 --max-num-seqs 128
python -m simpletool_eval.strict --predictions results/new-run --data data --output reports/new-run/strict
```

`PYTHON_BIN`, `MODEL_DIR`, and `DATA_DIR` may override local paths. Use a new output directory: do not resume into the published run after changing code or paths. The runner rejects incompatible resume fingerprints. Defaults for context/output are 32,768/128 tokens, temperature 0, seed 0; overlong inputs fail rather than truncate. GPU kernels and software changes may cause small decoding differences; bitwise identical output is not guaranteed.

To re-score the included raw outputs without a GPU:

```bash
python -m simpletool_eval.evaluate --predictions results/RT-Qwen3-4B-eager --output reports/recheck-legacy
python -m simpletool_eval.strict --predictions results/RT-Qwen3-4B-eager --data data --output reports/recheck-strict
```

The legacy evaluator also emits historical five-group and three-group aggregates. Those are different from the five paper categories above; do not mix their labels or averaging conventions.

## Artifacts and limitations

- `data/`: original local snapshots and possible answers; `manifest.json` pins hashes and excluded IDs. See [DATA_SOURCES.md](DATA_SOURCES.md).
- `simpletool_eval/`: recovered inference protocol, positional decoder, historical scorer, and explicit strict diagnostic.
- `results/RT-Qwen3-4B-eager/`: all eight raw heads per input, stop reasons and token counts. `run.json` is the original immutable execution receipt; later scorer changes do not rewrite it.
- `reports/RT-Qwen3-4B-eager/`: saved reports; `strict/` contains the strict per-row diagnostics.
- `reports/legacy_parity.json`: regression against 5,647 original local predictions; this checks scorer parity, separately from the new GPU run.

The initial compiled vLLM attempt failed with CUDA illegal memory access; the reported run is the successful full eager rerun. RTX 4090 W4A16-v2 latency has now been measured separately: see [speed protocol](SPEED_BENCHMARK.md) and [results](reports/speed-4090/summary.md). Full v2 accuracy, a matched speedup baseline, and official BFCL checker evaluation remain unmeasured here. See [AUDIT_FINDINGS.md](AUDIT_FINDINGS.md) for scoring limitations and [ISSUE_RESPONSE_DRAFT.md](ISSUE_RESPONSE_DRAFT.md) for a suggested response.

## BFCL-v3 serial latency

The author clarifies that the paper’s 16 Hz came from BFCL-v3. The three demo scenarios above are supplementary. See [BFCL_SPEED_PROTOCOL.md](BFCL_SPEED_PROTOCOL.md) for the new direct BFCL-v3 single-turn runner: one question in flight, with that question’s heads decoded together. It retains incorrect/capped calls in latency aggregates and separately reports ordinary traversal, cache-reset and repeated-input conditions.

Completed BFCL-v3 v1 W4A16 results: **13.40 Hz sequential**, **10.03 Hz with per-question cache reset**, **15.25 Hz after priming each identical question**. See [full report](reports/bfcl-speed-4090/v1/summary.md). The warm-identical rate excludes priming and is not a first-request throughput claim. The ordinary traversal did not exactly reproduce the paper’s 16 Hz.
