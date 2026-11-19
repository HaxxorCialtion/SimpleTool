---
language:
- zh
- en
license: cc-by-4.0
pretty_name: SimpleTool Diagnostic 800
size_categories:
- n<1K
task_categories:
- text-generation
tags:
- tool-calling
- function-calling
- counterfactual
- synthetic
- evaluation
configs:
- config_name: default
  data_files:
  - split: cf_zh
    path: data/cf_zh.jsonl
  - split: cf_en
    path: data/cf_en.jsonl
  - split: hard
    path: data/hard.jsonl
  - split: multiarg
    path: data/multiarg.jsonl
---
# SimpleTool Diagnostic 800

A synthetic, bilingual, paired diagnostic benchmark for **single tool calls**. It contains 800 cases: 400 Chinese and 400 English; four separately reported batches of 200. This is a held-out quality diagnostic, not a replacement for BFCL/Mobile Actions or a production-user benchmark.

## Batches and targets

| Batch | Cases | Languages | Targets |
|---|---:|---|---|
| cf-zh | 200 | Chinese | Environment changes, pseudo-history, entity/parameter binding, semantics |
| cf-en | 200 | English | Independently authored English cases, not a translation of cf-zh |
| hard | 200 | 100 Chinese + 100 English | Multi-step constraints, transformations, boundaries, dependencies |
| multiarg | 200 | 100 Chinese + 100 English | 4–6 parameter calls, role binding, array alignment, arithmetic consistency |

Cases are organized as 400 related pairs. Small input edits may require a changed call; invariant controls require an unchanged correct call. Inputs are not independent IID samples: report paired metrics when comparing models. A `history` field is supplied context, not a live multi-turn rollout. Each case asks for one next call; no tools are executed.

Every case includes capability tags, a target/objective, literal expected call, and rationale. Tags overlap; their counts are not additive. These records were authored in November 2026 through agent-assisted synthetic construction and deterministic validation/review. They were not sampled from real users or independently blind-audited; residual label/design errors remain possible. Do not use the test answers for training and still describe resulting scores as held-out.

## Loading and answer isolation

```python
from datasets import load_dataset
benchmark = load_dataset("Cialtion/SimpleTool-Diagnostic-800")
row = benchmark["hard"][0]
import json
tools = json.loads(row["tools"])
expected = json.loads(row["expected"])  # scoring only; never send to model
```

Viewer-facing nested payloads (`tools`, `environment`, `history`, `expected`) are JSON strings to avoid heterogeneous Arrow-schema coercion. `frozen/<batch>/cases.jsonl` preserves the exact original nested records, while `inputs.jsonl` contains answer-free inputs. Never pass a full labeled row to the model. Do not include `expected`, rationale, objective, tags or pair metadata in the model prompt.

`frozen/<batch>/prompts.jsonl` preserves the originally evaluated prompts. Depending on the checkpoint protocol, append the explicit empty-think block `<think>\n\n</think>\n\n` once before the head marker, or use the historical direct-head prefix as recorded in the result. Do not silently change the frozen user/system text. Required properties come first, `required` order matches properties order, and arg1…argN bind to that property order.

For SimpleTool, generate function and the candidate schema's maximum arity of argument heads from the same prompt; do not generate content. Then select the predicted function schema and ignore unneeded trailing slots. Argument heads are not conditioned on the newly generated function head. End each head at its actual tokenizer's matching closing marker.

## Scoring

```bash
python frozen/cf-zh/scripts/score.py --cases frozen/cf-zh/cases.jsonl --predictions predictions.jsonl --out scores.json
```

A prediction can be `{"id":"…","name":"function_name","arguments":{…}}`, or `{"id":"…","heads":{"function":"function_name","arg1":"raw value"}}`. For head inputs, string bodies are raw strings; other parameter types require valid JSON. Keep each scorer with its own batch library.

The primary score is function-name correctness AND correctness of every required argument. Also report function-name accuracy, schema validity, tag/family/arity breakdown and paired joint accuracy. Matching is type-sensitive and literal: booleans are not numbers, arrays retain order/duplicates, object key order is irrelevant, malformed JSON and non-finite values fail. Extra unused heads are ignored. Missing cases count as failures. This is a project-local scorer, not an official BFCL score.

## Observed 5% SimpleTool baselines

All numbers below use these frozen 800 cases. FP16 in the compute column describes runtime, not necessarily checkpoint storage. INT4 is affine RTN/group64 serialized in AWQ layout, without activation calibration. Models differ in base family, prompt protocol and training details; these are observed runs, not a controlled scaling-law fit.

| Model | Weights / compute | cf-zh /200 | cf-en /200 | hard /200 | multiarg /200 | Full call /800 | Function /800 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.8B | INT4/FP16 | 112 | 113 | 52 | 24 | 301 | 733 |
| 2B | INT4/FP16 | 137 | 136 | 58 | 45 | 376 | 743 |
| 4B | INT4/FP16 | 169 | 167 | 91 | 93 | 520 | 775 |
| 9B | INT4/FP16 | 166 | 173 | 87 | 125 | 551 | 770 |
| 35B | FP8/FP16 | 188 | 192 | 117 | 146 | 643 | 786 |
| 122B | FP8/FP16 | 186 | 189 | 134 | 159 | 668 | 787 |
| 397B | FP8/BF16 | 190 | 195 | 144 | 153 | 682 | 792 |

35B is Qwen3.6-35B-A3B; other rows are Qwen3.5. 397B uses the completed 5074-step 5% checkpoint and FP8/BF16. Its 3286 heads all closed normally; original BF16-weights/BF16-compute reference scored 685/800 full calls and 791/800 functions. The failing FP8/FP16 run is excluded from quality baselines. Newly converted BF16 small-model checkpoints do not inherit the INT4 scores as verified BF16 scores.

## Files, provenance and license

Original cases/prompts retain their frozen SHA256 hashes in `source_data_hashes.json`. Root `manifest.json` inventories the actual release. `frozen/*/manifest.json` is historical and also lists authoring assets not included here. `results/` contains aggregate baseline counts; no internal paths, tokens or deployment endpoints are required.

Dataset: CC BY 4.0. Scoring software: Apache License 2.0 (see CODE_LICENSE). Attribute this dataset to the SimpleTool project / Cialtion and retain the distinction between synthetic diagnostic scores and externally validated task performance.
