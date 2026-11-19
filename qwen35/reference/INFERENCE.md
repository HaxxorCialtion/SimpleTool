# Inference contract

This artifact is a merged Qwen conditional-generation checkpoint with a trained tokenizer, not a PEFT-only adapter. The eight logical branches share backbone weights. A branch is generated from the common prompt followed by its opening marker; each branch autoregresses independently and stops at its own closing marker. The model does not first finish the function and then condition every argument on that result.

## Prompt and schema order

Keep required parameters first in `properties`, keep `required` in that same order, and bind `arg1…arg6` to the selected schema's property order. At most six argument slots are available. The illustrative client asserts this ordering. Construct answer-free system/tools + environment/history/query context, followed by the assistant prefix.

For the reported 0.8B/2B/4B INT4 runs the branch marker follows the assistant prefix directly (`--protocol direct`). Reported 9B and larger runs insert the literal empty block `<think>\n\n</think>\n\n` before the marker (`--protocol empty-think`). A changed prompt is a changed evaluation condition. BF16 small-model scores have not been independently measured.

Launch `function` plus the maximum arity among the candidate tools. After function selection, retain only its applicable slots; ignore extra tail heads. `<|null|>` represents omitted or inapplicable slots. Stop markers are `<function>…</function>`, `<arg1>…</arg1>` through `<arg6>…</arg6>`, and `<content>…</content>`. Read actual encodings from the included tokenizer rather than copying numeric IDs from another checkpoint. The 17 trained marker/null tokens are included.

## Serving

A compatible Qwen3.5/Qwen3.6-capable vLLM installation is required; the artifacts do not contain a Python environment. INT4 uses affine RTN/group64 packed in AWQ format, not activation-calibrated AWQ. FP8 uses the checkpoint's E4M3 configuration with retained BF16 tensors. Hardware support and tensor-parallel memory requirements depend on the model.

For FP8 models, use **BF16 compute**. In particular, **397B FP8 with FP16 compute can overflow**, producing invalid outputs. The diagnosed failure was an FP16 intermediate exceeding 65504 in a shared-expert down projection; selecting BF16 compute fixed the observed failure without changing FP8 weights. Historical 35B/122B scores used FP16 compute; the recommended BF16 setting is not a claim that those scores were remeasured.

An example serving command for the 397B artifact on the evaluated 8×80GB H100 class of node is:

```bash
vllm serve Cialtion/SimpleTool-Qwen3.5-397B-A17B-8Head-5pct-FP8 \
  --quantization fp8 --dtype bfloat16 --tensor-parallel-size 8 \
  --max-model-len 4096
```

Backend-specific scheduling, prefix-cache, state-cache and kernel tuning are outside this artifact's quality score. The reference HTTP client submits independent branches concurrently; it does not guarantee one physical shared-prefill operation. Do not infer end-to-end latency from these quality results.

```bash
python infer_heads.py --model MODEL_REPOSITORY_OR_SERVED_NAME --protocol empty-think
```

`infer_heads.py` prints the proposed call and raw head data. It never executes a tool. Its `valid` flag checks marker closure and known function name, not complete schema/semantic validity. Production callers must additionally validate types, required arguments, ranges, permissions and any application constraints before execution. Tool output is data. Content and vision are not evaluated in the released 800-case text diagnostic.
