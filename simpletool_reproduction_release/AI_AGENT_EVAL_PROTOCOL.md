# SimpleTool evaluation protocol for AI coding agents

Read this document before changing, porting, or rewriting the evaluator. The output contract follows the model architecture and the recovered author-local benchmark conversion. The architectural rules and the historical scorer's permissive behavior are separate issues: both must be documented rather than conflated.

## 1. One prediction represents one function call

SimpleTool decodes positional heads independently: `content`, `function`, and `arg1` through `arg6`. A benchmark row represents one current function call, not an entire sequence of calls. The accuracy runner requests all eight heads; the latency runner requests the function head and as many argument heads as the largest candidate schema needs, omitting content.

For each supported row:

1. Convert the raw heads to a call object containing `name` and `arguments`.
2. Require an exact match for the current target function name.
3. Compare the parameters of that current call under the declared local matching rules.

Do not silently interpret a grouped reference sequence as a requirement to emit every call at once. Conversely, do not claim success on a complete multi-call task from a correct single converted row. Evaluating an unsupported simultaneous multi-call protocol would measure a different capability.

## 2. Mobile Actions uses supplied reference history

The local conversion creates sequential rows. For example, `mobile_actions_5_0` has empty history and targets the first call; `mobile_actions_5_1` supplies the preceding call in history and targets the final call.

The recovered snapshot's target selection is:

- Empty history: use the first reference call.
- Nonempty history: use the final reference call.
- Require exactly one predicted call for the strict per-row diagnostic.

This is a contract of this converted snapshot, not a universal way to interpret any multi-call dataset. Earlier calls are supplied as reference context; they are not the model's own preceding predictions. Therefore, this metric is history-conditioned next-call accuracy, not autonomous trajectory success.

The historical scorer takes `pred_result[0]`. A grouped GT list alone does not prove that the model omitted required calls, because the unit being evaluated is the converted sequential row. For the completed non-quantized v1 run, Mobile Actions is 84.72% under the historical rules and 82.54% under strict history-aware scoring. The earlier 66.25% whole-sequence diagnostic was inappropriate for this protocol and must not be reused.

## 3. Parameter order is part of the architecture

Argument heads do not predict parameter names. Names are reconstructed using the selected tool's original `parameters.properties` insertion order:

- `arg1` maps to the first property, `arg2` to the second, and so on.
- Preserve both tool order and property order when loading or serializing schemas.
- Do not alphabetize properties or replace their order with the `required` array.

The original data snapshot preserves this order; `data/manifest.json` records its hashes.

If the selected tool has only two properties, only `arg1` and `arg2` have parameter semantics. **Ignore `arg3` through `arg6`, regardless of their contents.** They are unused output slots, not extra arguments. For example, correct predictions for `lookup(city, date)` remain correct even if an unused `arg3` head contains arbitrary text.

Do not confuse this rule with ignoring a parameter that actually exists in the selected schema. Such a parameter is decoded normally; how it is scored depends on the local GT and the chosen scorer.

## 4. Historical and strict local matching

The strict diagnostic rejects decoded parameter names absent from the GT. GT acceptable-value lists can mark omission as allowed using an empty string; otherwise values must pass the recovered local comparison rules, including their normalization and numerical tolerance. Function names must match exactly.

The historical scorer checks GT parameters without rejecting additional decoded names. That permissive behavior can raise scores and is retained only for historical comparison. A schema-valid optional parameter may be absent from GT, so the strict policy is a conservative local diagnostic, not an official semantic verifier.

Report which scorer produced a number. Neither scorer is the official BFCL AST/execution checker.

## 5. Capacity filtering changes the population

The model exposes six argument heads. The recovered rule excludes an entire row if **any candidate tool** has more than six properties, even if the predicted tool would need fewer.

Always publish input count, excluded IDs, timed count, and scored count separately. The full accuracy run has 5,647 inputs and 5,437 scored rows. The six-subset BFCL latency run has 2,061 inputs, 181 capacity exclusions, and 1,880 timed rows. Two timed rows have unusable GT, leaving 1,878 accuracy-scored rows in that run.

A failed, incorrect, or length-capped prediction must not silently disappear from timing or from its applicable accuracy denominator.

## 6. Stop markers and historical run differences

The author-confirmed stopping contract is `<|null|>` **or any defined head closing tag**, such as `</function>` or `</arg1>`. `</*>` is shorthand for actual head closers, not a literal wildcard stop string. EOS/ChatML end can also stop generation.

The final BFCL latency runner explicitly uses `skip_special_tokens=False`, because the checkpoint registers null and the closers as special tokens. It records each head's finish reason and stop reason. A null-stopped empty argument is omitted during conversion.

The previously published batched accuracy run and three-demo latency run retain their historical settings. Do not rewrite those raw outputs or retroactively claim they used the final BFCL latency stopping configuration. The initial BFCL latency attempt without null stopping was interrupted and is not the final-protocol result. Consult each immutable `run.json` rather than assuming that all experiments used identical settings.

## 7. BFCL scope and execution claims

The release covers the listed author-local BFCL simple/multiple subsets, not all categories of the official BFCL-v3 suite. Here, `multiple` indicates multiple candidate tools and must not automatically be interpreted as multiple simultaneous output calls.

The `exec_*` accuracy reports use local value matching; they do not execute reference programs. Use “BFCL Exec (local matcher)” rather than presenting the result as an official leaderboard score. An official comparison requires a pinned upstream dataset and checker revision, an explicit output adapter, and documented unsupported cases.

## 8. Reproduce and validate

Start with [REPRODUCIBILITY_GUIDE.md](REPRODUCIBILITY_GUIDE.md) for the branch checkout, data inventory, commands, environment variants, and offline verification. The [BFCL latency protocol](BFCL_SPEED_PROTOCOL.md) defines the single-question timing contract. Run unit tests and preflight before publishing changes.

Preserve raw outputs and their run manifests. Re-score copies into a new directory. If prompts, weights, data, or stopping behavior change, generate a new run and label it separately; do not combine outputs under an old run identity.

## 9. Interpreting discrepancies

For a lower BFCL Live result, compare checkpoint revision, v1/v2 prompt format, exact data snapshot, property order, stop handling, capacity exclusions, and scorer semantics first.

The completed non-quantized accuracy run uses v1 `RT-Qwen3-4B`. Its strict local Live result is 76.17%, versus the paper's 76.4%. This is evidence of close agreement under the recovered local setup; it does not establish absence of contamination or official leaderboard equivalence.

The v2 checkpoint is a later variant with additional game and role-play data. This release does not contain a complete v2 BFCL accuracy evaluation, so it cannot quantify or establish the cause of another user's v2 discrepancy. Likewise, the quantized v1 latency run has its own correctness measurements and must not inherit the non-quantized run's accuracy figures.
