Thanks for reporting this. We have released the recovered local evaluation scripts, exact data snapshot, prompts, model revision, and complete per-example outputs in `simpletool_reproduction_release/`. Please start with `EVALUATION_PROTOCOL.md`.

The paper comparison uses the v1 protocol. Our new full run evaluates non-quantized `RT-Qwen3-4B` at revision `aca7673dd2d90a919159613261ea374332e9d32f`, not v2. RT-Qwen3-4B-v2 is a later variant trained with additional game/role-play data; it should be evaluated separately, and its accuracy difference has not been quantified by this run.

SimpleTool generates independent `content`, `function`, and `arg1`–`arg6` heads. Argument names are recovered in the selected tool's original schema property order; heads beyond that tool's arity are ignored. One prediction represents one call. Mobile Actions rows use reference prior-call history and evaluate the current call, not an entire sequence at once.

Our historical local scores are Non-Live 92.67%, Live 76.17%, Exec 89.93%, Mobile Actions 84.72%, Others 86.60%. A stricter local diagnostic rejecting extra decoded parameter names gives 92.67%, 76.17%, 87.25%, 82.54%, and 86.03%, respectively. Scores pool samples within each category. The paper's Live figure is 76.4%.

These are local function/argument matches, not official BFCL AST/execution scores. The protocol excludes rows where any candidate schema exceeds six parameters, and documents two unusable GT cases. Full per-row outputs and denominators are included so these choices can be audited. Please compare the checkpoint, prompt version, schema order, data snapshot and excluded IDs when diagnosing the discrepancy; this v1 run alone cannot establish the cause of your v2 result.
