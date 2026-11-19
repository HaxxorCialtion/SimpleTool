# Frozen quality evaluation

- `diagnostic800/`: the public bilingual synthetic diagnostic, 800 rows / 400 pairs / four 200-row batches. Includes inputs, literal labels, tags, scorer and data card. Public dataset: https://huggingface.co/datasets/Cialtion/SimpleTool-Diagnostic-800 ; mirror: https://modelscope.cn/datasets/cialtion/SimpleTool-Diagnostic-800 . Dataset CC-BY-4.0; scorer CODE_LICENSE is separate.
- `public4786/`: frozen local matcher across BFCL/Seal-Tools/OpenFunctions/ToolAlpaca/Mobile Actions. Read `README_最终评测标准与运行.md`. The historical `upstream/` protocol is preserved for provenance; later root protocol governs this frozen suite.

Single-call tasks only; Mobile Actions has 640 single-call rows. Required arguments precede optional properties, required order agrees, function + candidate maximum arity heads are decoded, content excluded. After selecting a function ignore tail heads beyond its arity. Primary accuracy = function plus all required arguments under the frozen local acceptable-value matcher. Report function selection, types, full arguments and closing markers separately. These scores are not official BFCL AST/execution scores.

Never feed gold labels into prompts, silently change exclusions/denominators, repair a prediction with another model, or treat saved history as autonomous rollout success. Public4786's original infer_tools_parallel.py fixes FP16/TP1; do not use it unchanged for 397B FP8. Adapt transport while keeping prompts/scorer fixed and record the runtime change.

CPU verification from public4786: `python check_frozen.py` and `python -m unittest test_adapter -v`. From upstream: `python -m unittest discover -s tests -v`. Diagnostic scorers and schema tests live under each frozen batch's scripts. The top-level check_release.py orchestrates the available checks.
