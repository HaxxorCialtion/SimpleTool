# Instructions for coding agents working in this directory

Read `EVALUATION_PROTOCOL.md` and `AI_AGENT_EVAL_PROTOCOL.md` before changing prompts, conversion, scoring, or reports.

- Keep raw inference files and their historical `run.json` immutable.
- Preserve schema property order; ignore argument heads beyond the selected schema arity.
- Evaluate one call per converted sequential row. Supplied history contains reference preceding actions; this is not autonomous trajectory success.
- Never silently change denominators, drop failed predictions, or describe local matching as official BFCL AST/execution evaluation.
- Keep legacy and strict reports separate. Re-score saved outputs after scorer changes and record any score changes.
- Run `python -m unittest discover -s tests -v` and the preflight before publishing changes.
