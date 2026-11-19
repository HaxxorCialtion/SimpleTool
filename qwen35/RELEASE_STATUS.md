# Release status — 2026-11-19

| Material | Status |
|---|---|
| Seven sizes / eleven 5% weight variants | Public HF + ModelScope; live file inventory and size metadata verified |
| Diagnostic-800 | Public both hubs; frozen data/scorers included |
| Public4786 local-matcher suite | Frozen inputs/scorers/protocol included, preserving source notices |
| Qwen hybrid-state CUDA implementation | Source snapshot, exact terminal state, TP worker, graphs, API and CPU tests included |
| Apple INT4/A16/KV8 source and Metal operators | Current Swift snapshot + dependency sources; historical operator results included |
| Training / preprocessing / merge source | Actual three cloud lineages and selection manifests included |
| Original training corpus | Not publicly bundled yet; exact reconstruction requires retained sources |
| Clean-machine GPU/iPhone acceptance of this package | Not rerun in this publication turn |
| Universal bit-exactness / same-base JSON AR comparison | Not claimed |

The eleven Qwen formats are not every historical checkpoint. The separate [Gemma4 adapter release](../gemma4/README.md) has its own artifact contract and quality status. BF16 / INT4 / FP8 scores must retain their measured compute dtype and prompt protocol. A16 and KV8 are independent choices in the Apple implementation; cloud AWQ/FP8 runs use a different backend.

Performance tables are retained historical measurements, not newly generated during source integration. Content is excluded from tool latency. Fixed-prefix task fixtures are not executed robot/game/digital-human rollouts. Data scaling gains flatten on these diagnostics for some sizes; this does not prove a general data-saturation law.
