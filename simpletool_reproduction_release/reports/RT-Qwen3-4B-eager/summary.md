# SimpleTool accuracy reproduction

SimpleTool local legacy rules (not official BFCL execution/AST scoring)

Full coverage

| Dataset | Input | Scored | Filtered | Missing GT | Function | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BFCL_v3_live_simple | 258 | 244 | 14 | 0 | 100.00% | 78.28% |
| BFCL_v3_simple | 400 | 400 | 0 | 0 | 99.75% | 93.50% |
| BFCL_v3_live_multiple | 1053 | 885 | 167 | 1 | 95.25% | 75.59% |
| BFCL_v3_multiple | 200 | 200 | 0 | 0 | 99.00% | 91.00% |
| BFCL_v3_exec_multiple_dataset | 50 | 49 | 0 | 1 | 100.00% | 87.76% |
| BFCL_v3_exec_simple_dataset | 100 | 100 | 0 | 0 | 100.00% | 91.00% |
| openfunctions_v1_test | 109 | 109 | 0 | 0 | 100.00% | 55.05% |
| sealtools_test_1tools | 293 | 292 | 1 | 0 | 100.00% | 90.07% |
| sealtools_test_2tools | 293 | 291 | 2 | 0 | 100.00% | 90.38% |
| sealtools_test_3tools | 293 | 288 | 5 | 0 | 100.00% | 89.93% |
| sealtools_test_4tools | 293 | 287 | 6 | 0 | 100.00% | 90.59% |
| sealtools_test_5tools | 293 | 289 | 4 | 0 | 100.00% | 90.31% |
| sealtools_test_6tools | 293 | 288 | 5 | 0 | 100.00% | 89.93% |
| sealtools_test_in_domain | 199 | 195 | 4 | 0 | 100.00% | 89.23% |
| sealtools_test_out_domain | 94 | 94 | 0 | 0 | 100.00% | 93.62% |
| toolalpaca_test_real_converted | 92 | 92 | 0 | 0 | 93.48% | 61.96% |
| toolalpaca_test_simulated_converted | 51 | 51 | 0 | 0 | 98.04% | 52.94% |
| mobile_actions | 1283 | 1283 | 0 | 0 | 99.92% | 84.72% |

## five_groups

- BFCL: 86.19%
- SealTools: 91.02%
- MobileAct: 84.72%
- OpenFunc: 55.05%
- ToolAlpaca: 57.45%

Overall macro average (including zero groups): 74.88%
Original script nonzero-only average: 74.88%

## three_groups

- BFCL: 86.19%
- MobileAct: 84.72%
- Other: 86.60%

Overall macro average (including zero groups): 85.84%
Original script nonzero-only average: 85.84%
