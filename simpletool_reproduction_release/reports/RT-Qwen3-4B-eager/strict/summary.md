# Strict SimpleTool accuracy audit

This is the primary conservative local score. It requires exact call count and rejects extra predicted argument names. It does not execute official BFCL programs.

| Category | Scored | Function | Overall |
| --- | ---: | ---: | ---: |
| BFCL Non-Live | 600 | 99.50% | 92.67% |
| BFCL Live | 1129 | 96.28% | 76.17% |
| BFCL Exec | 149 | 100.00% | 87.25% |
| Mobile Actions | 1283 | 96.34% | 82.54% |
| Others | 2276 | 99.69% | 86.03% |

## Per dataset

| Dataset | Scored | Correct | Function | Overall |
| --- | ---: | ---: | ---: | ---: |
| BFCL_v3_live_simple | 244 | 191 | 100.00% | 78.28% |
| BFCL_v3_simple | 400 | 374 | 99.75% | 93.50% |
| BFCL_v3_live_multiple | 885 | 669 | 95.25% | 75.59% |
| BFCL_v3_multiple | 200 | 182 | 99.00% | 91.00% |
| BFCL_v3_exec_multiple_dataset | 49 | 41 | 100.00% | 83.67% |
| BFCL_v3_exec_simple_dataset | 100 | 89 | 100.00% | 89.00% |
| openfunctions_v1_test | 109 | 60 | 100.00% | 55.05% |
| sealtools_test_1tools | 292 | 261 | 100.00% | 89.38% |
| sealtools_test_2tools | 291 | 261 | 100.00% | 89.69% |
| sealtools_test_3tools | 288 | 257 | 100.00% | 89.24% |
| sealtools_test_4tools | 287 | 258 | 100.00% | 89.90% |
| sealtools_test_5tools | 289 | 259 | 100.00% | 89.62% |
| sealtools_test_6tools | 288 | 258 | 100.00% | 89.58% |
| sealtools_test_in_domain | 195 | 173 | 100.00% | 88.72% |
| sealtools_test_out_domain | 94 | 87 | 100.00% | 92.55% |
| toolalpaca_test_real_converted | 92 | 57 | 93.48% | 61.96% |
| toolalpaca_test_simulated_converted | 51 | 27 | 98.04% | 52.94% |
| mobile_actions | 1283 | 1059 | 96.34% | 82.54% |
