# Licenses and third-party provenance

New project code is Apache-2.0, matching the root LICENSE. Model/base licenses are provided in each Hub artifact. Weight format or fine-tuning does not replace a base license.

- Apple MLX Swift / MLX / MLX-C: MIT; source and licenses are in inference/apple/Vendor/mlx-swift including nested dependencies. MLX-LM-derived arithmetic and QMV attribution are retained in inference/apple/Licenses and operator_lab/third_party.
- vLLM / associated FLA source overlays retain their upstream copyright/SPDX headers and Apache-2.0 terms. NVIDIA FlashInfer kernels are dependencies, not silently relicensed project code. See corresponding packages' notices when installing them.
- Diagnostic-800 data: CC-BY-4.0, with separate CODE_LICENSE for its scripts.
- Public4786 combines several upstream datasets. Preserve evaluation/public4786/upstream/data/source_notices and upstream DATA_SOURCES.md. Do not assume a single root code license covers the mixed datasets.
- Original training data is not distributed here. Its source/rights documentation still needs to accompany a later corpus release.

This first source release packages previously measured research implementations. It does not claim ownership of upstream kernels or rename measurements from other precision/device configurations.
