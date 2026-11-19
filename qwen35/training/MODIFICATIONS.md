# Packaging modifications

- Original training sources were fetched read-only from their retained cloud directories.
- Config MODEL_DIR/DATA_DIR/OUTPUT_DIR now read SFT_MODEL_DIR/SFT_DATA_DIR/SFT_OUTPUT_DIR.
- Small-model prepare_scale.py takes --source-root, --tokenizer and --output rather than private paths.
- 9B prepare_8192.py reads SFT_SOURCE_PACKS_DIR.
- INDEX.json source paths are reduced to source basenames; selection, split, pack and hash fields are unchanged.
- Training arithmetic, LoRA targets, losses, segment isolation and checkpoint tensors are not rewritten.

This source packaging has CPU syntax/manifest checks. No new training or weight conversion was performed for the release.
