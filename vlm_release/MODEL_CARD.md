# SimpleTool-VLM-4B model card

## Summary

`corrected_mixed_frozen_r4` is a Qwen3-VL-4B multimodal checkpoint fine-tuned for SimpleTool's parallel structured output protocol. The model receives an image plus a policy/task prompt and a function-tool schema. It produces independent textual branches for `function` and the arguments required by the schema.

## Input contract

A request contains a PNG image, system/user messages, and JSON-schema tools. The image is the source of invoice values in the demo; the client does not insert the printed numbers into the prompt. The model may still fail or return an illegal schema, so callers must inspect `legal` and retain `raw` outputs.

## Modes

- **Direct**: emit function and active argument heads.
- **Adaptive**: emit a bounded short `content` field, then the active heads. Empty content is valid and common.

Each head terminates on its own protocol stop marker. The public runtime dynamically activates only the argument heads needed by the largest tool schema.

## Training and evaluation boundary

The checkpoint was trained from a Qwen3-VL-4B base with mixed text/image structured-call data. The internal deployment smoke test passed the prepared invoice contract on the released serving path. Those probes do not establish general OCR accuracy, accounting correctness, or performance on arbitrary images.

Strict adapter-to-merged numerical parity was not established. The merged checkpoint was independently loaded and probed. Do not describe this release as a drop-in OCR replacement or as a reasoning trace generator.

## Quantization

BF16 is the reference release. W4A16 and FP8 files are separate exports with their own backend requirements and should be validated with the provided protocol before use.

## Safety

Tool calls should be validated against an allow-list and business rules before execution. Do not use the invoice example to approve payments automatically.
