#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_DIR="${MODEL_DIR:-models/RT-Qwen3-4B}"
DATA_DIR="${DATA_DIR:-data}"
RESULT_DIR="${RESULT_DIR:-results/RT-Qwen3-4B}"
REPORT_DIR="${REPORT_DIR:-reports/RT-Qwen3-4B}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
"$PYTHON_BIN" -m simpletool_eval.run --model "$MODEL_DIR" --data "$DATA_DIR" --output "$RESULT_DIR" --resume "$@"
"$PYTHON_BIN" -m simpletool_eval.evaluate --data "$DATA_DIR" --predictions "$RESULT_DIR" --output "$REPORT_DIR"
