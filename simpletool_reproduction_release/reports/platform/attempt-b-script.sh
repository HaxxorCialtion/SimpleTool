#!/usr/bin/env bash
set -euo pipefail
cd /inspire/hdd/global_user/shixiaoxin-253107030017/simpletool-benchmark
exec > >(tee -a reports/platform/gpu_job-b.log) 2>&1
export PYTHON_BIN=/inspire/hdd/global_user/shixiaoxin-253107030017/sxx/vllm_env/bin/python
export MODEL_DIR=/inspire/hdd/project/high-dimensionaldata/shixiaoxin-253107030017/simpletool-reproduction/models/RT-Qwen3-4B
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=8
nvidia-smi
ls -ld "$MODEL_DIR" "$PYTHON_BIN" data
"$PYTHON_BIN" -c 'import torch, vllm; print(torch.__version__, vllm.__version__); assert torch.cuda.is_available()'
"$PYTHON_BIN" -m simpletool_eval.run --model "$MODEL_DIR" --datasets BFCL_v3_simple --limit 8 --output results/smoke-hf --resume
"$PYTHON_BIN" -m simpletool_eval.evaluate --predictions results/smoke-hf --datasets BFCL_v3_simple --allow-partial --output reports/smoke-hf
bash scripts/run_accuracy.sh
