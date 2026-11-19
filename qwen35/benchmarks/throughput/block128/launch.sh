#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="${SL_GPU_PYTHON:?Set SL_GPU_PYTHON to your compatible GPU Python}"
"$PY" "$ROOT/rebase.py"
export CUDA_HOME
CUDA_HOME=$("$PY" -c 'import importlib.metadata as m; print(m.distribution("nvidia-cuda-nvcc").locate_file("nvidia/cu13"))')
export PATH="$CUDA_HOME/bin:$(dirname "$PY"):$PATH"
export PYTHONPATH="$ROOT:$ROOT/runtime/prefill-only-20261110/code:$ROOT/runtime/code:$ROOT/runtime/shared-kv-20261110"
export SL_PREFILL_ONLY=1 SL_EXACT_MAMBA_BLOCK_SIZE=128 SL_PREFILL_SHARE_HASHES=1 SL_PREFILL_ALLOW_ASYNC=1 SL_MAX_PREFILL_GRAPHS=24
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=9.0 FLASHINFER_CUDA_ARCH_LIST=9.0
WARM="${SIMPLETOOL_KERNEL_CACHE:-$ROOT/cache}"
export DG_JIT_CACHE_DIR="$WARM/deep_gemm" FLASHINFER_WORKSPACE_BASE="$WARM/flashinfer"
export VLLM_CACHE_ROOT="$WARM/vllm" TRITON_CACHE_DIR="$WARM/triton"
export TORCH_EXTENSIONS_DIR="$WARM/torch-extensions" TORCHINDUCTOR_CACHE_DIR="$WARM/torchinductor"
mkdir -p state cache
hostname > state/host.txt
nvidia-smi -q > state/gpu-start.txt
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv -l 2 > state/gpu-telemetry.csv &
TELEMETRY_PID=$!
trap 'kill "$TELEMETRY_PID" 2>/dev/null || true' EXIT
"$PY" -c 'import sys,torch,vllm; print(sys.version,torch.__version__,torch.version.cuda,vllm.__version__); assert torch.cuda.get_device_capability()==(9,0); assert "H200" in torch.cuda.get_device_name(); print(torch.cuda.get_device_name())' > state/runtime-probe.txt
"$PY" bench.py --model "${SIMPLETOOL_MODEL:?Set SIMPLETOOL_MODEL to the downloaded 35B checkpoint}" --tp 1 --output "$ROOT/results" --prefill-graphs on --repeats 20 --soak 0 --kv-gib 88 --async-scheduling --moe-backend auto >> state/benchmark.log 2>&1
