#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SL_GPU_PYTHON="${SL_GPU_PYTHON:?Set SL_GPU_PYTHON to your compatible GPU Python}"
"$SL_GPU_PYTHON" "$ROOT/infra/rebase.py"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/infra:$ROOT/infra/runtime/prefill-only-20261110/code:$ROOT/infra/runtime/code"
export SL_PREFILL_ONLY=1 SL_EXACT_MAMBA_BLOCK_SIZE=16 SL_PREFILL_SHARE_HASHES=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST=9.0 FLASHINFER_CUDA_ARCH_LIST=9.0
export DG_JIT_CACHE_DIR="$ROOT/cache/deep_gemm" FLASHINFER_WORKSPACE_BASE="$ROOT/cache/flashinfer"
export VLLM_CACHE_ROOT="$ROOT/cache/vllm" TRITON_CACHE_DIR="$ROOT/cache/triton"
export SL_MAX_PREFILL_GRAPHS="$("$SL_GPU_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_prefill_graphs"])' "$ROOT/engine.json")"
# Fail before a 397B load if isolated scheduler-hook dependencies are missing.
"$SL_GPU_PYTHON" -c 'import fork_gate; assert all(callable(getattr(fork_gate,n)) for n in ("on_add","blocked","before_free","after_update"))'
mkdir -p "$ROOT/state"
exec "$SL_GPU_PYTHON" "$ROOT/server.py" --config "$ROOT/engine.json" >> "$ROOT/state/server.log" 2>&1
