# H200 branch-batch throughput experiments (2026-11-16)

`batch32/` preserves the measured B1–32 run; `block128/` preserves the final B1–1024 throughput experiment, including its own runtime snapshot and historical report/CSV. Local launchers have been rebased, with model/Python/cache paths provided by environment variables. Overlay hashes are checked before import; installed vLLM is untouched. Same version constraints as the CUDA service apply.

```bash
export SL_GPU_PYTHON=/absolute/path/to/compatible/python
export SIMPLETOOL_MODEL=/absolute/path/to/35B-FP8
bash qwen35/benchmarks/throughput/block128/launch.sh
```

A 35B FP8 model plus the configured 88 GiB cache requires an H200-class memory budget; do not copy this cache setting to an iPhone or small GPU. Outputs are created in results/ and an existing result directory is refused.

These are fixed-length, one-group-at-a-time branch-batch experiments. Beyond eight branches the trained markers repeat. They are not independent-user concurrency, a 1024-head trained model, or natural tool-call latency. Block128 avoids recursive-state checkpoint fragmentation observed in failed block16 experiments. Each branch retains independent mutable GDN state; full public-prefix hits and exact scheduled batch shape were asserted on accepted timings. The report includes rejected/failed configurations and a broad throughput plateau, not a claim of a universal optimum.

The latent-recurrence prototype is a separate untrained architecture experiment and is not part of this model release.
