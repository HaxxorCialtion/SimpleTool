import json
import statistics
import time
import argparse
from pathlib import Path

import mlx.core as mx

from simpletool.kernels import int4_matmul


def timed(fn, repetitions=20):
    for _ in range(3):
        mx.eval(fn())
    start = time.perf_counter()
    for _ in range(repetitions):
        mx.eval(fn())
    return (time.perf_counter() - start) / repetitions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--compute',choices=['float16','float32'],default='float32')
    args=p.parse_args()
    dtype={'float16':mx.float16,'float32':mx.float32}[args.compute]
    mx.random.seed(13)
    results = []
    for n, k in [(9216, 2560), (2560, 9216)]:
        w = mx.random.normal((n, k)).astype(mx.float16) * 0.02
        packed, scales, biases = mx.quantize(w, group_size=64, bits=4)
        scales=scales.astype(dtype);biases=biases.astype(dtype)
        mx.eval(packed, scales, biases)
        del w
        for m in [1, 4, 8]:
            x = mx.random.normal((m, k)).astype(dtype)
            mx.eval(x)
            custom = lambda: int4_matmul(x, packed, scales, biases)
            default = lambda: mx.quantized_matmul(x, packed, scales, biases,
                               transpose=True, group_size=64, bits=4)
            samples = {"custom": [], "mlx": []}
            for round_idx in range(3):
                order = [("custom", custom), ("mlx", default)]
                if round_idx % 2:
                    order.reverse()
                for name, fn in order:
                    samples[name].append(timed(fn))
            c, d = statistics.median(samples["custom"]), statistics.median(samples["mlx"])
            error = float(mx.max(mx.abs(custom().astype(mx.float32) - default().astype(mx.float32))).item())
            row = {"N": n, "K": k, "M": m, "custom_ms": c*1000,
                   "mlx_ms": d*1000, "speedup": d/c,
                   "custom_effective_weight_GB_s": (packed.nbytes+scales.nbytes+biases.nbytes)/c/1e9,
                   "max_absolute_difference_vs_mlx": error,
                   "samples_seconds": samples}
            results.append(row)
            print(json.dumps(row), flush=True)
    Path(f"reports/kernel_bench_{args.compute}.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
