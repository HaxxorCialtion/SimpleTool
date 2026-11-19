"""Actual packed int4 all-layer projection fixtures, without dense weight expansion."""
import argparse
import hashlib
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from simpletool.row_reuse_int4 import row_reuse_int4

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', choices=('4b', '2b'), required=True)
    a = p.parse_args()
    out = ROOT / f'artifacts/weighted_qmv_{a.size}_20261008'
    out.mkdir(exist_ok=True)
    checkpoint = ROOT / f'artifacts/{a.size}_int4_compact/model.safetensors'
    with checkpoint.open('rb') as f:
        header_length = int.from_bytes(f.read(8), 'little')
        header = json.loads(f.read(header_length))
    checkpoint_sha = sha(checkpoint)
    weights = mx.load(str(checkpoint))
    mx.set_cache_limit(256 * 1024**2)
    mx.random.seed(837012)
    down = sorted(k[:-7] for k in weights if '.mlp.down_proj.weight' in k)
    paths = {'down': down, 'gate': [k.replace('down_proj', 'gate_proj') for k in down],
             'vocabulary': ['language_model.model.embed_tokens']}
    families = []
    files = {}
    for family, names in paths.items():
        n, words = weights[names[0] + '.weight'].shape
        k = words * 8
        mats = []
        for name in names:
            w, s, b = [weights[name + suffix] for suffix in ('.weight', '.scales', '.biases')]
            assert w.dtype == mx.uint32 and s.dtype == b.dtype == mx.float16
            assert w.shape == (n, k // 8) and s.shape == b.shape == (n, k // 64)
            descriptors = {}
            for label, suffix, dtype in [('w', '.weight', 'U32'), ('s', '.scales', 'F16'), ('b', '.biases', 'F16')]:
                entry = header[name + suffix]
                assert entry['dtype'] == dtype
                start, end = entry['data_offsets']
                descriptors[label] = dict(offset=8 + header_length + start, bytes=end-start)
            mats.append(dict(path=name, buffers=descriptors))
        batches = []
        for batch in range(1, 5):
            x = mx.random.normal((batch, k)).astype(mx.float32)
            mx.eval(x)
            x_path = out / f'{family}_x{batch}.bin'
            np.asarray(x).tofile(x_path)
            expected_path = out / f'{family}_expected{batch}.bin'
            with expected_path.open('wb') as f:
                for name in names:
                    arrays = [weights[name + suffix] for suffix in ('.weight', '.scales', '.biases')]
                    rows = 4 if n >= 65536 else 8
                    threads = 256 if k <= 2560 or n >= 65536 else 128
                    y = row_reuse_int4(x, *arrays, rows, threads)
                    mx.eval(y)
                    yy = np.asarray(y)
                    assert np.isfinite(yy).all() and yy.dtype == np.float32
                    f.write(yy.tobytes())
            files[x_path.name] = sha(x_path)
            files[expected_path.name] = sha(expected_path)
            batches.append(dict(batch=batch, x=x_path.name, expected=expected_path.name))
        families.append(dict(name=family, n=n, k=k, matrices=mats, batches=batches))
        print(a.size, family, len(names), n, k, 'FP32 gold saved', flush=True)
    assert sha(checkpoint) == checkpoint_sha
    result = dict(size=a.size, checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                  weight_bits=4, group_size=64, metadata='float16 storage', compute='float32',
                  families=families, fixture_sha256=files, seed=837012,
                  scope='Actual all gate/down layer weights and tied vocabulary; standalone projections, not full model. B1 gold is row-reuse, not single-path MLX fallback.')
    (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
