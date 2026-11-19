#!/usr/bin/env python3
"""Bake a head-LoRA adapter (moe/runs/<key>_<TAG>/train/final) into a base checkpoint for vLLM.

Base may be bf16 (e.g. Qwen3.5-4B) or an official block-FP8 checkpoint (e.g. Qwen3.5-122B-A10B-FP8,
weight + weight_scale_inv, 128x128 blocks). The output has the base's layout (same shard names, index,
config, quantization_config) and loads in vLLM as the original architecture; no vLLM LoRA support needed.

  * bf16 target   : W <- W + (alpha/r) * B @ A                     (fp32 math, cast back)
  * FP8 target    : W <- requant(dequant(Wq, s) + (alpha/r) * B @ A) with the same block layout
                    (delta = 0 reproduces Wq bit for bit; see --selftest)
  * embed_tokens  : all 17 start/end/null rows replaced (tied input and output weights)
  * tokenizer     : replaced by the adapter's (head tokens registered)
Shards without a changed tensor are symlinked to the base (--materialize copies them instead, for a
self-contained directory you can move to another machine).

  python merge_model.py --base <dir> --adapter <.../train/final> --out <dir> [--dry-run] [--materialize]
  python merge_model.py --selftest
Runs in the SFT env (torch + safetensors); CPU only.
"""
import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

LORA_KEY = re.compile(r"^base_model\.model\.(.+)\.lora_([AB])\.weight$")
BLOCK = 128


# --------------------------------------------------------------------------- checkpoint helpers
def checkpoint_index(d):
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    from safetensors import safe_open
    with safe_open(str(d / "model.safetensors"), framework="pt") as h:
        return {k: "model.safetensors" for k in h.keys()}


def text_key(key):
    """Checkpoint key -> text-model key used by the adapter (same rule as moe/common.py)."""
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    if key.startswith(("model.visual.", "mtp.", "model.mtp.")):
        return None
    if key.startswith(("model.", "lm_head.")):
        return key
    return None


def scale_key(wkey):
    return wkey + "_scale_inv" if wkey.endswith(".weight") else None


def lora_pairs(adapter):
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    assert not cfg.get("use_rslora") and not cfg.get("use_dora"), "only plain LoRA is supported"
    from safetensors import safe_open
    pairs = defaultdict(dict)
    with safe_open(str(adapter / "adapter_model.safetensors"), framework="pt") as h:
        for key in h.keys():
            m = LORA_KEY.match(key)
            if not m:
                raise RuntimeError(f"unexpected adapter tensor {key}")
            pairs[m.group(1) + ".weight"][m.group(2)] = h.get_tensor(key)
    bad = [k for k, v in pairs.items() if set(v) != {"A", "B"}]
    assert not bad, bad[:5]
    return dict(pairs), cfg["lora_alpha"] / cfg["r"]


# --------------------------------------------------------------------------- block FP8
def fp8_dequant(q, s, block=BLOCK):
    import torch
    r, c = q.shape
    sf = s.float().repeat_interleave(block, 0)[:r].repeat_interleave(block, 1)[:, :c]
    return q.to(torch.float32) * sf


def fp8_quant(w, qdtype, sdtype, block=BLOCK):
    import torch
    import torch.nn.functional as F
    fmax = torch.finfo(qdtype).max
    r, c = w.shape
    pr, pc = (-r) % block, (-c) % block
    t = F.pad(w.float(), (0, pc, 0, pr)).view((r + pr) // block, block, (c + pc) // block, block)
    s = t.abs().amax(dim=(1, 3)).clamp_min(1e-12) / fmax
    q = (t / s[:, None, :, None]).clamp(-fmax, fmax).to(qdtype).view(r + pr, c + pc)[:r, :c]
    return q.contiguous(), s.to(sdtype)


def selftest():
    import torch
    torch.manual_seed(0)
    worst = 0.0
    for r, c in ((256, 384), (300, 200), (129, 1000)):
        w = torch.randn(r, c) * 0.02
        q, s = fp8_quant(w, torch.float8_e4m3fn, torch.float32)
        assert q.shape == (r, c) and s.shape == (-(-r // BLOCK), -(-c // BLOCK)), (q.shape, s.shape)
        q2, s2 = fp8_quant(fp8_dequant(q, s), torch.float8_e4m3fn, torch.float32)
        assert torch.equal(q2.view(torch.uint8), q.view(torch.uint8)), "requant changed FP8 values"
        assert torch.allclose(s2, s, rtol=1e-6, atol=0), "requant changed scales"
        delta = torch.randn(r, c) * 0.002
        q3, s3 = fp8_quant(fp8_dequant(q, s) + delta, torch.float8_e4m3fn, torch.float32)
        ref = fp8_dequant(q, s) + delta
        err = ((fp8_dequant(q3, s3) - ref).norm() / ref.norm()).item()
        worst = max(worst, err)
        assert err < 0.05, err
    print(f"selftest OK: idempotent requant, padded edges, merged rel. error {worst:.2e}")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base")
    ap.add_argument("--adapter")
    ap.add_argument("--out")
    ap.add_argument("--dry-run", action="store_true", help="print the plan (FP8/bf16 targets, shards, space) only")
    ap.add_argument("--materialize", action="store_true", help="copy unchanged shards instead of symlinking")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.base and a.adapter and a.out):
        ap.error("--base, --adapter and --out are required")
    base, adapter, out = Path(a.base), Path(a.adapter), Path(a.out)
    if (out / "MERGE_DONE").exists():
        print("already merged:", out)
        return
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    t0 = time.time()

    pairs, scale = lora_pairs(adapter)
    with safe_open(str(adapter / "head_tokens.safetensors"), framework="pt") as h:
        head_ids, head_rows = h.get_tensor("ids").long(), h.get_tensor("rows")
    wmap = checkpoint_index(base)
    to_ckpt = {text_key(k): k for k in wmap if text_key(k) is not None}
    missing = [k for k in pairs if k not in to_ckpt]
    assert not missing, f"LoRA targets without checkpoint tensor: {missing[:5]}"
    changed = {to_ckpt[k]: k for k in pairs}                       # ckpt weight key -> adapter key
    embed_key = to_ckpt["model.embed_tokens.weight"]

    handles = {}

    def h_of(fname):
        if fname not in handles:
            handles[fname] = safe_open(str(base / fname), framework="pt")
        return handles[fname]

    fp8 = {k: scale_key(k) for k in changed if scale_key(k) in wmap}
    for k in changed:
        dt = h_of(wmap[k]).get_slice(k).get_dtype()
        if k not in fp8:
            assert "F8" not in dt.upper(), f"{k} is {dt} but has no {scale_key(k)}: unsupported FP8 layout"
    edt = h_of(wmap[embed_key]).get_slice(embed_key).get_dtype()
    assert "F8" not in edt.upper(), f"embed_tokens is {edt}; head rows need an unquantized embedding"
    eshape = h_of(wmap[embed_key]).get_slice(embed_key).get_shape()
    assert int(head_ids.max()) < eshape[0], (head_ids.tolist(), eshape)

    touched = set(changed) | set(fp8.values()) | {embed_key}
    shards = sorted(set(wmap.values()))
    rewrite = sorted({wmap[k] for k in touched})
    link = [f for f in shards if f not in rewrite]
    need = sum((base / f).stat().st_size for f in rewrite + (link if a.materialize else []))
    out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out.parent).free
    qc = json.loads((base / "config.json").read_text())
    qc = qc.get("quantization_config") or qc.get("text_config", {}).get("quantization_config") or {}
    print(f"base={base}\nadapter={adapter}\nout={out}\n"
          f"quant={qc.get('quant_method', 'none')} block={qc.get('weight_block_size')} | LoRA pairs={len(pairs)} "
          f"(FP8 targets {len(fp8)}, plain {len(changed) - len(fp8)}), scale={scale}, head ids={head_ids.tolist()}\n"
          f"rewrite {len(rewrite)}/{len(shards)} shards, {'copy' if a.materialize else 'symlink'} {len(link)}; "
          f"needs {need / 2**30:.1f} GiB, free at {out.parent}: {free / 2**30:.1f} GiB", flush=True)
    if fp8:
        bs = qc.get("weight_block_size") or [BLOCK, BLOCK]
        assert list(bs) == [BLOCK, BLOCK], f"block size {bs} != {BLOCK}"
    if a.dry_run:
        print("DRY_RUN_OK")
        return
    if free < need * 1.05:
        print("NOT ENOUGH SPACE", flush=True)
        sys.exit(3)

    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for f in link:
        if a.materialize:
            shutil.copy2(os.path.realpath(base / f), tmp / f)
        else:
            os.symlink(os.path.realpath(base / f), tmp / f)
    for f in base.iterdir():                        # config, index, processor, chat template ...
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, tmp / f.name)
    for f in (adapter / "tokenizer").iterdir():     # tokenizer with the head tokens
        shutil.copy2(f, tmp / f.name)

    def delta_of(k, shape):
        p = pairs[changed[k]]
        d = (p["B"].float() @ p["A"].float()) * scale
        assert d.shape == tuple(shape), (k, d.shape, shape)
        return d

    fp8_new, errs = {}, []

    def fp8_merge(k):                               # computes weight and scale together, cached
        if k not in fp8_new:
            sk = fp8[k]
            q = h_of(wmap[k]).get_tensor(k)
            s = h_of(wmap[sk]).get_tensor(sk)
            assert q.dim() == 2, (k, q.shape)
            w = fp8_dequant(q, s) + delta_of(k, q.shape)
            q2, s2 = fp8_quant(w, q.dtype, s.dtype)
            errs.append(((fp8_dequant(q2, s2) - w).norm() / w.norm()).item())
            fp8_new[k] = (q2, s2)
        return fp8_new[k]

    inv_scale = {sk: k for k, sk in fp8.items()}
    merged = embed_done = 0
    for i, fname in enumerate(rewrite):
        tensors = {}
        with safe_open(str(base / fname), framework="pt") as h:
            for key in h.keys():
                if key in fp8:
                    t = fp8_merge(key)[0]
                    merged += 1
                elif key in inv_scale:
                    t = fp8_merge(inv_scale[key])[1]
                elif key in changed:
                    t = h.get_tensor(key)
                    t = (t.float() + delta_of(key, t.shape)).to(t.dtype)
                    merged += 1
                else:
                    t = h.get_tensor(key)
                if key == embed_key:
                    t = t.clone()
                    t[head_ids] = head_rows.to(t.dtype)
                    embed_done += 1
                tensors[key] = t.contiguous()
        save_file(tensors, str(tmp / fname), metadata={"format": "pt"})
        print(f"  shard {i + 1}/{len(rewrite)} {fname}: {len(tensors)} tensors", flush=True)
        del tensors
    assert merged == len(pairs), (merged, len(pairs))
    assert embed_done == 1
    info = {"base": str(base), "adapter": str(adapter), "lora_scale": scale, "merged_tensors": merged,
            "fp8_requantized": len(fp8), "fp8_requant_rel_err_max": max(errs) if errs else None,
            "rewritten_shards": len(rewrite), "linked_or_copied_shards": len(link), "materialized": a.materialize,
            "head_ids": head_ids.tolist(), "seconds": round(time.time() - t0)}
    (tmp / "MERGE_INFO.json").write_text(json.dumps(info, indent=2))
    (tmp / "MERGE_DONE").write_text("ok\n")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    print("merged ->", out, json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
