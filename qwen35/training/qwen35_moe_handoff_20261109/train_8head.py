#!/usr/bin/env python3
"""Eight-head Qwen3.5 LoRA on independent packed tool-calling branches.

Same packing / isolation design as the 4B train.py (RoPE reset, FA2 varlen,
DeltaNet cu_seqlens + conv seq_idx, startup isolation self-test). Differences:
  * frozen BF16 base, LoRA (FP32) on attention, DeltaNet projections and the
    shared expert; routed experts and the router stay frozen (no routing drift)
  * the four head tokens use padding rows; only a (4, H) delta is trained
  * per-token loss weights: head weights, and empty command targets down-weighted
  * PARALLEL='ddp'    : one full model per GPU (model fits one GPU, e.g. 35B-A3B)
    PARALLEL='layers' : one process, layers spread over all visible GPUs (122B)
    PARALLEL='ep'     : data parallel for everything except routed experts, which are split
                        across ranks (expert parallel, all_to_all); LoRA grads all-reduced

Launch:  SFT_CONFIG=moe/config_35b.py torchrun --standalone --nproc_per_node=4 moe/train_moe.py
         SFT_CONFIG=moe/config_122b.py torchrun --standalone --nproc_per_node=1 moe/train_moe.py
Env overrides: SFT_OUTPUT_DIR, SFT_MAX_STEPS.
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import packed_store
import common8 as C  # noqa: E402  (must precede torch: FLA backend env)

import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

DEFAULTS = dict(
    PARALLEL="ddp", EXPERT_DTYPE="bf16", EPOCHS=1, MAX_STEPS=None, PACKS_PER_GPU_PER_STEP=4, PACKS_PER_FORWARD=1,
    CE_CHUNK_TOKENS=512, MOE_IMPL="auto",
    LR=5e-5, EMB_LR=5e-5, LR_SCHEDULE="cosine", MIN_LR_RATIO=0.1, WEIGHT_DECAY=0.0, WARMUP_STEPS=20,
    GRAD_CLIP=1.0, LORA_RANK=64, LORA_ALPHA=128, LORA_DROPOUT=0.0,
    LORA_TARGETS=("q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b",
                  "in_proj_a", "out_proj", "gate_proj", "up_proj", "down_proj"),
    HEAD_LOSS_WEIGHTS={"content": 0.01, "function": 2.0, "arg1": 1.1, "arg2": 1.2,
                       "arg3": 1.3, "arg4": 1.4, "arg5": 1.4, "arg6": 1.4},
    SELF_TEST=True, SELF_TEST_FATAL=True, SELF_TEST_MEAN_TOL=0.02, SELF_TEST_MAX_TOL=0.5,
    GRAD_CHECKPOINTING=True, HEAD_DECAY_START=0.5, HEAD_DECAY_END=0.75,
    CONTENT_MIN_FACTOR=0.1, FUNCTION_MIN_FACTOR=0.2, HEAD_DECAY_ENABLED=True,
    WALL_SECONDS=0, VALIDATION_MAX_PACKS=128, LOG_EVERY=10, SAVE_EVERY=100, MAX_CONSECUTIVE_SKIPS=3, SEED=1101,
)
REQUIRED = ("MODEL_DIR", "DATA_DIR", "OUTPUT_DIR")


def load_config():
    path = os.environ.get("SFT_CONFIG")
    if not path or not os.path.isfile(path):
        raise SystemExit(f"SFT_CONFIG not found: {path}")
    spec = importlib.util.spec_from_file_location("moe_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = {k: getattr(mod, k) for k in REQUIRED}
    cfg.update({k: getattr(mod, k, v) for k, v in DEFAULTS.items()})
    if os.environ.get("SFT_DATA_DIR"):
        cfg["DATA_DIR"] = os.environ["SFT_DATA_DIR"]
    if os.environ.get("SFT_WALL_SECONDS"):
        cfg["WALL_SECONDS"] = float(os.environ["SFT_WALL_SECONDS"])
    if os.environ.get("SFT_GC"):
        cfg["GRAD_CHECKPOINTING"] = os.environ["SFT_GC"] == "1"
    if os.environ.get("SFT_PPF"):
        cfg["PACKS_PER_FORWARD"] = int(os.environ["SFT_PPF"])
    if os.environ.get("SFT_OUTPUT_DIR"):
        cfg["OUTPUT_DIR"] = os.environ["SFT_OUTPUT_DIR"]
    if os.environ.get("SFT_MAX_STEPS"):
        v = os.environ["SFT_MAX_STEPS"].strip().lower()
        cfg["MAX_STEPS"] = None if v in ("", "0", "none") else int(v)
    assert cfg["PARALLEL"] in ("ddp", "layers", "ep")
    assert cfg["EXPERT_DTYPE"] in ("bf16", "fp8")
    assert cfg["EXPERT_DTYPE"] == "bf16" or cfg["PARALLEL"] == "ep", "FP8 expert storage is implemented for EP"
    assert cfg["PACKS_PER_GPU_PER_STEP"] % cfg["PACKS_PER_FORWARD"] == 0
    assert cfg["MOE_IMPL"] in ("auto", "grouped", "loop", "eager")
    assert set(cfg["HEAD_LOSS_WEIGHTS"]) == set(C.HEADS)
    cfg["CONFIG_PATH"] = path
    return cfg


CFG = load_config()
RANK = int(os.environ.get("RANK", 0))


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def _ce(h, weight, targets, delta, ids):
    # Tied special-token delta participates in both input embeddings and output
    # logits. Only the 17 special rows train; the large base matrix stays frozen.
    logits = F.linear(h, weight).float()
    adjustment = F.linear(h.float(), delta.float()).float()
    logits = logits.index_add(-1, ids, adjustment)
    return F.cross_entropy(logits, targets, reduction="none")


class PackedLM(nn.Module):
    """Returns (weighted CE sum, plain CE sum) over the supervised positions."""

    def __init__(self, model, ce_chunk):
        super().__init__()
        self.model = model
        self.ce_chunk = ce_chunk

    def hidden(self, input_ids, position_ids, cu_i32, max_len, fa_varlen=True):
        if fa_varlen:
            kw = dict(cu_seq_lens_q=cu_i32, cu_seq_lens_k=cu_i32, max_length_q=max_len, max_length_k=max_len)
        else:   # explicit one-sequence boundary: the leakage negative control
            n = input_ids.shape[1]
            one = torch.tensor([0, n], dtype=torch.int32, device=input_ids.device)
            kw = dict(cu_seq_lens_q=one, cu_seq_lens_k=one, max_length_q=n, max_length_k=n)
        out = self.model.get_decoder()(input_ids=input_ids, position_ids=position_ids, use_cache=False, **kw)
        h = out.last_hidden_state
        assert h.shape[:2] == input_ids.shape, ("decoder changed sequence shape", tuple(h.shape))
        return h

    def token_ce(self, hidden, hidden_index, targets):
        weight = self.model.get_output_embeddings().weight
        special = self.model.get_output_embeddings()
        if not hasattr(special, 'delta'):
            special = self.model.get_input_embeddings()
        dev = hidden.device
        h = hidden.reshape(-1, hidden.shape[-1]).index_select(0, hidden_index.to(dev))
        targets = targets.to(dev)
        parts = []
        for s in range(0, h.shape[0], self.ce_chunk):
            args = (h[s:s + self.ce_chunk], weight, targets[s:s + self.ce_chunk], special.delta, special.ids)
            parts.append(checkpoint(_ce, *args, use_reentrant=False)
                         if CFG["GRAD_CHECKPOINTING"] and torch.is_grad_enabled() else _ce(*args))
        return torch.cat(parts) if parts else hidden.sum() * 0.0

    def forward(self, input_ids, position_ids, cu_i32, max_len, hidden_index, targets, weights):
        ce = self.token_ce(self.hidden(input_ids, position_ids, cu_i32, max_len), hidden_index, targets)
        return (ce * weights.to(ce.device)).sum(), ce.detach().sum()


def run_loss(module, b):
    return module(b["input_ids"], b["position_ids"], b["cu_seqlens_i32"], b["max_len"],
                  b["hidden_index"], b["targets"], b["weights"])


def segment_weight(name, ids, mask):
    head = name.rsplit(":", 1)[1]
    weights = torch.full_like(ids, CFG["HEAD_LOSS_WEIGHTS"][head], dtype=torch.float32)
    is_special = torch.isin(ids, torch.tensor(TOKEN_IDS, dtype=torch.long))
    weights[is_special] *= 1.5
    if head.startswith("arg"):
        focal = {"arg1": 1.0, "arg2": 1.0, "arg3": 1.5, "arg4": 2.0, "arg5": 2.5, "arg6": 3.0}[head]
        weights[ids != NULL_ID] *= focal
    return weights


def head_factors_at(progress):
    if not CFG["HEAD_DECAY_ENABLED"]:
        return {"content": 1.0, "function": 1.0}
    p = min(1.0, max(0.0, (progress-CFG["HEAD_DECAY_START"])/(CFG["HEAD_DECAY_END"]-CFG["HEAD_DECAY_START"])))
    return {h: low+(1-low)*0.5*(1+math.cos(math.pi*p))
            for h,low in [("content",CFG["CONTENT_MIN_FACTOR"]),("function",CFG["FUNCTION_MIN_FACTOR"])]}


def scheduled_batch(segments, total_len, pad_id, factors):
    batch=C.build_batch(segments,total_len,pad_id)
    # Keep the original global denominator: per-head reductions must not
    # renormalize themselves away or inflate the other heads' gradients.
    offset=0
    for ids,mask,weights in segments:
        pos=mask.nonzero(as_tuple=True)[0]
        assert pos.numel()>0
        head_token=int(ids[int(pos[0])-1])
        factor=1.0
        for head in ("content","function"):
            if head_token == HEAD_IDS[head]: factor=factors[head]
        n=int(pos.numel())
        batch["weights"][offset:offset+n] *= factor
        offset+=n
    assert offset==batch["n_sup"]
    return batch


# ---------------------------------------------------------------------------
# Isolation self-test (same protocol as the 4B trainer)
# ---------------------------------------------------------------------------
@torch.no_grad()
def isolation_self_test(core, packs, pack_len, pad_id, device):
    scan = sorted(range(min(len(packs), 512)), key=lambda i: -len(packs[i]["cu_seqlens"]))
    segs = C.pack_segments(packs[scan[0]], pack_len, segment_weight)
    full = C.to_device(C.build_batch(segs, pack_len, pad_id), device)
    offsets, n = full["seg_offsets"], len(segs)
    picks = sorted({i for i in (1, 2, n // 2, n - 1) if 1 <= i < n and segs[i][0].numel() >= 8})
    assert picks, "self-test needs a pack with >= 2 segments"

    def nll(batch, start, length, isolate):
        pos = torch.arange(start + 1, start + length, device=device)
        tgt = batch["input_ids"][0, pos]
        ctx = C.packed_context(batch["cu_seqlens"], batch["seq_idx"]) if isolate else C.packed_context()
        with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            h = core.hidden(batch["input_ids"], batch["position_ids"], batch["cu_seqlens_i32"],
                            batch["max_len"], fa_varlen=isolate)
            return core.token_ce(h, pos - 1, tgt).float().cpu()

    core.eval()
    worst_mean = worst_max = ctrl = 0.0
    log(f"Isolation self-test on {len(picks)} non-leading segments (shape (1, {pack_len})):")
    for i in picks:
        length = segs[i][0].numel()
        alone = C.to_device(C.build_batch([segs[i]], pack_len, pad_id), device)
        ref = nll(alone, 0, length, True)
        got = nll(full, offsets[i], length, True)
        ctl = nll(full, offsets[i], length, False)
        d, dc = (got - ref).abs(), (ctl - ref).abs()
        log(f"  seg#{i:<3d} len={length:<5d} offset={offsets[i]:<6d} isolated |dNLL| mean={d.mean():.2e} "
            f"max={d.max():.2e} | no-isolation mean={dc.mean():.2e} max={dc.max():.2e}")
        worst_mean, worst_max, ctrl = max(worst_mean, d.mean().item()), max(worst_max, d.max().item()), \
            max(ctrl, dc.mean().item())
    core.train()
    if ctrl <= max(4 * worst_mean, 1e-3):
        log("WARNING: no-isolation control is not clearly worse; self-test may be weak.")
    if worst_mean > CFG["SELF_TEST_MEAN_TOL"] or worst_max > CFG["SELF_TEST_MAX_TOL"]:
        msg = f"Segment isolation FAILED: mean={worst_mean:.3e}, max={worst_max:.3e}"
        if CFG["SELF_TEST_FATAL"]:
            raise RuntimeError(msg)
        log(f"WARNING: {msg} (SELF_TEST_FATAL=False, continuing)")
        return
    log(f"Isolation self-test PASSED (worst mean={worst_mean:.2e}, max={worst_max:.2e})")


def lr_factor(step, total):
    if step <= CFG["WARMUP_STEPS"]:
        return step / max(1, CFG["WARMUP_STEPS"])
    if CFG["LR_SCHEDULE"] == "constant":
        return 1.0
    p = min(1.0, (step - CFG["WARMUP_STEPS"]) / max(1, total - CFG["WARMUP_STEPS"]))
    return CFG["MIN_LR_RATIO"] + (1 - CFG["MIN_LR_RATIO"]) * 0.5 * (1 + math.cos(math.pi * p))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(tok, head_ids, devices, moe_tokens, world):
    from peft import LoraConfig, get_peft_model
    ep = CFG["PARALLEL"] == "ep"
    expert_slice = None
    if ep:
        n_exp = C.text_config(CFG["MODEL_DIR"]).num_experts
        assert n_exp % world == 0, (n_exp, world)
        n_local = n_exp // world
        expert_slice = (RANK * n_local, (RANK + 1) * n_local)
    model, plan = C.load_text_model(CFG["MODEL_DIR"], devices, attn="flash_attention_2", log=log,
                                    expert_slice=expert_slice)
    assert model.config._attn_implementation == "flash_attention_2"
    model.requires_grad_(False)
    if ep:
        C.install_moe_ep(model, dist.new_group(ranks=list(range(world))), RANK, world, log=log)
        if CFG["EXPERT_DTYPE"] == "fp8":
            C.quantize_experts_fp8(model, log=log)
        choice = None
    else:
        choice = C.install_moe_impl(model, CFG["MOE_IMPL"], tokens=moe_tokens, log=log)
    if world > 1 and choice is not None:             # every rank must run the same implementation
        obj = [choice]
        dist.broadcast_object_list(obj, src=0)
        if obj[0] != choice:
            C.install_moe_impl(model, obj[0], log=lambda *a: None)
    n_patched = C.patch_linear_attention(model, log=log)
    head = C.install_head_embedding(model, tok, head_ids)
    log(f"Patched {n_patched} Gated DeltaNet layers; head rows init norm "
        f"{head.final_rows().norm(dim=1).mean():.3f} vs vocab mean "
        f"{head.base.weight[:4096].float().norm(dim=1).mean():.3f}")
    model = C.dispatch_layers(model, plan)
    if CFG["GRAD_CHECKPOINTING"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    log(f"Gradient checkpointing={CFG['GRAD_CHECKPOINTING']}")
    lcfg = LoraConfig(r=CFG["LORA_RANK"], lora_alpha=CFG["LORA_ALPHA"], lora_dropout=CFG["LORA_DROPOUT"],
                      target_modules=list(CFG["LORA_TARGETS"]), bias="none")
    model = get_peft_model(model, lcfg)
    head.delta.requires_grad_(True)
    output_head = model.get_output_embeddings()
    if hasattr(output_head, 'delta'):
        output_head.delta.requires_grad_(True)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    for p in lora:
        if p.dtype != torch.float32:
            p.data = p.data.float()
    others = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    assert others == [n for n, _ in model.named_parameters()
                      if n.endswith(('embed_tokens.delta', 'lm_head.delta'))], others
    by_kind = {}
    for n, p in model.named_parameters():
        if "lora_A" in n:
            kind = n.split(".lora_A")[0].rsplit(".", 1)[-1]
            by_kind[kind] = by_kind.get(kind, 0) + 1
    log(f"LoRA r={CFG['LORA_RANK']} alpha={CFG['LORA_ALPHA']}: {sum(p.numel() for p in lora)/1e6:.1f}M params "
        f"on {by_kind}; head delta {tuple(head.delta.shape)}")
    if "experts" in str(by_kind) or "gate" in by_kind:
        raise RuntimeError("LoRA reached routed experts or the router; check LORA_TARGETS")
    return model, head, lora


def save(model, head, tok, head_ids, path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path), save_embedding_layers=False)
    C.save_head_rows(path / "head_tokens.safetensors", head, head_ids, model.get_output_embeddings())
    tok.save_pretrained(str(path / "tokenizer"))
    (path / "train_config.json").write_text(json.dumps(CFG, ensure_ascii=False, indent=2, default=str))
    log(f"Saved adapter and independent input/output token rows: {path}")


@torch.no_grad()
def validate(core, path, pack_len, pad_id, device, world):
    obj = packed_store.load(path.parent,"val") if (path.parent / "INDEX.json").exists() else torch.load(path, map_location="cpu", weights_only=True)
    stats = torch.zeros(4, device=device, dtype=torch.float64)
    core.eval()
    selected = random.Random(1101).sample(range(len(obj["packs"])), min(len(obj["packs"]), CFG["VALIDATION_MAX_PACKS"]))
    if CFG["PARALLEL"] == "ep" and world > 1:
        # Every EP forward performs collectives; ranks must execute equal numbers.
        assert len(selected) >= world and len(selected) % world == 0, "EP validation packs must divide world"
    for i in selected[RANK::world]:
        pack = obj["packs"][i]
        batch = C.to_device(C.build_batch(C.pack_segments(pack, pack_len, segment_weight), pack_len, pad_id), device)
        with C.packed_context(batch["cu_seqlens"], batch["seq_idx"]), torch.autocast("cuda", dtype=torch.bfloat16):
            weighted, plain = run_loss(core, batch)
        assert torch.isfinite(weighted) and torch.isfinite(plain), "non-finite validation loss"
        stats += torch.tensor([float(weighted), float(plain), batch["w_sup"], batch["n_sup"]],
                              device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(stats)
    weighted, plain, weight, count = stats.tolist()
    result = {"loss_w": weighted / weight, "ce": plain / count, "supervised_tokens": int(count),
              "evaluated_packs": len(selected), "available_packs": len(obj["packs"]), "pack_sampling": "uniform seed1101"}
    log("Validation: " + json.dumps(result))
    core.train()
    return result


def main():
    rank, local_rank, world = RANK, int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if CFG["PARALLEL"] == "layers":
        assert world == 1, "PARALLEL='layers' runs as a single process (nproc_per_node=1)"
        devices = [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    else:
        devices = [torch.device("cuda", local_rank)]
    device = devices[0]
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(CFG["SEED"] + rank)
    random.seed(CFG["SEED"] + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    data = Path(CFG["DATA_DIR"])
    mmap_data = (data / "INDEX.json").exists()
    manifest = json.loads((data / ("INDEX.json" if mmap_data else "PREPARE_MANIFEST.json")).read_text())
    if not mmap_data:
        assert Path(manifest["model"]).resolve() == Path(CFG["MODEL_DIR"]).resolve(), "data prepared for another model"
    tok, head_ids = C.load_tokenizer(data / "tokenizer")
    global TOKEN_IDS, NULL_ID, HEAD_IDS
    HEAD_IDS = head_ids
    TOKEN_IDS = [tok.convert_tokens_to_ids(t) for t in C.SPECIAL_TOKENS]
    NULL_ID = tok.convert_tokens_to_ids("<|null|>")
    if not mmap_data:
        assert head_ids == {k: int(v) for k, v in manifest["head_ids"].items()}
        assert manifest["special_token_ids"] == dict(zip(C.SPECIAL_TOKENS, TOKEN_IDS))
    else:
        base_tok, base_heads = C.load_tokenizer(CFG["MODEL_DIR"])
        assert base_tok.get_vocab() == tok.get_vocab() and base_heads == head_ids
        assert manifest["train_validation_prompt_group_overlap"] == 0
        assert manifest['thinking_protocol']=='explicit_empty_think_before_head_v1'
        assert json.loads((data/'PREFLIGHT.json').read_text())['thinking_protocol']==manifest['thinking_protocol']
    pad_id = tok.convert_tokens_to_ids(C.END_TOKEN)
    train_obj = packed_store.load(data,"train") if mmap_data else torch.load(data / "train_packed.pt", map_location="cpu", weights_only=True)
    packs, pack_len = train_obj["packs"], int(train_obj["pack_length"])
    out = Path(CFG["OUTPUT_DIR"])
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "run_config.json").write_text(json.dumps({"config": CFG, "world": world, "devices": [str(d) for d in devices],
                                                         "manifest": manifest}, ensure_ascii=False, indent=2, default=str))
    log(f"world={world} parallel={CFG['PARALLEL']} devices={[str(d) for d in devices]} "
        f"gpu={torch.cuda.get_device_name(device)} torch={torch.__version__} TileLang={C.USE_TILELANG}")
    log(f"data={data} packs={len(packs)} pack_len={pack_len} head_ids={head_ids}")

    ppf = CFG["PACKS_PER_FORWARD"]
    model, head, lora = build_model(tok, head_ids, devices, moe_tokens=pack_len * ppf, world=world)
    # EP has no DDP constructor to synchronize rank-dependent LoRA/head init.
    # Frozen expert shards differ intentionally; only replicated trainable tensors synchronize.
    if CFG["PARALLEL"] == "ep" and world > 1:
        for parameter in model.parameters():
            if parameter.requires_grad:
                dist.broadcast(parameter.data, src=0)
        log("EP replicated LoRA and input/output token rows synchronized from rank 0")
    core = PackedLM(model, CFG["CE_CHUNK_TOKENS"])
    core.train()
    if CFG["SELF_TEST"]:
        isolation_self_test(core, packs, pack_len, pad_id, device)
    ep = CFG["PARALLEL"] == "ep"
    if world > 1:
        dist.barrier()
    if world > 1 and not ep:
        runner = DDP(core, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False,
                     find_unused_parameters=False, gradient_as_bucket_view=True)
    else:          # EP: plain module; LoRA grads are all-reduced by hand after backward
        runner = core
    token_parameters = [head.delta]
    if hasattr(model.get_output_embeddings(), 'delta'):
        token_parameters.append(model.get_output_embeddings().delta)
    optimizer = torch.optim.AdamW([
        {"params": lora, "lr": CFG["LR"], "weight_decay": CFG["WEIGHT_DECAY"], "base_lr": CFG["LR"]},
        {"params": token_parameters, "lr": CFG["EMB_LR"], "weight_decay": 0.0, "base_lr": CFG["EMB_LR"]},
    ])
    trainable = lora + token_parameters

    ppg = CFG["PACKS_PER_GPU_PER_STEP"]
    steps_per_epoch = math.ceil(len(packs) / (world * ppg))
    planned = CFG["EPOCHS"] * steps_per_epoch
    if CFG["MAX_STEPS"] is not None:
        planned = min(planned, CFG["MAX_STEPS"])
    log(f"global batch={world * ppg} packs/step ({ppf} pack(s) = {pack_len * ppf} tokens per forward), "
        f"steps/epoch={steps_per_epoch}, planned steps={planned}")

    step = skipped = consecutive = 0
    interrupted = False
    t_start = time.perf_counter()
    loss_hist = []
    for epoch in range(CFG["EPOCHS"]):
        order = list(range(len(packs)))
        random.Random(CFG["SEED"] + epoch).shuffle(order)
        order += [order[i % len(order)] for i in range(steps_per_epoch * world * ppg - len(order))]
        local = order[rank::world]
        for s in range(0, len(local), ppg):
            head_factors = head_factors_at((step + 1) / planned)
            ids = local[s:s + ppg]
            batches = [scheduled_batch([seg for i in ids[k:k + ppf]
                                      for seg in C.pack_segments(packs[i], pack_len, segment_weight)],
                                     pack_len * ppf, pad_id, head_factors)
                       for k in range(0, len(ids), ppf)]
            stats = torch.tensor([sum(b["w_sup"] for b in batches), sum(b["n_sup"] for b in batches)],
                                 dtype=torch.float64, device=device)
            if world > 1:
                dist.all_reduce(stats)
            w_global, n_global = stats.tolist()
            assert w_global > 0
            scale = (1.0 if ep else world) / w_global    # DDP averages over ranks; EP sums by hand
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            wsum = psum = 0.0
            for j, cb in enumerate(batches):
                b = C.to_device(cb, device)
                sync = runner.no_sync() if (world > 1 and not ep and j < len(batches) - 1) else contextlib.nullcontext()
                with sync, C.packed_context(b["cu_seqlens"], b["seq_idx"]):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        weighted, plain = run_loss(runner, b)
                    (weighted * scale).backward()
                wsum += float(weighted.detach())
                psum += float(plain)
                del b, weighted, plain
            if ep and world > 1:                        # replicated LoRA / head delta: sum grads over ranks
                grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in trainable]
                flat = torch._utils._flatten_dense_tensors(grads)
                dist.all_reduce(flat)
                for p, g in zip(trainable, torch._utils._unflatten_dense_tensors(flat, grads)):
                    p.grad = g
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, CFG["GRAD_CLIP"])
            step += 1
            fac = lr_factor(step, planned)
            if bool(torch.isfinite(grad_norm).item()):
                for g in optimizer.param_groups:
                    g["lr"] = g["base_lr"] * fac
                optimizer.step()
                consecutive = 0
                if step == 1 and world > 1:
                    chk = torch.stack([p.detach().double().sum() for p in trainable])
                    lo, hi = chk.clone(), chk.clone()
                    dist.all_reduce(lo, op=dist.ReduceOp.MIN)
                    dist.all_reduce(hi, op=dist.ReduceOp.MAX)
                    assert float((hi - lo).abs().max()) < 1e-8, "weights differ between ranks after step 1"
                    log("first-step replica check PASSED")
            else:
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                consecutive += 1
                log(f"!! non-finite grad at step {step}: skipped")
                if consecutive >= CFG["MAX_CONSECUTIVE_SKIPS"]:
                    raise RuntimeError("too many consecutive non-finite steps")
            for d in devices:
                torch.cuda.synchronize(d)
            dt = time.perf_counter() - t0
            m = torch.tensor([wsum, psum, sum(b["real_tokens"] for b in batches), dt], dtype=torch.float64, device=device)
            if world > 1:
                dist.all_reduce(m)
            wl, pl, real, dt_sum = m.tolist()
            step_s = dt_sum / world
            loss_hist.append(pl / n_global)
            if step == 1 or step % CFG["LOG_EVERY"] == 0:
                peak = max(torch.cuda.max_memory_allocated(d) for d in devices) / 2**30
                eta = (planned - step) * step_s / 3600
                log(f"epoch={epoch + 1} step={step}/{planned} loss_w={wl / w_global:.4f} ce={pl / n_global:.4f} "
                    f"lr={CFG['LR'] * fac:.2e} tok/s={real / step_s:,.0f} sup_tok/s={n_global / step_s:,.0f} "
                    f"step_s={step_s:.2f} grad_norm={float(grad_norm):.3f} skipped={skipped} "
                    f"peak_vram={peak:.1f}GB eta={eta:.2f}h "
                    f"content_factor={head_factors['content']:.4f} function_factor={head_factors['function']:.4f}")
            if step % CFG["SAVE_EVERY"] == 0 and step < planned:
                if world > 1:
                    dist.barrier()
                if rank == 0:
                    save(model, head, tok, head_ids, out / f"step-{step:06d}")
                    torch.save({"optimizer": optimizer.state_dict(), "step": step, "planned": planned,
                                "epoch": epoch, "local_offset": s, "seed": CFG["SEED"]},
                               out / f"step-{step:06d}" / "optimizer_state.pt")
                if world > 1:
                    dist.barrier()
                state_dir = out / f"step-{step:06d}"
                torch.save({'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device),
                            'python': random.getstate(), 'epoch': epoch, 'next_local_offset': s+ppg,
                            'step': step, 'world': world}, state_dir / f'rng-rank{rank:03d}.pt')
                if world > 1:
                    dist.barrier()
            deadline = torch.tensor(int(CFG["WALL_SECONDS"] > 0 and time.perf_counter()-t_start >= CFG["WALL_SECONDS"]),device=device)
            if world > 1: dist.all_reduce(deadline,op=dist.ReduceOp.MAX)
            if deadline.item() and step < planned:
                interrupted = True
                break
            if step >= planned:
                break
        if step >= planned or interrupted:
            break

    if world > 1:
        dist.barrier()
    # Final optimizer/RNG state is retained along with the final adapter.
    if not interrupted:
        if rank == 0:
            (out / 'final').mkdir(parents=True, exist_ok=True)
            torch.save({'optimizer': optimizer.state_dict(), 'step': step, 'planned': planned,
                        'epoch': epoch, 'next_local_offset': s+ppg, 'seed': CFG['SEED'], 'world': world},
                       out / 'final/optimizer_state.pt')
        if world > 1:
            dist.barrier()
        torch.save({'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device),
                    'python': random.getstate(), 'epoch': epoch, 'next_local_offset': s+ppg,
                    'step': step, 'world': world}, out / f'final/rng-rank{rank:03d}.pt')
        if world > 1:
            dist.barrier()
    if interrupted:
        if rank == 0:
            save(model, head, tok, head_ids, out / "interrupted")
            torch.save({"optimizer": optimizer.state_dict(), "step": step, "planned": planned,
                        "epoch": epoch, "local_offset": s, "seed": CFG["SEED"]}, out / "interrupted/optimizer_state.pt")
            (out / "INTERRUPTED.json").write_text(json.dumps({"step":step,"planned":planned,"reason":"wallclock budget", "complete":False}))
        if world > 1:
            dist.barrier(); dist.destroy_process_group()
        return
    validation = validate(core, data / "val_packed.pt", pack_len, pad_id, device, world)
    if rank == 0:
        save(model, head, tok, head_ids, out / "final")
        k = max(1, len(loss_hist) // 10)
        summary = {"completed_steps": step, "planned_steps": planned, "skipped_steps": skipped,
                   "validation": validation,
                   "ce_first10pct": sum(loss_hist[:k]) / k, "ce_last10pct": sum(loss_hist[-k:]) / k,
                   "wall_seconds": time.perf_counter() - t_start,
                   "peak_vram_gib": max(torch.cuda.max_memory_allocated(d) for d in devices) / 2**30}
        assert step == planned and skipped == 0, summary
        (out / "training_summary.json").write_text(json.dumps(summary, indent=2))
        (out / "training_complete.json").write_text(json.dumps(summary, indent=2))
        log(json.dumps(summary))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
