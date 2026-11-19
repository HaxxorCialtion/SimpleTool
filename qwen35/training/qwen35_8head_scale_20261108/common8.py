"""Shared pieces for head-LoRA training / eval on Qwen3.5 / 3.6 (MoE or dense).

Import this module BEFORE torch anywhere: like the 4B train.py it configures the
FLA backend (Hopper + Triton >= 3.4 needs TileLang, FLA #640) before fla loads.

Contents
  * text-only model loading from a ConditionalGeneration (VLM) checkpoint:
    meta init -> shard-by-shard assign, visual/mtp skipped, every key verified
  * layer placement over several GPUs (accelerate hooks) for models > 1 GPU
  * head-token embedding: 4 padding rows initialised, only a (4, H) delta trains
  * Gated DeltaNet varlen injection (cu_seqlens / seq_idx), device aware
  * packs -> fixed-length varlen batch with per-token loss weights
"""
import os
import re
import sys
from importlib import metadata


def _triton_version():
    try:
        m = re.match(r"(\d+)\.(\d+)", metadata.version("triton"))
        return (int(m.group(1)), int(m.group(2))) if m else None
    except metadata.PackageNotFoundError:
        return None


TRITON_VERSION = _triton_version()
USE_TILELANG = TRITON_VERSION is not None and TRITON_VERSION >= (3, 4)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if USE_TILELANG:
    os.environ.pop("FLA_DISABLE_BACKEND_DISPATCH", None)
else:
    os.environ.pop("FLA_TILELANG", None)
    os.environ["FLA_DISABLE_BACKEND_DISPATCH"] = "1"
    sys.modules["tilelang"] = None

import contextlib  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

HEADS = ("content", "function", "arg1", "arg2", "arg3", "arg4", "arg5", "arg6")
SPECIAL_TOKENS = tuple(t for h in HEADS for t in (f"<{h}>", f"</{h}>")) + ("<|null|>",)
OLD_INSTR = {
    "zh": '返回 JSON：{"heads":{"speech":"","inner":"","expression":"","command":""}}。',
    "en": 'Return JSON: {"heads":{"speech":"","inner":"","expression":"","command":""}}.',
    "ja": '出力は JSON：{"heads":{"speech":"","inner":"","expression":"","command":""}}。',
}
NEW_INSTR = {
    "zh": "当前通道由 <speech>、<expression>、<inner> 或 <command> 指定，输出该通道的正文。",
    "en": "The current channel is specified by <speech>, <expression>, <inner>, or <command>. "
          "Output the content of that channel.",
    "ja": "現在のチャネルは <speech>、<expression>、<inner>、<command> のいずれかで指定されます。"
          "そのチャネルの本文を出力してください。",
}
END_TOKEN = "<|im_end|>"


# ---------------------------------------------------------------------------
# Tokenizer / prompts
# ---------------------------------------------------------------------------
def load_tokenizer(path):
    """Register all eight start/end markers and the null output token."""
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    vocab = tok.get_vocab()
    missing = [t for t in SPECIAL_TOKENS if t not in vocab]
    if missing:
        tok.add_tokens(missing, special_tokens=True)
    head_ids = {}
    for h in HEADS:
        ids = tok.encode(f"<{h}>", add_special_tokens=False)
        assert len(ids) == 1, (h, ids)
        head_ids[h] = ids[0]
    assert len(set(head_ids.values())) == len(HEADS)
    return tok, head_ids


def build_messages(row, mode):
    """mode 'heads': channel instruction; mode 'json': the original JSON instruction."""
    m, lang = row["messages"], row["language"]
    system = m[0]["content"]
    assert system.count(OLD_INSTR[lang]) == 1, row["source_id"]
    if mode == "heads":
        system = system.replace(OLD_INSTR[lang], NEW_INSTR[lang])
    return [{"role": "system", "content": system}, m[1]]


def prompt_ids(tok, row, mode):
    text = tok.apply_chat_template(build_messages(row, mode), tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def persona_of(row):
    system = row["messages"][0]["content"]
    return system.split("\n\n", 1)[1].split("\n\nexpressions:\n", 1)[0]


# ---------------------------------------------------------------------------
# Model loading (text model out of a VLM checkpoint) + placement
# ---------------------------------------------------------------------------
def text_config(model_dir):
    cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    return getattr(cfg, "text_config", cfg)


def map_checkpoint_key(key):
    """Checkpoint key -> text-model key, or None to skip (vision tower, MTP head)."""
    if key.startswith("model.language_model."):
        return "model." + key[len("model.language_model."):]
    if key.startswith(("model.visual.", "mtp.", "model.mtp.")):
        return None
    if key.startswith(("model.", "lm_head.")):
        return key
    return None


def placement_plan(n_layers, devices):
    """Module-prefix -> device. Layers split evenly; embedding first, head last."""
    plan = {"model.embed_tokens": devices[0], "model.rotary_emb": devices[0],
            "model.norm": devices[-1], "lm_head": devices[-1]}
    per = -(-n_layers // len(devices))
    for i in range(n_layers):
        plan[f"model.layers.{i}"] = devices[min(i // per, len(devices) - 1)]
    return plan


def device_of(plan, name):
    best = max((p for p in plan if name == p or name.startswith(p + ".")), key=len, default=None)
    if best is None:
        raise KeyError(f"no placement for {name}")
    return plan[best]


def _instantiate(cfg, attn, dtype):
    with torch.device("meta"):
        try:
            return AutoModelForCausalLM.from_config(cfg, dtype=dtype, attn_implementation=attn)
        except TypeError:
            return AutoModelForCausalLM.from_config(cfg, torch_dtype=dtype, attn_implementation=attn)


def _set_submodule(root, name, new):
    parent, _, child = name.rpartition(".")
    setattr(root.get_submodule(parent) if parent else root, child, new)


def checkpoint_index(model_dir):
    d = Path(model_dir)
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    from safetensors import safe_open
    with safe_open(str(d / "model.safetensors"), framework="pt") as h:
        return {k: "model.safetensors" for k in h.keys()}


EXPERT_KEY = re.compile(r"\.mlp\.experts\.(gate_up_proj|down_proj)$")


def load_text_model(model_dir, devices, attn, dtype=torch.bfloat16, log=print, expert_slice=None):
    """Text-only CausalLM with real weights placed per placement_plan(devices).
    expert_slice=(a, b): load only routed experts a..b-1 (expert parallelism)."""
    from safetensors import safe_open
    cfg = text_config(model_dir)
    model = _instantiate(cfg, attn, dtype)
    plan = placement_plan(cfg.num_hidden_layers, devices)
    meta_sd = model.state_dict()
    expected = set(meta_sd)
    weight_map = checkpoint_index(model_dir)
    tied = ("lm_head.weight" in expected
            and not any(map_checkpoint_key(k) == "lm_head.weight" for k in weight_map))
    if tied:                                     # small checkpoints tie lm_head to the embedding
        assert "model.embed_tokens.weight" in expected
        expected.discard("lm_head.weight")
    by_file, unexpected = defaultdict(list), []
    for key, fname in weight_map.items():
        new = map_checkpoint_key(key)
        if new is None:
            continue
        (by_file[fname].append((key, new)) if new in expected else unexpected.append(key))
    if unexpected:
        raise RuntimeError(f"{len(unexpected)} checkpoint keys have no slot in the text model, e.g. {unexpected[:5]}")
    loaded, sliced = set(), {}
    for fname in sorted(by_file):
        sd = {}
        with safe_open(str(Path(model_dir) / fname), framework="pt") as h:
            for key, new in by_file[fname]:
                ref = meta_sd[new]
                if expert_slice is not None and EXPERT_KEY.search(new):
                    a, b = expert_slice
                    full = tuple(h.get_slice(key).get_shape())
                    if full != tuple(ref.shape):
                        raise RuntimeError(f"shape mismatch {key}: checkpoint {full} vs model {tuple(ref.shape)}")
                    sliced[new] = h.get_slice(key)[a:b].to(device=device_of(plan, new), dtype=ref.dtype)
                    continue
                shape = tuple(h.get_slice(key).get_shape())
                if shape != tuple(ref.shape):
                    raise RuntimeError(f"shape mismatch {key}: checkpoint {shape} vs model {tuple(ref.shape)}")
                sd[new] = h.get_tensor(key).to(device=device_of(plan, new), dtype=ref.dtype)
        model.load_state_dict(sd, strict=False, assign=True)
        loaded |= set(sd)
        del sd
    for new, t in sliced.items():                  # EP: local experts only (shape differs from meta)
        parent, _, leaf = new.rpartition(".")
        setattr(model.get_submodule(parent), leaf, nn.Parameter(t, requires_grad=False))
        loaded.add(new)
    missing = expected - loaded
    if missing:
        raise RuntimeError(f"{len(missing)} model tensors missing from checkpoint, e.g. {sorted(missing)[:5]}")
    # Non-persistent buffers (RoPE inv_freq) were created on meta: rebuild those modules.
    for name, mod in list(model.named_modules()):
        if not any(b.is_meta for b in mod.buffers(recurse=False)):
            continue
        if not hasattr(mod, "inv_freq"):
            raise RuntimeError(f"meta buffer in {name} ({type(mod).__name__}); do not know how to rebuild")
        dev = device_of(plan, name)
        try:
            fresh = type(mod)(config=mod.config, device=dev)
        except TypeError:
            fresh = type(mod)(mod.config).to(dev)
        _set_submodule(model, name, fresh)
    if tied:
        model.get_output_embeddings().weight = model.get_input_embeddings().weight
        assert model.get_output_embeddings().weight is model.get_input_embeddings().weight
        log("lm_head tied to embed_tokens (checkpoint has no lm_head.weight)")
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    assert not left, f"tensors still on meta: {left[:5]}"
    model.config.use_cache = False
    log(f"loaded {type(model).__name__} from {model_dir}: {len(loaded)} tensors, "
        f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, devices={sorted({str(d) for d in plan.values()})}")
    return model, plan


def dispatch_layers(model, plan):
    """Attach accelerate hooks so activations follow the placement (weights already placed)."""
    if len({str(d) for d in plan.values()}) == 1:
        return model
    from accelerate import dispatch_model
    names = [n for n, _ in model.named_parameters()]
    uncovered = [n for n in names if not any(n == p or n.startswith(p + ".") for p in plan)]
    assert not uncovered, f"params outside the plan: {uncovered[:5]}"
    device_map = {k: v.index for k, v in plan.items() if k in dict(model.named_modules())}
    return dispatch_model(model, device_map=device_map)


# ---------------------------------------------------------------------------
# Head-token embedding
# ---------------------------------------------------------------------------
class HeadTokenEmbedding(nn.Module):
    """Frozen base embedding + trainable (n_heads, H) delta on the head-token rows only."""

    def __init__(self, base, ids, rows):
        super().__init__()
        self.base = base
        ids_t = torch.tensor(list(ids), dtype=torch.long, device=base.weight.device)
        lookup = torch.zeros(base.num_embeddings, dtype=torch.long, device=base.weight.device)
        lookup[ids_t] = torch.arange(1, len(ids_t) + 1, device=base.weight.device)
        self.register_buffer("ids", ids_t, persistent=False)
        self.register_buffer("lookup", lookup, persistent=False)
        with torch.no_grad():
            base.weight[ids_t] = rows.to(base.weight.device, base.weight.dtype)
        self.delta = nn.Parameter(torch.zeros(len(ids_t), base.embedding_dim, device=base.weight.device,
                                              dtype=torch.float32))

    @property
    def weight(self):                      # transformers helpers look for .weight
        return self.base.weight

    @property
    def num_embeddings(self):
        return self.base.num_embeddings

    @property
    def embedding_dim(self):
        return self.base.embedding_dim

    def forward(self, input_ids):
        out = self.base(input_ids)
        table = torch.cat([self.delta.new_zeros(1, self.delta.shape[1]), self.delta])
        return out + F.embedding(self.lookup[input_ids], table).to(out.dtype)

    @torch.no_grad()
    def final_rows(self):
        return (self.base.weight[self.ids].float() + self.delta.float()).cpu()


def mean_init_rows(tok, embed_weight, heads=HEADS):
    rows = []
    for h in heads:
        ids = tok.encode(h, add_special_tokens=False)
        rows.append(embed_weight[torch.tensor(ids, device=embed_weight.device)].float().mean(0))
    return torch.stack(rows)


def install_head_embedding(model, tok, head_ids, rows=None):
    dec = model.get_decoder()
    base = dec.embed_tokens
    assert isinstance(base, nn.Embedding), type(base)
    token_ids = [tok.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS]
    assert max(token_ids) < base.num_embeddings, "special token id beyond embedding rows"
    assert model.get_output_embeddings().weight is base.weight, "this trainer requires tied embeddings"
    if rows is None:
        # Semantic means for all 17 rows. These same rows are used for input
        # conditioning and output prediction; no extra full vocab copy is made.
        refs = [t.strip("<>").replace("|", "") for t in SPECIAL_TOKENS]
        rows = mean_init_rows(tok, base.weight, refs)
    wrapper = HeadTokenEmbedding(base, token_ids, rows)
    dec.embed_tokens = wrapper
    return wrapper


def save_head_rows(path, wrapper, head_ids):
    from safetensors.torch import save_file
    save_file({"ids": wrapper.ids.cpu(), "rows": wrapper.final_rows()}, str(path),
              metadata={"heads": json.dumps(head_ids), "tokens": json.dumps(list(SPECIAL_TOKENS)),
                        "tied_input_output": "true"})


def load_head_rows(path, head_ids):
    from safetensors import safe_open
    with safe_open(str(path), framework="pt") as h:
        ids, rows = h.get_tensor("ids").tolist(), h.get_tensor("rows")
        saved = json.loads(h.metadata()["heads"])
    assert saved == {k: int(v) for k, v in head_ids.items()}, (saved, head_ids)
    assert ids == [head_ids[x] for x in HEADS]
    return rows


# ---------------------------------------------------------------------------
# Gated DeltaNet varlen injection (device aware for layer placement)
# ---------------------------------------------------------------------------
PACK_CTX = {"cu_seqlens": None, "seq_idx": None, "cache": {}}


@contextlib.contextmanager
def packed_context(cu_seqlens=None, seq_idx=None):
    """Active for forward AND backward (checkpoint recompute reads it)."""
    prev = dict(PACK_CTX)
    PACK_CTX.update(cu_seqlens=cu_seqlens, seq_idx=seq_idx, cache={})
    try:
        yield
    finally:
        PACK_CTX.clear()
        PACK_CTX.update(prev)


def _ctx_on(key, device):
    t = PACK_CTX[key]
    if t is None or t.device == device:
        return t
    cache = PACK_CTX["cache"]
    if (key, device) not in cache:
        cache[(key, device)] = t.to(device)
    return cache[(key, device)]


def _first_device(args, kwargs):
    for x in list(args) + list(kwargs.values()):
        if torch.is_tensor(x):
            return x.device
    raise RuntimeError("no tensor argument")


def _wrap_chunk_rule(orig):
    @functools.wraps(orig)
    def wrapped(*args, **kwargs):
        if PACK_CTX["cu_seqlens"] is not None:
            if kwargs.get("initial_state") is not None:
                raise RuntimeError("packed training expects initial_state=None")
            kwargs["cu_seqlens"] = _ctx_on("cu_seqlens", _first_device(args, kwargs))
        return orig(*args, **kwargs)
    wrapped._packed_wrapper = True
    return wrapped


def _wrap_causal_conv(orig):
    @functools.wraps(orig)
    def wrapped(*args, **kwargs):
        if PACK_CTX["seq_idx"] is None:
            return orig(*args, **kwargs)
        args = list(args)
        x = args.pop(0) if args else kwargs.pop("x")
        if x.stride(1) != 1:                       # seq_idx needs channel-last (B, D, T)
            x = x.transpose(1, 2).contiguous().transpose(1, 2)
        seq_idx = _ctx_on("seq_idx", x.device)
        if len(args) >= 3:                         # (weight, bias, seq_idx, ...)
            args[2] = seq_idx
        else:
            kwargs["seq_idx"] = seq_idx
        return orig(x, *args, **kwargs)
    wrapped._packed_wrapper = True
    return wrapped


def patch_linear_attention(model, log=print):
    patched = 0
    for name, m in model.named_modules():
        if not (hasattr(m, "chunk_gated_delta_rule") and hasattr(m, "causal_conv1d_fn")):
            continue
        if getattr(m.chunk_gated_delta_rule, "_packed_wrapper", False):
            continue
        if m.causal_conv1d_fn is None:
            raise RuntimeError(f"{name}: causal_conv1d unavailable; cannot isolate segments")
        mod = getattr(m.chunk_gated_delta_rule, "__module__", "") or ""
        if not mod.startswith("fla"):
            raise RuntimeError(f"{name}: chunk_gated_delta_rule from '{mod}', need FLA (cu_seqlens)")
        if patched == 0:
            log(f"Linear-attn {name} ({type(m).__name__}): conv={m.causal_conv1d_fn.__module__}, delta={mod}")
        m.chunk_gated_delta_rule = _wrap_chunk_rule(m.chunk_gated_delta_rule)
        m.causal_conv1d_fn = _wrap_causal_conv(m.causal_conv1d_fn)
        patched += 1
    if patched == 0:
        raise RuntimeError("No Gated DeltaNet modules found to patch")
    return patched


# ---------------------------------------------------------------------------
# MoE experts: sort-by-expert implementations replacing the per-expert where/sync loop
# ---------------------------------------------------------------------------
def _flatten_route(hidden_states, top_k_index):
    k = top_k_index.shape[-1]
    flat = top_k_index.reshape(-1)
    order = torch.argsort(flat, stable=True)
    return flat, order, order // k


def _combine(y_sorted, order, top_k_weights, shape, dtype):
    """Deterministic: un-permute with unique indices, then sum the k slots in fp32.
    (One index_add_ over all slots would race k atomics per token in bf16 -> run-to-run and
    batch-composition dependent rounding, which MoE routing amplifies.)"""
    k = top_k_weights.shape[-1]
    y = (y_sorted * top_k_weights.reshape(-1)[order].unsqueeze(-1)).to(dtype)   # same rounding as eager
    slots = y.new_empty(y.shape).index_copy(0, order, y)
    return torch.sum(slots.view(-1, k, shape[-1]), dim=1, dtype=torch.float32).to(dtype).reshape(shape)


def experts_forward_loop(self, hidden_states, top_k_index, top_k_weights):
    """Tokens sorted by expert, one host sync per layer, contiguous slices per expert."""
    shape = hidden_states.shape
    x2 = hidden_states.reshape(-1, shape[-1])
    flat, order, token_of = _flatten_route(x2, top_k_index)
    counts = torch.bincount(flat, minlength=self.num_experts).tolist()
    xs = x2[token_of]
    outs, start = [], 0
    for e, n in enumerate(counts):
        if n and e < self.num_experts:
            gate, up = F.linear(xs[start:start + n], self.gate_up_proj[e]).chunk(2, dim=-1)
            outs.append(F.linear(self.act_fn(gate) * up, self.down_proj[e]))
        elif n:                                   # index == num_experts: masked slot, contributes nothing
            outs.append(xs.new_zeros(n, shape[-1]))
        start += n
    return _combine(torch.cat(outs), order, top_k_weights, shape, x2.dtype)


_GROUP_ALIGN = 16


def _grouped_ffn(gate_up_proj, down_proj, act_fn, xs, sorted_e, counts):
    """Rows xs sorted by expert (sorted_e), counts per expert -> expert FFN outputs in the same row order."""
    n_exp, n = counts.numel(), xs.shape[0]
    # every group gets >= 16 rows (unused experts multiply zero rows): no zero-size groups reach the kernel
    padded = ((counts + _GROUP_ALIGN - 1) // _GROUP_ALIGN * _GROUP_ALIGN).clamp_min(_GROUP_ALIGN)
    rows = -(-(n + n_exp * _GROUP_ALIGN) // _GROUP_ALIGN) * _GROUP_ALIGN          # static upper bound
    ends = torch.cumsum(padded, 0)
    starts = ends - padded
    offs = ends.clone()
    offs[-1] = rows                                  # slack rows (zeros) belong to the last group
    rank = torch.arange(n, device=xs.device) - (torch.cumsum(counts, 0) - counts)[sorted_e]
    pos = starts[sorted_e] + rank
    xbuf = xs.new_zeros(rows, xs.shape[-1]).index_copy(0, pos, xs)
    offs = offs.to(torch.int32)
    gate, up = torch._grouped_mm(xbuf, gate_up_proj.transpose(-2, -1), offs=offs).chunk(2, dim=-1)
    y = torch._grouped_mm(act_fn(gate) * up, down_proj.transpose(-2, -1), offs=offs)
    return y.index_select(0, pos)


def experts_forward_grouped(self, hidden_states, top_k_index, top_k_weights):
    """torch._grouped_mm over expert groups padded to 16 rows; no host sync."""
    shape = hidden_states.shape
    x2 = hidden_states.reshape(-1, shape[-1])
    flat, order, token_of = _flatten_route(x2, top_k_index)
    counts = torch.bincount(flat, minlength=self.num_experts)[:self.num_experts]
    y = _grouped_ffn(self.gate_up_proj, self.down_proj, self.act_fn, x2[token_of], flat[order], counts)
    return _combine(y, order, top_k_weights, shape, x2.dtype)


def _all_to_all(x, out_splits, in_splits, group):
    """Autograd-aware variable-size all_to_all."""
    try:
        from torch.distributed._functional_collectives import all_to_all_single_autograd
        y = all_to_all_single_autograd(x, out_splits, in_splits, group)
        return y.wait() if hasattr(y, "wait") else y
    except ImportError:
        from torch.distributed.nn.functional import all_to_all_single
        out = x.new_empty((sum(out_splits),) + tuple(x.shape[1:]))
        return all_to_all_single(out, x, out_splits, in_splits, group=group)


FP8_BLOCK = 128
FP8_MAX = 448.0                                    # float8_e4m3fn


def fp8_block_quant(w, block=FP8_BLOCK):
    """(L, R, C) bf16 -> float8_e4m3fn values + fp32 scales per (block x block) tile."""
    n, r, c = w.shape
    assert r % block == 0 and c % block == 0, w.shape
    tiles = w.float().view(n, r // block, block, c // block, block)
    scale = tiles.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12) / FP8_MAX
    q = (tiles / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(n, r, c)
    return q, scale.view(n, r // block, c // block)


def fp8_block_dequant(q, scale, dtype=torch.bfloat16, block=FP8_BLOCK):
    n, r, c = q.shape
    tiles = q.view(n, r // block, block, c // block, block).float() * scale.view(n, r // block, 1, c // block, 1)
    return tiles.view(n, r, c).to(dtype)


def expert_weights(m):
    """(gate_up, down) as bf16 for the grouped kernel; dequantised per call when stored in FP8."""
    if hasattr(m, "gate_up_q"):
        return fp8_block_dequant(m.gate_up_q, m.gate_up_s), fp8_block_dequant(m.down_q, m.down_s)
    return m.gate_up_proj, m.down_proj


@torch.no_grad()
def quantize_experts_fp8(model, log=print):
    """Store every (local) routed-expert weight as FP8 + block scales; frees the bf16 copies."""
    experts = [m for m in model.modules()
               if isinstance(getattr(m, "gate_up_proj", None), torch.Tensor) and m.gate_up_proj.dim() == 3]
    assert experts, "no fused experts to quantize"
    before = torch.cuda.memory_allocated(experts[0].gate_up_proj.device)
    worst = 0.0
    for i, m in enumerate(experts):
        for name, qname in (("gate_up_proj", "gate_up"), ("down_proj", "down")):
            w = getattr(m, name)
            q, s = fp8_block_quant(w)
            if i == 0:
                err = ((fp8_block_dequant(q, s).float() - w.float()).norm() / w.float().norm()).item()
                worst = max(worst, err)
            delattr(m, name)
            m.register_buffer(f"{qname}_q", q, persistent=False)
            m.register_buffer(f"{qname}_s", s, persistent=False)
            del w
        torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated(experts[0].gate_up_q.device)
    log(f"FP8 experts: {len(experts)} layers quantized ({FP8_BLOCK}x{FP8_BLOCK} blocks, e4m3); "
        f"layer-0 relative error {worst:.2e}; memory {before / 2**30:.1f} -> {after / 2**30:.1f} GiB")


def experts_forward_ep(self, hidden_states, top_k_index, top_k_weights):
    """Expert parallelism: this rank holds experts [rank*L, (rank+1)*L). Tokens go to the rank
    owning their expert (all_to_all), run there with grouped_mm, come back, are combined locally."""
    import torch.distributed as dist
    shape = hidden_states.shape
    x2 = hidden_states.reshape(-1, shape[-1])
    n_rank, n_local, group = self.ep_size, self.n_local, self.ep_group
    flat, order, token_of = _flatten_route(x2, top_k_index)          # sorted by global expert id
    counts = torch.bincount(flat, minlength=n_rank * n_local)[:n_rank * n_local]
    recv = torch.empty_like(counts)
    dist.all_to_all_single(recv, counts, group=group)                  # recv[s*L + j]: rows from rank s for my expert j
    send_split = counts.view(n_rank, n_local).sum(1)
    recv_by = recv.view(n_rank, n_local)
    splits = torch.stack([send_split, recv_by.sum(1)]).tolist()       # one host sync per layer
    xr = _all_to_all(x2[token_of], splits[1], splits[0], group)
    local_e = torch.repeat_interleave(torch.arange(n_local, device=x2.device).repeat(n_rank), recv_by.reshape(-1))
    perm = torch.argsort(local_e, stable=True)                         # group received rows by local expert
    gate_up, down = expert_weights(self)
    y = _grouped_ffn(gate_up, down, self.act_fn, xr.index_select(0, perm), local_e[perm], recv_by.sum(0))
    y = y.new_empty(y.shape).index_copy(0, perm, y)                    # back to arrival order
    ys = _all_to_all(y, splits[0], splits[1], group)                   # back to this rank, expert-sorted
    return _combine(ys, order, top_k_weights, shape, x2.dtype)


def install_moe_ep(model, group, rank, size, log=print):
    experts = _find_experts(model)
    assert experts, "EP needs a MoE model"
    cls = type(experts[0])
    if not hasattr(cls, "_hf_forward"):
        cls._hf_forward = cls.forward
    for m in experts:
        n_local = m.gate_up_proj.shape[0]
        assert n_local * size == m.num_experts, (n_local, size, m.num_experts)
        m.ep_group, m.ep_rank, m.ep_size, m.n_local = group, rank, size, n_local
    cls.forward = experts_forward_ep
    log(f"MoE experts: expert parallel over {size} ranks, {experts[0].n_local} local experts per rank")


MOE_IMPLS = {"loop": experts_forward_loop, "grouped": experts_forward_grouped}


def _find_experts(model):
    found = [m for m in model.modules()
             if isinstance(getattr(m, "gate_up_proj", None), torch.Tensor) and m.gate_up_proj.dim() == 3]
    for m in found:
        missing = [a for a in ("down_proj", "num_experts", "act_fn") if not hasattr(m, a)]
        if missing:
            raise RuntimeError(f"{type(m).__name__} lacks {missing}; cannot replace its forward safely")
    return found


def install_moe_impl(model, impl="auto", tokens=16384, log=print):
    """Replace the experts forward of a MoE model. 'auto' checks every candidate against
    the original (outputs and input gradients) on a real layer and keeps the fastest that passes."""
    import inspect
    import time
    experts = _find_experts(model)
    if not experts:
        log("MoE: no fused experts module found (dense model); nothing to patch")
        return None
    cls = type(experts[0])
    assert all(type(m) is cls for m in experts)
    if not hasattr(cls, "_hf_forward"):
        cls._hf_forward = cls.forward
    original = cls._hf_forward
    assert len(inspect.signature(original).parameters) == 4, inspect.signature(original)
    if impl == "eager":
        cls.forward = original
        log(f"MoE experts: {cls.__name__}.forward -> eager (original)")
        return "eager"
    if impl != "auto":
        cls.forward = MOE_IMPLS[impl]
        log(f"MoE experts: {cls.__name__}.forward -> {impl} (forced)")
        return impl
    m = experts[0]
    dev, dtype = m.gate_up_proj.device, m.gate_up_proj.dtype
    hidden, n_exp = m.gate_up_proj.shape[-1], m.num_experts
    k = getattr(model.config, "num_experts_per_tok", 8)
    g = torch.Generator(device=dev).manual_seed(0)
    x0 = (torch.randn(tokens, hidden, device=dev, generator=g) * 0.5).to(dtype)
    w = torch.softmax(torch.randn(tokens, k, device=dev, generator=g), -1)
    gy = torch.randn(tokens, hidden, device=dev, generator=g).to(dtype)

    def routing(mode):
        scores = torch.rand(tokens, n_exp, device=dev, generator=g)
        if mode == "skewed":                   # like a pad run: few experts used, many empty, one hot set
            scores[:, 64:] = -1.0
            scores[tokens // 2:] = scores[tokens // 2]
        return scores.topk(k, dim=-1).indices

    routes = {"uniform": routing("uniform"), "skewed": routing("skewed")}
    subsets = (slice(0, 512), slice(tokens // 2 - 256, tokens // 2 + 256), slice(tokens - 300, tokens))

    def row_err(a, b):
        return ((a.float() - b.float()).norm(dim=-1) / b.float().norm(dim=-1).clamp_min(1e-3)).max().item()

    @torch.no_grad()
    def forward_errors(fn):
        """(max error vs eager, max self-inconsistency: rerun and subset-vs-full; must be ~0)."""
        vs_ref = self_err = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for idx in routes.values():
                ref = original(m, x0, idx, w)
                full = fn(m, x0, idx, w)
                vs_ref = max(vs_ref, row_err(full, ref))
                self_err = max(self_err, row_err(fn(m, x0, idx, w), full))
                for sub in subsets:
                    self_err = max(self_err, row_err(fn(m, x0[sub], idx[sub], w[sub]), full[sub]))
        return vs_ref, self_err

    def timed(fn, reps):
        best = None
        for _ in range(reps):
            x = x0.clone().requires_grad_(True)
            torch.cuda.synchronize(dev)
            t = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = fn(m, x, routes["uniform"], w)
            y.backward(gy)
            torch.cuda.synchronize(dev)
            best = time.perf_counter() - t if best is None else min(best, time.perf_counter() - t)
        return x.grad.float(), best

    ref_g, t_ref = timed(original, 2)
    chosen, chosen_t = "eager", t_ref
    _, eager_self = forward_errors(original)
    log(f"MoE experts check ({cls.__name__}, {tokens} tokens, top{k}; uniform + skewed routing): "
        f"eager {t_ref * 1e3:.1f} ms fwd+bwd, eager self-inconsistency {eager_self:.1e}")
    for name in ("loop", "grouped"):
        try:
            vs_ref, self_err = forward_errors(MOE_IMPLS[name])
            gr, t = timed(MOE_IMPLS[name], 3)
            ge = row_err(gr, ref_g)
            ok = vs_ref < 5e-2 and ge < 5e-2 and self_err <= max(eager_self, 1e-6)
            log(f"  {name:8s} {t * 1e3:8.1f} ms  vs eager out={vs_ref:.2e} grad={ge:.2e}  "
                f"self-inconsistency={self_err:.1e}  {'OK' if ok else 'REJECTED'}")
            if ok and t < chosen_t:
                chosen, chosen_t = name, t
        except Exception as e:      # e.g. _grouped_mm unavailable / no backward in this torch
            log(f"  {name:8s} unavailable: {type(e).__name__}: {str(e)[:160]}")
    if chosen != "eager":
        cls.forward = MOE_IMPLS[chosen]
    log(f"MoE experts: using {chosen} ({t_ref / chosen_t:.1f}x vs eager on this layer)")
    del x0, w, gy, routes, ref_g
    torch.cuda.empty_cache()
    return chosen


# ---------------------------------------------------------------------------
# Packing (self-contained; same schema as the 4B 01_prepare_data.py packs)
# ---------------------------------------------------------------------------
def seg_of(prompt, target, src):
    import numpy as np
    ids = np.asarray(list(prompt) + list(target), dtype=np.int32)
    mask = np.r_[np.zeros(len(prompt), np.uint8), np.ones(len(target), np.uint8)]
    return {"ids": ids, "mask": mask, "srcs": [src]}


def best_fit_bins(lengths, cap):
    """Best-fit decreasing: each item goes to the fullest bin that still has room."""
    import bisect
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    bins, free = [], []                     # free: sorted (remaining, bin_id)
    for i in order:
        n = lengths[i]
        assert n <= cap, f"segment of {n} tokens exceeds pack length {cap}"
        k = bisect.bisect_left(free, (n, -1))
        if k < len(free):
            rem, b = free.pop(k)
            bins[b].append(i)
            bisect.insort(free, (rem - n, b))
        else:
            bins.append([i])
            bisect.insort(free, (cap - n, len(bins) - 1))
    return bins


def make_packs(segs, cap, seed):
    import random as _random
    import numpy as np
    bins = best_fit_bins([len(s["ids"]) for s in segs], cap)
    _random.Random(seed).shuffle(bins)
    packs = []
    for b in bins:
        group = [segs[i] for i in b]
        lens = [len(s["ids"]) for s in group]
        packs.append({
            "input_ids": torch.from_numpy(np.concatenate([s["ids"] for s in group]).astype(np.int32)),
            "loss_mask": torch.from_numpy(np.concatenate([s["mask"] for s in group]).astype(np.uint8)),
            "position_ids": torch.from_numpy(np.concatenate([np.arange(n) for n in lens]).astype(np.int32)),
            "cu_seqlens": torch.from_numpy(np.r_[0, np.cumsum(lens)].astype(np.int32)),
            "sources": [s["srcs"][0] for s in group],
        })
    return packs


# ---------------------------------------------------------------------------
# Packs -> one fixed-length varlen row with per-token loss weights
# ---------------------------------------------------------------------------
def pack_segments(pack, pack_len, weight_fn):
    cuts = pack["cu_seqlens"].tolist()
    ids, mask = pack["input_ids"].long(), pack["loss_mask"].bool()
    assert cuts[0] == 0 and cuts[-1] == ids.numel() <= pack_len
    segs = []
    for name, a, b in zip(pack["sources"], cuts[:-1], cuts[1:]):
        m = mask[a:b]
        segs.append((ids[a:b], m, weight_fn(name, ids[a:b], m)))
    return segs


def build_batch(segments, total_len, pad_id):
    lengths = [int(x.numel()) for x, _, _ in segments]
    used = sum(lengths)
    assert used <= total_len, f"{used} > {total_len}"
    ids = torch.full((total_len,), pad_id, dtype=torch.long)
    mask = torch.zeros(total_len, dtype=torch.bool)
    wt = torch.zeros(total_len, dtype=torch.float32)
    off = 0
    for x, m, w in segments:
        n = x.numel()
        ids[off:off + n], mask[off:off + n], wt[off:off + n] = x, m, w
        off += n
    seg_lens = lengths + ([total_len - used] if used < total_len else [])
    seg_lens_t = torch.tensor(seg_lens, dtype=torch.long)
    cu = torch.zeros(len(seg_lens) + 1, dtype=torch.long)
    cu[1:] = torch.cumsum(seg_lens_t, 0)
    seg_id = torch.repeat_interleave(torch.arange(len(seg_lens)), seg_lens_t)
    position_ids = torch.arange(total_len) - cu[:-1][seg_id]
    is_start = torch.zeros(total_len, dtype=torch.bool)
    is_start[cu[:-1]] = True
    tgt_pos = (mask & ~is_start).nonzero(as_tuple=True)[0]      # never a segment's first token
    return {
        "input_ids": ids.unsqueeze(0),
        "position_ids": position_ids.unsqueeze(0),
        "cu_seqlens": cu,
        "cu_seqlens_i32": cu.to(torch.int32),
        "seq_idx": seg_id.to(torch.int32).unsqueeze(0),
        "hidden_index": tgt_pos - 1,
        "targets": ids[tgt_pos],
        "weights": wt[tgt_pos],
        "max_len": max(seg_lens),
        "real_tokens": used,
        "n_sup": int(tgt_pos.numel()),
        "w_sup": float(wt[tgt_pos].sum()),
        "n_segments": len(lengths),
        "seg_offsets": cu[:len(lengths)].tolist(),
    }


TENSOR_KEYS = ("input_ids", "position_ids", "cu_seqlens", "cu_seqlens_i32", "seq_idx",
               "hidden_index", "targets", "weights")


def to_device(batch, device):
    out = dict(batch)
    for k in TENSOR_KEYS:
        out[k] = batch[k].pin_memory().to(device, non_blocking=True)
    return out
