"""Native Gemma text models, structured packed attention, and factored expert LoRA."""
import contextlib
from dataclasses import dataclass
import json
import math
import re
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForCausalLM

HEADS = ('content', 'function', 'arg1', 'arg2', 'arg3', 'arg4', 'arg5', 'arg6')
SPECIAL_TOKENS = tuple(t for h in HEADS for t in (f'<{h}>', f'</{h}>')) + ('<|null|>',)
BACKEND = 'simpletool_gemma_structured'
ATTENTION_KERNELS = {}


@dataclass
class PackedBias:
    lengths: tuple
    window: int | None

    def dense(self, device, dtype):
        lengths = torch.tensor(self.lengths, device=device)
        ids = torch.repeat_interleave(torch.arange(len(self.lengths), device=device), lengths)
        pos = torch.arange(sum(self.lengths), device=device)
        distance = pos[:, None] - pos[None, :]
        allow = (ids[:, None] == ids[None, :]) & (distance >= 0)
        if self.window:
            allow &= distance < self.window
        return torch.zeros(allow.shape, dtype=dtype, device=device).masked_fill_(~allow, -torch.inf)[None, None]


def structured_attention(module, query, key, value, attention_mask, dropout=0., scaling=1., **kwargs):
    assert isinstance(attention_mask, PackedBias)
    assert query.shape[0] == 1 and query.shape[-2] == sum(attention_mask.lengths)
    if query.shape[1] != key.shape[1]:
        # xFormers' 4-D backward requires explicit GQA heads; do not claim 5-D GQA backward support.
        groups = query.shape[1] // key.shape[1]
        key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
    if query.is_cuda:
        import xformers.ops as xops
        from xformers.ops.fmha import flash, flash3, cutlass
        from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask
        bias = BlockDiagonalCausalMask.from_seqlens(list(attention_mask.lengths))
        if attention_mask.window:
            bias = bias.make_local_attention(attention_mask.window)
        # Require a real FlashAttention kernel for supported head dimensions.
        # Global Gemma heads at 512 exceed FA2/FA3's 256-dimensional limit.
        if max(query.shape[-1], value.shape[-1]) <= 256:
            use_fa3 = (flash3.FwOp.is_available() and flash3.BwOp.is_available()
                       and torch.cuda.get_device_capability(query.device) == (9, 0))
            implementation = flash3 if use_fa3 else flash
        else:
            implementation = cutlass
        operators = (implementation.FwOp, implementation.BwOp)
        assert all(op.is_available() for op in operators), 'required fused attention kernel unavailable'
        key_info = f'dim{query.shape[-1]}_window{attention_mask.window}'
        if key_info not in ATTENTION_KERNELS:
            ATTENTION_KERNELS[key_info] = {'forward':operators[0].NAME,'backward':operators[1].NAME}
            print(json.dumps({'event':'attention_kernel','shape':key_info,**ATTENTION_KERNELS[key_info]}),flush=True)
        result = xops.memory_efficient_attention(query.transpose(1, 2), key.transpose(1, 2),
                                                value.transpose(1, 2), attn_bias=bias,
                                                p=dropout, scale=scaling, op=operators)
        return result, None
    result = F.scaled_dot_product_attention(query, key, value,
                                            attn_mask=attention_mask.dense(query.device, query.dtype),
                                            dropout_p=dropout, scale=scaling)
    return result.transpose(1, 2).contiguous(), None


ALL_ATTENTION_FUNCTIONS.register(BACKEND, structured_attention)


def model_class(config):
    return Gemma4UnifiedForCausalLM if config.model_type == 'gemma4_unified_text' else Gemma4ForCausalLM


def make_meta(model_dir):
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True).text_config
    config.use_cache = False
    config._attn_implementation = 'sdpa'
    if config.enable_moe_block:
        config._experts_implementation = 'grouped_mm'
    with torch.device('meta'):
        model = model_class(config)(config).to(dtype=torch.bfloat16)
    return model


def source_schema(model_dir, model=None):
    model = model or make_meta(model_dir)
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items() if k != 'lm_head.weight'}
    found, unused = {}, []
    for shard in sorted(Path(model_dir).glob('model*.safetensors')):
        with safe_open(shard, framework='pt') as weights:
            for name in weights.keys():
                if not name.startswith('model.language_model.'):
                    continue
                key = 'model.' + name.removeprefix('model.language_model.')
                shape = tuple(weights.get_slice(name).get_shape())
                if key not in expected:
                    match = re.fullmatch(r'model.layers.(\d+).self_attn.(k_proj|v_proj|k_norm|v_norm).weight', key)
                    if match and model.get_decoder().layers[int(match[1])].self_attn.is_kv_shared_layer:
                        continue
                    unused.append(key)
                    continue
                assert expected[key] == shape, (key, expected[key], shape)
                assert key not in found
                found[key] = (str(shard), name)
    assert set(found) == set(expected), ('missing native text tensors', sorted(set(expected)-set(found)))
    # The three large models have no shared-KV layers and should have no surplus text tensors.
    assert not unused, ('unexpected native text tensors', unused)
    return found


def load_text(model_dir, device):
    model = make_meta(model_dir)
    schema = source_schema(model_dir, model)
    by_shard = {}
    for key, (shard, original) in schema.items():
        by_shard.setdefault(shard, []).append((key, original))
    for shard, keys in by_shard.items():
        with safe_open(shard, framework='pt') as weights:
            for key, original in keys:
                parent, _, name = key.rpartition('.')
                module = model.get_submodule(parent)
                value = weights.get_tensor(original).to(device=device, dtype=model.state_dict()[key].dtype)
                if name in module._parameters:
                    module._parameters[name] = nn.Parameter(value, requires_grad=False)
                else:
                    module._buffers[name] = value
    model.lm_head.weight = model.model.embed_tokens.weight
    # Rebuild meta-only nonpersistent buffers from the native constructors.
    for name, module in list(model.named_modules()):
        if hasattr(module, 'scalar_embed_scale'):
            module.embed_scale = torch.tensor(module.scalar_embed_scale, device=device, dtype=torch.bfloat16)
        if name.endswith('rotary_emb'):
            parent, _, leaf = name.rpartition('.')
            setattr(model.get_submodule(parent), leaf, type(module)(module.config, device=device))
    assert not [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    model.config._attn_implementation = BACKEND
    model.config._name_or_path = str(model_dir)
    return model


class HeadEmbedding(nn.Module):
    def __init__(self, base, tokenizer, ids):
        super().__init__()
        self.original_vocab_size = base.num_embeddings
        assert len(tokenizer) == self.original_vocab_size + len(ids)
        assert list(ids.values()) == list(range(self.original_vocab_size, len(tokenizer)))
        rows = torch.stack([base.weight[tokenizer.encode(h.strip('<>/|'), add_special_tokens=False)].float().mean(0) for h in ids])
        base.weight = nn.Parameter(torch.cat([base.weight.detach(), rows.to(base.weight.dtype)]), requires_grad=False)
        base.num_embeddings = len(tokenizer)
        self.base = base
        self.register_buffer('ids', torch.tensor(list(ids.values()), device=base.weight.device), persistent=False)
        self.delta = nn.Parameter(torch.zeros_like(rows, dtype=torch.float32))

    @property
    def weight(self):
        return self.base.weight

    def forward(self, ids):
        lookup = (ids - self.original_vocab_size + 1).clamp(min=0)
        table = torch.cat([self.delta.new_zeros(1, self.delta.shape[-1]), self.delta])
        return self.base(ids) + F.embedding(lookup, table).to(self.weight.dtype) * self.base.embed_scale.to(self.weight.dtype)

    def final_rows(self):
        return self.weight[self.ids].float() + self.delta


class HeadOutput(nn.Module):
    def __init__(self, base, embedding):
        super().__init__()
        self.base = base
        self.delta = embedding.delta
        self.register_buffer('ids', embedding.ids, persistent=False)
        base.weight = embedding.weight
        base.out_features = embedding.weight.shape[0]

    @property
    def weight(self):
        return self.base.weight

    def forward(self, hidden):
        result = self.base(hidden).float()
        with torch.autocast(hidden.device.type, enabled=False):
            extra = F.linear(hidden.float(), self.delta)
        return result.index_add(-1, self.ids, extra)


def grouped_linear(x, weight, offsets):
    if x.is_cuda:
        assert hasattr(F, 'grouped_mm'), 'PyTorch 2.11 grouped_mm is required'
        return F.grouped_mm(x, weight.to(x.dtype).transpose(-1, -2), offs=offsets)
    # Small CPU correctness reference; never silently use a Python expert loop on GPU.
    ends = offsets.tolist()
    starts = [0, *ends[:-1]]
    return torch.cat([F.linear(x[a:b], weight[i]) for i, (a, b) in enumerate(zip(starts, ends))])


class GroupedExpertLoRA(nn.Module):
    """Factored LoRA on every routed expert, without materializing full 3-D delta matrices."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        experts, twice_intermediate, hidden = base.gate_up_proj.shape
        intermediate = twice_intermediate // 2
        self.scale = alpha / rank
        self.num_experts = experts
        for stem, ins, outs in [('gate_up', hidden, twice_intermediate), ('down', intermediate, hidden)]:
            a = nn.Parameter(torch.empty(experts, rank, ins, device=base.gate_up_proj.device, dtype=torch.float32))
            b = nn.Parameter(torch.zeros(experts, outs, rank, device=a.device, dtype=torch.float32))
            for expert in a:
                nn.init.kaiming_uniform_(expert, a=math.sqrt(5))
            self.register_parameter(f'expert_lora_{stem}_A', a)
            self.register_parameter(f'expert_lora_{stem}_B', b)

    def linear(self, x, stem, base, offsets):
        a = getattr(self, f'expert_lora_{stem}_A')
        b = getattr(self, f'expert_lora_{stem}_B')
        return grouped_linear(x, base, offsets) + self.scale * grouped_linear(grouped_linear(x, a, offsets), b, offsets)

    def forward(self, hidden, top_k_index, top_k_weights):
        tokens, width = hidden.shape
        topk = top_k_index.shape[-1]
        expert, permutation = top_k_index.flatten().sort(stable=True)
        # Fixed output size avoids bincount's CUDA synchronization for max().
        offsets = torch.histc(expert.float(), bins=self.num_experts, min=0,
                              max=self.num_experts-1).cumsum(0).to(torch.int32)
        x = hidden[permutation // topk]
        gate, up = self.linear(x, 'gate_up', self.base.gate_up_proj, offsets).chunk(2, dim=-1)
        y = self.linear(self.base.act_fn(gate) * up, 'down', self.base.down_proj, offsets)
        y = y * top_k_weights.flatten()[permutation, None]
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(len(permutation), device=permutation.device)
        return y[inverse].view(tokens, topk, width).sum(1)


def attach_adapters(model, tokenizer, head_ids, rank=128, alpha=256, expert_rank=None, expert_alpha=None):
    from peft import LoraConfig, get_peft_model
    model.requires_grad_(False)
    special_ids = {t: tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}
    assert all(tokenizer.encode(t, add_special_tokens=False) == [i] for t,i in special_ids.items())
    head = HeadEmbedding(model.get_input_embeddings(), tokenizer, special_ids)
    decoder = model.get_decoder()
    if getattr(model.config, 'hidden_size_per_layer_input', 0):
        decoder.embed_tokens_per_layer = HeadEmbedding(decoder.embed_tokens_per_layer, tokenizer, special_ids)
        model.config.vocab_size_per_layer_input = len(tokenizer)
    model.set_input_embeddings(head)
    model.set_output_embeddings(HeadOutput(model.get_output_embeddings(), head))
    model.config.vocab_size = len(tokenizer)
    model.get_decoder().vocab_size = model.vocab_size = len(tokenizer)
    model = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0., bias='none',
                                            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    head.delta.requires_grad_(True)
    if getattr(model.config, 'hidden_size_per_layer_input', 0):
        model.get_decoder().embed_tokens_per_layer.delta.requires_grad_(True)
    if model.config.enable_moe_block:
        for layer in model.get_decoder().layers:
            layer.experts = GroupedExpertLoRA(layer.experts, expert_rank or rank, expert_alpha or alpha)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert 'lora_' in name or name.endswith('.delta'), name
            parameter.data = parameter.data.float()
    return model


def save_adapters(model, tokenizer, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, save_embedding_layers=False, safe_serialization=True)
    tokenizer.save_pretrained(path / 'tokenizer')
    head = model.get_input_embeddings()
    rows = {'ids':head.ids.cpu(), 'rows':head.final_rows().detach().cpu()}
    if getattr(model.config, 'hidden_size_per_layer_input', 0):
        rows['per_layer_rows'] = model.get_decoder().embed_tokens_per_layer.final_rows().detach().cpu()
    save_file(rows, path / 'head_tokens.safetensors',
              metadata={'semantics':'absolute unscaled input/output tied rows; optional per_layer_rows must also be restored', 'tokens':json.dumps(SPECIAL_TOKENS)})
    extra = {name: p.detach().cpu().contiguous() for name, p in model.named_parameters() if 'expert_lora_' in name}
    if extra:
        save_file(extra, path / 'grouped_expert_lora.safetensors', metadata={'format':'simpletool_grouped_expert_lora_v1'})
    (path / 'ADAPTER_FORMAT.json').write_text(json.dumps({'format':'simpletool_gemma_eighthead_v1',
        'requires_custom_loader':True, 'base_model':model.config._name_or_path,
        'lora_rank':model.peft_config['default'].r, 'lora_alpha':model.peft_config['default'].lora_alpha,
        'expert_rank': next((m.expert_lora_gate_up_A.shape[1] for m in model.modules() if isinstance(m,GroupedExpertLoRA)),None),
        'expert_alpha': next((m.scale*m.expert_lora_gate_up_A.shape[1] for m in model.modules() if isinstance(m,GroupedExpertLoRA)),None),
        'head_rows':'head_tokens.safetensors', 'expert_lora':bool(extra),
        'attention_backend':BACKEND}, indent=2)+'\n')


def load_adapters(model_dir, artifact, tokenizer, head_ids, device):
    artifact = Path(artifact)
    config = json.loads((artifact / 'ADAPTER_FORMAT.json').read_text())
    model = attach_adapters(load_text(model_dir, device), tokenizer, head_ids, config['lora_rank'], config['lora_alpha'],
                            config.get('expert_rank'), config.get('expert_alpha'))
    return restore_adapters(model, artifact, device)


def restore_adapters(model, artifact, device):
    from peft import set_peft_model_state_dict
    artifact = Path(artifact)
    config = json.loads((artifact / 'ADAPTER_FORMAT.json').read_text())
    result = set_peft_model_state_dict(model, load_file(artifact / 'adapter_model.safetensors', device=str(device)))
    assert not result.unexpected_keys
    head = model.get_input_embeddings()
    rows = load_file(artifact / 'head_tokens.safetensors', device=str(device))
    with torch.no_grad():
        assert torch.equal(rows['ids'],head.ids)
        head.delta.copy_(rows['rows'] - head.weight[head.ids].float())
        if getattr(model.config, 'hidden_size_per_layer_input', 0):
            ple = model.get_decoder().embed_tokens_per_layer
            ple.delta.copy_(rows['per_layer_rows'] - ple.weight[ple.ids].float())
        if config['expert_lora']:
            extra = load_file(artifact / 'grouped_expert_lora.safetensors', device=str(device))
            expected = {n: p for n, p in model.named_parameters() if 'expert_lora_' in n}
            assert set(extra) == set(expected)
            for name, parameter in expected.items():
                parameter.copy_(extra[name])
    return model
