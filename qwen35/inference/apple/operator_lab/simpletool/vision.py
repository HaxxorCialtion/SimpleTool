import hashlib
import json
import struct
import subprocess
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class PositionRope(nn.Module):
    def __init__(self, base, dimensions, theta, sections):
        super().__init__()
        self.base = base
        self.dimensions = dimensions
        self.theta = theta
        self.sections = sections
        self.positions = None
        self.decode_delta = 0

    def __call__(self, x, offset=0):
        if self.positions is None:
            return self.base(x, offset=offset + self.decode_delta)
        d = self.dimensions
        indices = np.arange(d//2)
        axes = np.zeros(d//2, dtype=np.int32)
        for axis in [1, 2]:
            axes[(indices % 3 == axis) & (indices < self.sections[axis]*3)] = axis
        positions = mx.take(self.positions, mx.array(axes), axis=0).T.astype(mx.float32)
        freqs = positions * (self.theta ** (-mx.arange(0, d, 2, dtype=mx.float32) / d))
        cos = mx.cos(freqs).astype(x.dtype)[None, None]
        sin = mx.sin(freqs).astype(x.dtype)[None, None]
        first, second = x[..., :d//2], x[..., d//2:d]
        return mx.concatenate([first*cos-second*sin, second*cos+first*sin, x[..., d:]], axis=-1)


def install_position_rope(model):
    cfg = model.language_model.args
    for layer in model.layers:
        if not layer.is_linear and not isinstance(layer.self_attn.rope, PositionRope):
            layer.self_attn.rope = PositionRope(layer.self_attn.rope,
                int(cfg.head_dim * cfg.partial_rotary_factor), cfg.rope_theta,
                cfg.rope_parameters.get("mrope_section", [11, 11, 10]))


def set_positions(model, positions=None, decode_delta=0):
    for layer in model.layers:
        if not layer.is_linear and isinstance(layer.self_attn.rope, PositionRope):
            layer.self_attn.rope.positions = positions
            layer.self_attn.rope.decode_delta = decode_delta


def read_bridge(path):
    data = memoryview(Path(path).read_bytes())
    if data[:4] != b"STVI": raise ValueError("invalid vision bridge")
    dim, chunks = struct.unpack_from("<II", data, 4)
    offset = 12
    result = []
    for _ in range(chunks):
        kind, n, npos = struct.unpack_from("<IIi", data, offset)
        offset += 12
        length = n*4 if kind == 0 else n*dim*4
        values = np.frombuffer(data[offset:offset+length], dtype="<i4" if kind == 0 else "<f4").copy()
        if kind != 0: values = values.reshape(n, dim)
        offset += length
        positions = np.frombuffer(data[offset:offset+n*12], dtype="<i4").reshape(n, 3).copy()
        offset += n*12
        result.append((kind, values, positions, npos))
    next_position, = struct.unpack_from("<i", data, offset)
    if offset + 4 != len(data): raise ValueError("unexpected trailing bridge bytes")
    return dim, result, next_position


def image_prefix(model, tokenizer, messages, image_path, gguf, mmproj, exporter, cache_dir):
    from PIL import Image
    prepare_start = time.perf_counter()
    messages = [dict(m) for m in messages]
    if messages[-1]["role"] != "user" or not isinstance(messages[-1]["content"], str):
        raise ValueError("image bridge requires latest user text message")
    if not any("<__media__>" in m.get("content", "") for m in messages):
        messages[-1]["content"] = "<__media__>\n" + messages[-1]["content"]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                      add_generation_prompt=True, enable_thinking=False)
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((1024, 1024))
    rgb = image.tobytes()
    identity = json.dumps({"format_version": 2, "image": hashlib.sha256(rgb).hexdigest(),
          "size": image.size, "gguf": str(Path(gguf).resolve()), "gguf_mtime": Path(gguf).stat().st_mtime_ns,
          "mmproj": str(Path(mmproj).resolve()), "mmproj_mtime": Path(mmproj).stat().st_mtime_ns,
          "exporter_mtime": Path(exporter).stat().st_mtime_ns}, sort_keys=True).encode()
    output = Path(cache_dir) / (hashlib.sha256(identity).hexdigest()+".bin")
    output.parent.mkdir(parents=True, exist_ok=True)
    hit = output.is_file()
    start = time.perf_counter()
    if not hit:
        with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
            tmp = Path(tmp)
            (tmp/"prompt.txt").write_text("<__media__>")
            (tmp/"rgb.raw").write_bytes(rgb)
            with output.with_suffix(".log").open("w") as log:
                subprocess.run([str(Path(exporter).resolve()), str(Path(gguf).resolve()),
                    str(Path(mmproj).resolve()), str(tmp/"prompt.txt"), str(tmp/"rgb.raw"),
                    str(image.width), str(image.height), str(tmp/"output.bin"), "1024"],
                    stdout=log, stderr=log, check=True)
            (tmp/"output.bin").replace(output)
    seconds = time.perf_counter() - start
    dim, chunks, _ = read_bridge(output)
    if dim != model.language_model.args.hidden_size:
        raise ValueError("projector hidden dimension does not match language model")
    if len(chunks) != 3 or [c[0] for c in chunks] != [0, 1, 0]:
        raise ValueError("unexpected projector framing; expected one image chunk")
    start_id, end_id = tokenizer.convert_tokens_to_ids("<|vision_start|>"), tokenizer.convert_tokens_to_ids("<|vision_end|>")
    if chunks[0][1].tolist() != [start_id] or chunks[2][1].tolist() != [end_id]:
        raise ValueError("unexpected vision start/end framing")
    before, after = prompt.split("<__media__>")
    prefix = tokenizer.encode(before, add_special_tokens=False) + [start_id]
    suffix = [end_id] + tokenizer.encode(after, add_special_tokens=False)
    _, image_values, image_positions, image_npos = chunks[1]
    n_image = len(image_values)
    ids = prefix + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * n_image + suffix
    dtype = (model.model.embed_tokens.scales.dtype if hasattr(model.model.embed_tokens, "scales")
             else model.model.embed_tokens.weight.dtype)
    embeddings = [model.model.embed_tokens(mx.array(prefix))[None],
                  mx.array(image_values, dtype=dtype)[None],
                  model.model.embed_tokens(mx.array(suffix))[None]]
    offset = len(prefix)
    image_positions = image_positions - image_positions[0, 0] + offset
    suffix_offset = offset + image_npos
    positions = [mx.broadcast_to(mx.arange(offset)[None], (3, offset)),
                 mx.array(image_positions.T),
                 mx.broadcast_to(mx.arange(suffix_offset, suffix_offset+len(suffix))[None], (3, len(suffix)))]
    next_position = suffix_offset + len(suffix)
    install_position_rope(model)
    return ids, mx.concatenate(embeddings, axis=1), mx.concatenate(positions, axis=1), next_position, {
        "vision_bridge_cache_hit": hit, "vision_bridge_s": seconds,
        "vision_preparation_s": time.perf_counter()-prepare_start,
        "vision_tokens": n_image,
        "vision_encoding_calls": 0 if hit else 1, "bridge_file": str(output.resolve())}
