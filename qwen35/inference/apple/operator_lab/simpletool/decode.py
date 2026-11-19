import time

import mlx.core as mx

from .attention import install_fork_attention
from .cache import (cache_arrays, clone_single, filter_batch, fork_batch,
                    memory_accounting)

HEADS = ("speech", "expression", "action", "command")


def prompt_ids(tokenizer, messages):
    return tokenizer.apply_chat_template(messages, tokenize=True,
                                         add_generation_prompt=True, enable_thinking=False)


def prefill(model, ids, chunk_size=256, embeddings=None, positions=None,
            next_position=None, prefix_cache=None, start_offset=0):
    from .vision import install_position_rope, set_positions
    if positions is not None:
        install_position_rope(model)
    set_positions(model)
    caches = clone_single(prefix_cache) if prefix_cache is not None else model.make_cache()
    start = time.perf_counter()
    for offset in range(start_offset, len(ids), chunk_size):
        batch = mx.array(ids[offset:offset + chunk_size])[None]
        emb = embeddings[:, offset:offset + chunk_size] if embeddings is not None else None
        if positions is not None:
            set_positions(model, positions[:, offset:offset + chunk_size])
        hidden = model.model(batch, cache=caches, input_embeddings=emb)
        mx.eval(hidden, *cache_arrays(caches))
    set_positions(model, decode_delta=(next_position-len(ids)) if next_position is not None else 0)
    return caches, time.perf_counter() - start


def stopped(tokens, eos, close, tokenizer=None, head=None):
    if tokens[-1] == eos:
        return "eos"
    if len(tokens) >= len(close) and tokens[-len(close):] == close:
        return "channel_close"
    # A BPE token may contain both preceding text and part of the closing tag.
    if tokenizer is not None and f"</{head}>" in tokenizer.decode(tokens[-16:], skip_special_tokens=False):
        return "channel_close"
    return None


def render(tokenizer, raw, head, reason):
    close = tokenizer.encode(f"</{head}>", add_special_tokens=False)
    visible = raw[:-1] if reason == "eos" else raw
    text = tokenizer.decode(visible, skip_special_tokens=False).split(f"</{head}>", 1)[0].strip()
    return {"text": text,
            "token_ids": raw, "stop_reason": reason,
            "closed": reason == "channel_close"}


def batched_generate(model, tokenizer, ids, max_tokens=128, heads=HEADS,
                     prefix_cache=None, embeddings=None, positions=None, next_position=None,
                     ignore_stops=False, compiled_step=None, suppress_tokens=(),decode_chunk_size=1,
                     head_token_limits=None):
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    limits = None
    if head_token_limits is not None:
        if set(head_token_limits) != set(heads):
            raise ValueError('head_token_limits must specify exactly the active heads')
        limits = [head_token_limits[h] for h in heads]
        if any(type(v) is not int or not 1 <= v <= max_tokens for v in limits):
            raise ValueError('head token limits must be integers within 1..max_tokens')
    if compiled_step and tuple(suppress_tokens) != compiled_step.suppress_tokens:
        raise ValueError("compiled sampler and requested token suppression must match")
    if decode_chunk_size not in (1,2,4,8) or (decode_chunk_size>1 and compiled_step is None):
        raise ValueError('decode chunks 2/4/8 require compiled inference')
    install_fork_attention(model)
    if prefix_cache is None:
        source, pp_seconds = prefill(model, ids, embeddings=embeddings, positions=positions,
                                     next_position=next_position)
    else:
        source, pp_seconds = prefix_cache, 0.0
    marker_ids = [tokenizer.encode(f"<{h}>", add_special_tokens=False) for h in heads]
    if len({len(m) for m in marker_ids}) != 1:
        raise ValueError("unequal marker lengths require length-aware batching")
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos is None:
        raise ValueError("chat EOS not present in tokenizer")
    closes = [tokenizer.encode(f"</{h}>", add_special_tokens=False) for h in heads]
    start = time.perf_counter()
    caches = fork_batch(source, len(heads)) if len(heads) > 1 or compiled_step else clone_single(source)
    if compiled_step:
        states = compiled_step.initialize(caches, len(heads), max_tokens+len(marker_ids[0]))
    used = 0
    mx.eval(*cache_arrays(caches))
    fork_seconds = time.perf_counter() - start
    start = time.perf_counter()
    for j in range(len(marker_ids[0])):
        if j == len(marker_ids[0]) - 1:
            marker_seconds = time.perf_counter() - start
            first_token_start = time.perf_counter()
        # One forward call, with B rows, for each marker token position.
        current = mx.array([[m[j]] for m in marker_ids])
        if compiled_step:
            prediction, states = compiled_step.forward(current, states, mx.array(used,dtype=mx.int32))
            mx.eval(prediction, states)
            used += 1
        else:
            logits = model(current, cache=caches)
            mx.eval(logits, *cache_arrays(caches))
    first_token_forward_seconds = time.perf_counter() - first_token_start
    raw = [[] for _ in heads]
    reasons = ["max_tokens"] * len(heads)
    active = list(range(len(heads)))
    batch_trace = []
    start = time.perf_counter()
    discarded_ahead=0
    if decode_chunk_size>1:
        generated=0
        while generated<max_tokens:
            count=min(decode_chunk_size,max_tokens-generated)
            block,states=compiled_step.advance_pending(prediction,states,mx.array(used,mx.int32),count)
            mx.eval(block,states)
            used+=count-1
            chosen=block.tolist()
            batch_trace.extend([len(active)]*count)
            keep=[];surviving=[]
            for row,(idx,sequence) in enumerate(zip(active,chosen)):
                reason=None
                for column,tok in enumerate(sequence):
                    raw[idx].append(tok)
                    reason=None if ignore_stops else stopped(raw[idx],eos,closes[idx],tokenizer,heads[idx])
                    if reason is None and limits is not None and len(raw[idx]) >= limits[idx]:
                        reason='external_token_limit'
                    if reason:
                        reasons[idx]=reason
                        discarded_ahead+=count-column-1
                        break
                if reason is None:keep.append(row);surviving.append(idx)
            generated+=count
            if not keep or generated==max_tokens:break
            if len(keep)!=len(active):states=compiled_step.filter(states,keep)
            active=surviving
            current=mx.array([[chosen[row][-1]] for row in keep])
            prediction,states=compiled_step.forward(current,states,mx.array(used,mx.int32))
            mx.eval(prediction,states)
            used+=1
    for step in range(max_tokens if decode_chunk_size==1 else 0):
        batch_trace.append(len(active))
        if suppress_tokens and not compiled_step:
            logits[..., list(suppress_tokens)] = -mx.inf
        next_ids = prediction if compiled_step else mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_ids)
        chosen = next_ids.tolist()
        keep, surviving = [], []
        for row, (idx, tok) in enumerate(zip(active, chosen)):
            raw[idx].append(tok)
            reason = None if ignore_stops else stopped(raw[idx], eos, closes[idx], tokenizer, heads[idx])
            if reason is None and limits is not None and len(raw[idx]) >= limits[idx]:
                reason = 'external_token_limit'
            if reason:
                reasons[idx] = reason
            else:
                keep.append(row)
                surviving.append(idx)
        if not keep or step + 1 == max_tokens:
            break
        if len(keep) != len(active):
            if compiled_step: states = compiled_step.filter(states, keep)
            else: filter_batch(caches, keep)
        active = surviving
        current = mx.array([[chosen[row]] for row in keep])
        if compiled_step:
            prediction, states = compiled_step.forward(current,states,mx.array(used,dtype=mx.int32))
            mx.eval(prediction,states)
            used += 1
        else:
            logits = model(current, cache=caches)
            mx.eval(logits, *cache_arrays(caches))
    decode_seconds = time.perf_counter() - start + first_token_forward_seconds
    n = sum(map(len, raw))
    if compiled_step:
        for c,state in zip(caches,states):
            if len(state)==4:
                c.tail.keys,c.tail.values=state[2],state[3]
                c.tail.offset=used
            else:c.cache=list(state)
    return {"channels": {h: render(tokenizer, r, h, why)
                         for h, r, why in zip(heads, raw, reasons)},
            "stats": {"prefill_s": pp_seconds, "fork_s": fork_seconds,
                      "marker_s": marker_seconds, "decode_s": decode_seconds,
                      "total_s": pp_seconds + fork_seconds + marker_seconds + decode_seconds,
                      "generated_tokens_including_stop": n,
                      "aggregate_decode_tps": n / max(decode_seconds, 1e-9),
                      "active_batch_by_step": batch_trace,
                      "marker_token_ids": dict(zip(heads, marker_ids)),
                      "compiled_decode": compiled_step is not None,
                      "decode_chunk_size":decode_chunk_size,
                      "discarded_ahead_tokens_after_stop":discarded_ahead,
                      "suppressed_token_ids": list(suppress_tokens),
                      **memory_accounting(caches)}}


def serial_generate(model, tokenizer, ids, max_tokens=128, heads=HEADS, prefix_cache=None):
    if prefix_cache is None:
        source, pp_seconds = prefill(model, ids)
    else:
        source, pp_seconds = prefix_cache, 0.0
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    channels = {}
    start = time.perf_counter()
    for h in heads:
        caches = clone_single(source)
        marker = tokenizer.encode(f"<{h}>", add_special_tokens=False)
        close = tokenizer.encode(f"</{h}>", add_special_tokens=False)
        raw, reason = [], "max_tokens"
        for tok in marker:
            logits = model(mx.array([[tok]]), cache=caches)
            mx.eval(logits, *cache_arrays(caches))
        for step in range(max_tokens):
            tok = int(mx.argmax(logits[0, -1]).item())
            raw.append(tok)
            end = stopped(raw, eos, close, tokenizer, h)
            if end:
                reason = end
                break
            if step + 1 < max_tokens:
                logits = model(mx.array([[tok]]), cache=caches)
                mx.eval(logits, *cache_arrays(caches))
        channels[h] = render(tokenizer, raw, h, reason)
    return {"channels": channels, "stats": {"prefill_s": pp_seconds,
            "serial_marker_decode_s": time.perf_counter() - start}}
