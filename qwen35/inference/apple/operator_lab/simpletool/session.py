from pathlib import Path

from .decode import HEADS, batched_generate, prefill, prompt_ids


def assistant_history(channels):
    return "\n".join(f"<{h}>\n{channels[h]['text']}\n</{h}>" for h in HEADS)


class TextSession:
    """Keep only a canonical common prefix; never commit one branch as history."""

    def __init__(self, model, tokenizer, system=None, max_context=8192, compiled_step=None):
        self.model = model
        self.tokenizer = tokenizer
        self.messages = [{"role":"system","content":system}] if system is not None else []
        self.prefix_ids = []
        self.prefix_cache = None
        self.max_context = max_context
        self.compiled_step = compiled_step

    def prepare(self, candidate):
        return prompt_ids(self.tokenizer, candidate), None, None, None, {}

    def reply(self, text, max_tokens=128):
        candidate = self.messages + [{"role":"user","content":text}]
        ids, embeddings, positions, next_position, extra = self.prepare(candidate)
        if len(ids)+max_tokens > self.max_context:
            raise ValueError(f'context exceeds {self.max_context} tokens; start a new session or configure max_context')
        # The disabled-thinking block is absent in concatenated assistant history.
        # Retain the cache BEFORE that block to preserve a true prefix next round.
        assistant_start = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        start = next((i for i in range(len(ids)-len(assistant_start),-1,-1)
                      if ids[i:i+len(assistant_start)] == assistant_start), None)
        if start is None: raise ValueError("assistant generation prefix missing")
        canonical_end = start + len(assistant_start)
        reuse = (self.prefix_cache is not None and ids[:len(self.prefix_ids)] == self.prefix_ids)
        cache, canonical_s = prefill(self.model, ids[:canonical_end],
             prefix_cache=self.prefix_cache if reuse else None,
             start_offset=len(self.prefix_ids) if reuse else 0,
             embeddings=embeddings[:, :canonical_end] if embeddings is not None else None,
             positions=positions[:, :canonical_end] if positions is not None else None)
        full, pending_s = prefill(self.model, ids, prefix_cache=cache, start_offset=canonical_end,
                                 embeddings=embeddings,positions=positions,next_position=next_position)
        result = batched_generate(self.model,self.tokenizer,ids,max_tokens,prefix_cache=full,
                                  compiled_step=self.compiled_step)
        result["stats"].update({"prefill_s":canonical_s+pending_s, **extra,
             "reused_prefix_tokens":len(self.prefix_ids) if reuse else 0,
             "incremental_prefill_tokens":len(ids)-(len(self.prefix_ids) if reuse else 0)})
        result["stats"]["total_s"] += canonical_s+pending_s+extra.get("vision_preparation_s",0)
        self.prefix_ids, self.prefix_cache = ids[:canonical_end], cache
        self.messages = candidate + [{"role":"assistant", "content":assistant_history(result["channels"])}]
        return result


class ImageSession(TextSession):
    """One immutable screenshot per session, with incremental language prefill."""

    def __init__(self, model, tokenizer, image, gguf, mmproj, exporter,
                 cache_dir=Path("artifacts/vision_cache"), system=None, max_context=8192, compiled_step=None):
        super().__init__(model, tokenizer, system, max_context, compiled_step)
        self.vision_args = (image,gguf,mmproj,exporter,cache_dir)
        self.image_version = None

    def prepare(self, candidate):
        from .vision import image_prefix
        user = next(m for m in candidate if m["role"] == "user")
        if "<__media__>" not in user["content"]:
            user["content"] = "<__media__>\n"+user["content"]
        prepared = image_prefix(self.model,self.tokenizer,candidate,*self.vision_args)
        version = prepared[-1]["bridge_file"]
        if self.image_version is not None and version != self.image_version:
            raise ValueError("screenshot changed; start a new ImageSession to preserve history semantics")
        self.image_version = version
        return prepared
