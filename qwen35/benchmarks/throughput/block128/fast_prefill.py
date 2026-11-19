"""Bounded, shape-specialized public-prefill CUDA graphs for serial exact-fork.

Only the model forward is replayed. The ordinary scheduler, input preparation,
terminal cache publication and native branch decoder remain authoritative.
Each replay stages current tokens, positions, slot maps and attention metadata.
"""
from collections import deque, Counter
import copy
import dataclasses
import hashlib
import json
import time

import torch
from vllm.forward_context import get_forward_context, override_forward_context


def signature(value, memo=None):
    if memo is None:
        memo = {}
    if torch.is_tensor(value):
        key = ("tensor", str(value.device), str(value.dtype), tuple(value.shape), tuple(value.stride()))
        return key + ((tuple(value.reshape(-1).tolist()),) if value.device.type == "cpu" else ())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    identity = id(value)
    if identity in memo:
        return ("ref", memo[identity])
    memo[identity] = len(memo)
    if isinstance(value, dict):
        return ("dict", tuple((k, signature(v, memo)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(signature(v, memo) for v in value))
    if dataclasses.is_dataclass(value):
        return (type(value).__qualname__, tuple((f.name, signature(getattr(value, f.name), memo))
                for f in dataclasses.fields(value)))
    raise TypeError(f"Unsupported graph metadata: {type(value)}")


def clone_tree(value, memo, bindings, path=()):
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if torch.is_tensor(value):
        target = value.clone()
        memo[identity] = target
        bindings.append((path, target))
        return target
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        target = {}
        memo[identity] = target
        target.update({k: clone_tree(v, memo, bindings, path + (k,)) for k, v in value.items()})
        return target
    if isinstance(value, (tuple, list)):
        target = type(value)(clone_tree(v, memo, bindings, path + (i,)) for i, v in enumerate(value))
        memo[identity] = target
        return target
    if dataclasses.is_dataclass(value):
        target = copy.copy(value)
        memo[identity] = target
        for f in dataclasses.fields(value):
            object.__setattr__(target, f.name,
                               clone_tree(getattr(value, f.name), memo, bindings, path + (f.name,)))
        return target
    raise TypeError(f"Unsupported graph metadata: {type(value)}")


def at_path(root, path):
    value = root
    for part in path:
        value = value[part] if isinstance(value, (tuple, list, dict)) else getattr(value, part)
    return value


class BoundedCalls(deque):
    def __init__(self, maxlen):
        super().__init__(maxlen=maxlen)
        self.counts = Counter()
    def append(self, row):
        self.counts[row['mode']] += 1
        return super().append(row)


class PublicPrefillGraphs:
    def __init__(self, runner, runtime, max_graphs=24):
        self.runner = runner
        self.runtime = runtime
        self.original_forward = runner.model.forward
        self.graphs = {}
        self.max_graphs = max_graphs
        self.calls = BoundedCalls(maxlen=64)
        self.current_prime = False
        self.enabled = True
        self.verify = False
        self.allow_capture = False
        original_execute = runner.execute_model

        def execute(scheduler_output, *args, **kwargs):
            ids = list(scheduler_output.num_scheduled_tokens)
            self.current_prime = bool(ids and all(x in runtime.PRIMES for x in ids))
            try:
                return original_execute(scheduler_output, *args, **kwargs)
            finally:
                self.current_prime = False
        runner.execute_model = execute
        runner.model.forward = self.forward

    def forward(self, *args, **kwargs):
        if not self.current_prime or not self.enabled:
            return self.original_forward(*args, **kwargs)
        host_start = time.perf_counter()
        ctx = get_forward_context()
        assert ctx.dp_metadata is None and ctx.ubatch_slices is None
        root = dict(args=args, kwargs=kwargs, metadata=ctx.attn_metadata,
                    slots=ctx.slot_mapping, padding=ctx.is_padding)
        key = signature(root)
        signature_ms = (time.perf_counter() - host_start) * 1000
        if key not in self.graphs:
            if not self.allow_capture:
                self.calls.append(dict(mode="eager-unwarmed-shape-fallback", signature_ms=signature_ms))
                return self.original_forward(*args, **kwargs)
            if len(self.graphs) >= self.max_graphs:
                self.calls.append(dict(mode="eager-capacity-fallback"))
                return self.original_forward(*args, **kwargs)
            bindings = []
            stable = clone_tree(root, {}, bindings)
            stable_ctx = dataclasses.replace(ctx, attn_metadata=stable['metadata'],
                                             slot_mapping=stable['slots'], is_padding=stable['padding'])

            def raw():
                with override_forward_context(stable_ctx):
                    return self.original_forward(*stable['args'], **stable['kwargs'])

            # Warmup writes are identical for cold states. For cache-hit states
            # the source slot can alias the mutable destination, so snapshot all
            # Mamba slots touched by this forward before any extra execution.
            snapshots = self.snapshot_mamba(stable['metadata'])
            def restore():
                for tensor, indices, data in snapshots:
                    tensor.index_copy_(0, indices, data)
            stream = torch.cuda.current_stream()
            side = torch.cuda.Stream()
            side.wait_stream(stream)
            t = time.perf_counter()
            from vllm.v1.worker.workspace import current_workspace_manager
            manager = current_workspace_manager()
            pointers = [x.data_ptr() if x is not None else None for x in manager._current_workspaces]
            was_locked = manager.is_locked()
            manager.unlock()
            try:
                with torch.cuda.stream(side):
                    for _ in range(2):
                        restore()
                        raw()
            finally:
                if was_locked:
                    manager.lock()
            assert pointers == [x.data_ptr() if x is not None else None
                                for x in manager._current_workspaces], 'Existing decoder workspace moved'
            stream.wait_stream(side)
            torch.cuda.synchronize()
            restore()
            graph = torch.cuda.CUDAGraph()
            # Use the same TP/PP/DP capture context as native vLLM graphs.
            # Plain torch.cuda.graph omits custom-all-reduce IPC buffer
            # registration. Every rank must finish that registration before
            # the first replay of the newly captured public forward.
            from vllm.distributed.parallel_state import graph_capture
            with graph_capture(device=self.runner.device) as distributed_capture:
                with torch.cuda.graph(graph, stream=distributed_capture.stream):
                    output = raw()
            torch.cuda.synchronize()
            restore()
            self.graphs[key] = dict(graph=graph, output=output, bindings=bindings,
                                    stable=stable, ctx=stable_ctx)
            capture_ms = (time.perf_counter() - t) * 1000
            mode = "capture-replay"
        else:
            mode, capture_ms = "replay", 0.
        stage_start = time.perf_counter()
        entry = self.graphs[key]
        gpu_targets, gpu_sources = [], []
        for path, target in entry['bindings']:
            source = at_path(root, path)
            assert source.shape == target.shape and source.dtype == target.dtype
            if target.device.type == 'cuda':
                gpu_targets.append(target)
                gpu_sources.append(source)
            else:
                target.copy_(source)
        torch._foreach_copy_(gpu_targets, gpu_sources, non_blocking=True)
        stage_host_ms = (time.perf_counter() - stage_start) * 1000
        initial = self.snapshot_mamba(entry['stable']['metadata']) if self.verify else None
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        entry['graph'].replay()
        end.record()
        verification = None
        if initial is not None:
            actual_hidden = entry['output'].clone()
            expected = self.snapshot_mamba(entry['stable']['metadata'])
            for tensor, indices, data in initial:
                tensor.index_copy_(0, indices, data)
            reference = self.original_forward(*args, **kwargs)
            assert torch.equal(reference, actual_hidden), 'Graph hidden states differ from raw'
            size = 0
            for tensor, indices, data in expected:
                assert torch.equal(tensor.index_select(0, indices), data), 'Terminal Mamba state differs'
                size += data.numel() * data.element_size()
            verification = dict(hidden_exact=True, terminal_mamba_exact=True,
                                state_tensors=len(expected), state_bytes=size)
        # The ordinary prime completion path will synchronize its own ready
        # event; defer reading elapsed time until the request result is built.
        self.calls.append(dict(mode=mode, capture_ms=capture_ms, start=start, end=end,
                               bindings=len(entry['bindings']), graphs=len(self.graphs), signature_ms=signature_ms, stage_host_ms=stage_host_ms, verification=verification))
        return entry['output']

    def snapshot_mamba(self, metadata):
        # Initial capture must preserve the real input state even when native
        # convolution updates it in-place. Snapshot the active slot, not the pool.
        snapshots = []
        layers = self.runner.vllm_config.compilation_config.static_forward_context
        for name, layer in layers.items():
            if name not in metadata or not hasattr(layer, 'kv_cache'):
                continue
            meta = metadata[name]
            indices = getattr(meta, 'prefill_state_indices', None)
            if indices is None:
                continue
            caches = layer.kv_cache
            if not isinstance(caches, (tuple, list)):
                continue
            for tensor in caches:
                if torch.is_tensor(tensor):
                    ids = indices.to(dtype=torch.long).reshape(-1).unique()
                    snapshots.append((tensor, ids, tensor.index_select(0, ids).clone()))
        return snapshots

    def receipt(self):
        rows = []
        for row in self.calls:
            result = {k: v for k, v in row.items() if k not in ('start', 'end')}
            if 'end' in row:
                assert row['end'].query()
                result['model_ms'] = row['start'].elapsed_time(row['end'])
            rows.append(result)
        self.calls.clear()
        return dict(graphs=len(self.graphs), calls=rows, cumulative_modes=dict(self.calls.counts), max_graphs=self.max_graphs,
                    capture_protocol='native-distributed-graph-context-v2')
