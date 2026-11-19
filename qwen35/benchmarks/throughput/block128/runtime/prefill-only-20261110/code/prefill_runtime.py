"""Opt-in single-GPU, in-process, serial-group prefill-only/fork experiment.

No installed source changes. No guessed/padded token is generated for the prime.
The completed terminal state is cached only AFTER its CUDA completion event.
"""
import os,time

PRIMES = set()
GROUPS = {}
EVENTS = []
GPU_SPANS = []
ACTIVE = False
CURRENT_STEP = 0
_INSTALLED = False

def record(kind, **kw):
    if ACTIVE:
        EVENTS.append(dict(kind=kind, step=CURRENT_STEP, perf=time.perf_counter(), **kw))

def metadata(request):
    p = request.sampling_params
    return (p.extra_args or {}).get('simpletool_prefill') if p else None

def is_prime(request):
    m = metadata(request)
    return bool(m and m['role'] == 'prime')

def complete(scheduler, request):
    """Called inside native update_from_output after the worker completion ack."""
    m = metadata(request)
    assert m and m['role'] == 'prime'
    assert request.num_computed_tokens == request.num_prompt_tokens == m['N']
    assert request.num_in_flight_tokens == 0 and request.num_output_tokens == 0
    scheduler.kv_cache_manager.cache_blocks(request, request.num_prompt_tokens)
    GROUPS[m['group']]['ready'] = True
    record('public_state_published', request_id=request.request_id, N=m['N'], output_tokens=0)

def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    import torch
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request

    if os.environ.get('SL_PREFILL_SHARE_HASHES') == '1':
        original_hashes = Request.update_block_hashes
        def hashes(self):
            m = metadata(self)
            if m and m['role'] == 'branch' and not self.block_hashes:
                group = GROUPS[m['group']]
                n = m['N']
                assert self.prompt_token_ids[:n] == group['prefix']
                assert len(self.prompt_token_ids) == n + 1
                assert self._block_hasher is group['hasher']
                assert self.cache_salt == group['cache_salt']
                assert self.lora_request == group['lora_request']
                assert not self.mm_features and self.prompt_embeds is None
                # Unit-one hashing only. Hash objects are immutable; each
                # branch still owns its list and computes its private suffix.
                assert len(group['prefix_hashes']) == n
                self.block_hashes.extend(group['prefix_hashes'])
                record('public_hashes_reused', request_id=self.request_id, count=n)
            return original_hashes(self)
        Request.update_block_hashes = hashes

    original_add = Scheduler.add_request
    def add(self, request):
        m = metadata(request)
        if m:
            assert not self.scheduler_config.async_scheduling or os.environ.get('SL_PREFILL_ALLOW_ASYNC')=='1', 'Async mode requires explicit experiment opt-in'
            assert self.connector is None and not request.mm_features and not request.resumable
            n, group, role = m['N'], m['group'], m['role']
            assert isinstance(n, int) and n > 0 and role in ('prime', 'branch')
            if role == 'prime':
                assert request.max_tokens == 1 and self.num_spec_tokens == 0
                assert group not in GROUPS
                assert len(request.prompt_token_ids) == n
                PRIMES.add(request.request_id)
                GROUPS[group] = dict(prefix=list(request.prompt_token_ids), ready=False,
                                     prime_id=request.request_id, branch_ids=[],
                                     prefix_hashes=tuple(request.block_hashes),
                                     hasher=request._block_hasher,
                                     cache_salt=request.cache_salt,
                                     lora_request=request.lora_request)
            else:
                assert group in GROUPS
                assert len(request.prompt_token_ids) == n + 1
                assert request.prompt_token_ids[:n] == GROUPS[group]['prefix']
                GROUPS[group]['branch_ids'].append(request.request_id)
            record('request_registered', role=role, request_id=request.request_id, N=n, group=group)
        return original_add(self, request)
    Scheduler.add_request = add

    original_schedule = Scheduler.schedule
    def schedule(self, *args, **kwargs):
        import dataclasses
        result = original_schedule(self, *args, **kwargs)
        if ACTIVE and result.num_scheduled_tokens:
            ids = list(result.num_scheduled_tokens)
            layouts = []
            for manager in self.kv_cache_manager.coordinator.single_type_managers:
                layouts.append(dict(manager=type(manager).__name__,
                    group_id=manager.kv_cache_group_id, block_size=manager.block_size,
                    blocks={rid:[b.block_id if not b.is_null else None
                                 for b in manager.req_to_blocks.get(rid, [])] for rid in ids}))
            record('cache_layout', layouts=layouts,
                   copies=[dataclasses.asdict(c) if dataclasses.is_dataclass(c)
                           else c._asdict() if hasattr(c, '_asdict') else str(c)
                           for c in (result.kv_cache_block_copies or [])])
        return result
    Scheduler.schedule = schedule

    original_blocked = Scheduler._handle_blocked_waiting_request
    def blocked(self, request):
        m = metadata(request)
        if m and m['role'] == 'branch' and not GROUPS[m['group']]['ready']:
            record('branch_held', request_id=request.request_id)
            return False
        return original_blocked(self, request)
    Scheduler._handle_blocked_waiting_request = blocked

    original_split = Scheduler._mamba_block_aligned_split
    def split(self, request, num_new_tokens, *args, **kwargs):
        # No next-token logits are needed, so there is no N-1 checkpoint.
        if is_prime(request):
            return num_new_tokens
        return original_split(self, request, num_new_tokens, *args, **kwargs)
    Scheduler._mamba_block_aligned_split = split

    original_needs = MambaManager._needs_internal_checkpoint
    def needs(self, request_id, *args, **kwargs):
        if request_id in PRIMES:
            return False
        return original_needs(self, request_id, *args, **kwargs)
    MambaManager._needs_internal_checkpoint = needs

    # Cache the actual terminal state N, rather than the default logits boundary N-1.
    # Mamba's ordinary state slot is the final slot; no midpoint snapshot required.
    def cache_terminal(manager, request, num_tokens):
        n = request.num_prompt_tokens
        if (num_tokens != n or request.num_computed_tokens != n
                or request.num_in_flight_tokens != 0):
            return None
        blocks = manager.req_to_blocks[request.request_id]
        block_idx = (n - 1) // manager.block_size
        block = blocks[block_idx]
        assert not block.is_null
        if block.block_hash_num_tokens == n:
            return None
        result = manager.block_pool.cache_partial_block(
            request=request, block=block, num_tokens=n,
            kv_cache_group_id=manager.kv_cache_group_id,
            block_size=manager.block_size, replace_existing_hashes=True)
        record('terminal_block_cached', group_id=manager.kv_cache_group_id,
               manager=type(manager).__name__, block_id=block.block_id, N=n)
        return result

    original_full = FullAttentionManager._cache_partial_tail_block
    def full(self, request, num_tokens):
        if is_prime(request):
            return cache_terminal(self, request, num_tokens)
        return original_full(self, request, num_tokens)
    FullAttentionManager._cache_partial_tail_block = full
    original_mamba = MambaManager._cache_partial_tail_block
    def mamba(self, request, num_tokens, retention_interval):
        if is_prime(request):
            return cache_terminal(self, request, num_tokens)
        return original_mamba(self, request, num_tokens, retention_interval)
    MambaManager._cache_partial_tail_block = mamba

    original_execute = GPUModelRunner.execute_model
    def execute(self, scheduler_output, *args, **kwargs):
        ids = list(scheduler_output.num_scheduled_tokens)
        prime = bool(ids and all(rid in PRIMES for rid in ids))
        if prime:
            assert len(ids) == 1
            self._sl_prefill_begin = torch.cuda.Event(enable_timing=True)
            self._sl_prefill_begin.record(self.main_stream)
        record('runner_forward_submitted', tokens=dict(scheduler_output.num_scheduled_tokens), prime=prime)
        if ACTIVE and ids:
            begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            begin.record(self.main_stream)
            result = original_execute(self, scheduler_output, *args, **kwargs)
            end.record(self.main_stream)
            GPU_SPANS.append((CURRENT_STEP, begin, end, dict(scheduler_output.num_scheduled_tokens)))
            return result
        return original_execute(self, scheduler_output, *args, **kwargs)
    GPUModelRunner.execute_model = execute

    original_sample_tokens = GPUModelRunner.sample_tokens
    def sample_tokens(self, grammar_output):
        state = self.execute_model_state
        if state is None or not state.input_batch.req_ids or not all(
                rid in PRIMES for rid in state.input_batch.req_ids):
            record('ordinary_sampling_dispatch')
            return original_sample_tokens(self, grammar_output)
        assert self.is_last_pp_rank and self.speculator is None and self.pp_handler is None
        assert self.pcp_manager is None and self.aux_output_connector is None
        batch = state.input_batch
        self.execute_model_state = None
        forward_end = torch.cuda.Event(enable_timing=True)
        forward_end.record(self.main_stream)
        self.postprocess_num_computed_tokens(batch)
        self.model_state.postprocess_state(batch.idx_mapping, 0)
        connector_output = self.kv_connector.post_forward(state.finished_req_ids)
        ready = torch.cuda.Event(enable_timing=True)
        ready.record(self.main_stream)
        begin_wait = time.perf_counter()
        ready.synchronize()
        record('prefill_only_ready', request_ids=list(batch.req_ids), output_tokens=0,
               stream_span_ms=self._sl_prefill_begin.elapsed_time(forward_end),
               postprocess_stream_ms=forward_end.elapsed_time(ready),
               ready_wait_host_ms=(time.perf_counter()-begin_wait)*1000,
               logits_called=False, sampler_called=False)
        return ModelRunnerOutput(
            req_ids=list(batch.req_ids),
            req_id_to_index={rid: i for i, rid in enumerate(batch.req_ids)},
            sampled_token_ids=[[] for _ in batch.req_ids],
            kv_connector_output=connector_output,
            ec_connector_output=state.ec_connector_output,
            cudagraph_stats=state.cudagraph_stats)
    GPUModelRunner.sample_tokens = sample_tokens
    original_sample = GPUModelRunner.sample
    def actual_sample(self, hidden_states, input_batch, grammar_output):
        assert not any(rid in PRIMES for rid in input_batch.req_ids), 'Prime reached actual sampler path'
        record('actual_sample_path', request_ids=list(input_batch.req_ids))
        return original_sample(self, hidden_states, input_batch, grammar_output)
    GPUModelRunner.sample = actual_sample
    return dict(mode='prefill-only + native terminal cache + gated branches')
