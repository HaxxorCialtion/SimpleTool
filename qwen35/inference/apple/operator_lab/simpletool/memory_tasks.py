"""Auditable single-head memory tasks for SimpleLove.

The main conversation remains the four-head path in :mod:`simpletool.decode`.
Memory work is deliberately a separate single-head workload because selection
is latency sensitive while write/compress can use a longer decode budget.  This
module does not write a database and does not turn model output into facts.
The host must validate evidence and event status before persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import time
from typing import Any, Callable, Mapping, Sequence

from .decode import batched_generate, prompt_ids


MEMORY_TASKS = ("memory-select", "memory-write", "memory-compress")


@dataclass(frozen=True)
class MemoryTaskResult:
    task: str
    raw_text: str
    parsed: Any
    valid: bool
    errors: tuple[str, ...]
    stats: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "raw_text": self.raw_text,
            "parsed": self.parsed,
            "valid": self.valid,
            "errors": list(self.errors),
            "stats": dict(self.stats),
        }


def _task_system(task: str) -> str:
    if task == "memory-select":
        return (
            "你是 SimpleLove 的 memory-select 阶段。只做候选记忆筛选，"
            "不回答用户，不解释，不生成 speech、expression、action 或 command。"
            "最终只输出 JSON 数组；数组元素只能是输入中已有的 memory_id，"
            "按相关性从高到低排列，没有相关记忆输出 []。"
        )
    if task == "memory-write":
        return (
            "你是 SimpleLove 的 session 结束记忆候选提取阶段。只依据输入中明确可见的"
            "对话和事件，不添加猜测。计划、提议和准备必须保持 proposed，不能写成 completed。"
            "最终只输出 JSON 数组；每项必须包含 memory_id、content、event_status、"
            "evidence_ids、confidence。没有值得保留的候选输出 []。"
        )
    if task == "memory-compress":
        return (
            "你是 SimpleLove 的 memory-compress 阶段。只合并、去重和修正输入记忆，"
            "不得添加输入不存在的事实。必须保留来源、日期语义、事件状态和不确定性。"
            "最终只输出 JSON 数组；每项必须包含 memory_id、content、source_ids、"
            "event_status、uncertainty。没有可压缩内容输出 []。"
        )
    raise ValueError(f"unknown memory task: {task}")


def _request_text(task: str, request: Mapping[str, Any]) -> str:
    # Keep the outer protocol readable to the current SFT model. The backend
    # still receives structured data; only the rendered prompt is task-shaped.
    if task == "memory-select":
        candidates = request.get("candidates", [])
        rows = "\n".join(json.dumps(item, ensure_ascii=False, sort_keys=True) for item in candidates)
        return (
            "任务：memory-select。只输出 JSON 数组，数组元素只能是候选 memory_id，"
            "按相关性从高到低排列；没有相关记忆输出 []。\n"
            f"当前用户消息：{request.get('current_user_message', '')}\n"
            f"候选记忆：\n{rows}\n输出："
        )
    if task == "memory-write":
        turns = json.dumps(request.get("turns", []), ensure_ascii=False, sort_keys=True)
        return (
            "任务：memory-write。只从对话明确内容提取长期有价值记忆。计划不可写成已完成，"
            "必须使用 proposed。只输出 JSON 数组，每项字段为 memory_id、content、"
            "event_status、evidence_ids、confidence；没有值得记忆输出 []。\n"
            f"session 结束原因：{request.get('end_reason', 'explicit')}\n"
            f"本 session 对话：{turns}\n输出："
        )
    memories = json.dumps(request.get("memories", []), ensure_ascii=False, sort_keys=True)
    return (
        "任务：memory-compress。合并重复记忆，不新增事实。保留来源和不确定性。"
        "只输出 JSON 数组，每项字段为 memory_id、content、source_ids、event_status、"
        f"uncertainty。输入记忆：\n{memories}\n输出："
    )


def _extract_json(text: str) -> tuple[Any, tuple[str, ...]]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].lstrip()
    try:
        return json.loads(candidate), ()
    except json.JSONDecodeError:
        # Models sometimes prepend one short sentence. Recover only a complete
        # top-level JSON value; never silently repair malformed fields.
        starts = [i for i in (candidate.find("["), candidate.find("{")) if i >= 0]
        for start in sorted(starts):
            try:
                return json.loads(candidate[start:]), ("recovered_json_suffix",)
            except json.JSONDecodeError:
                continue
        return None, ("invalid_json",)


def _validate(task: str, value: Any, request: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    if not isinstance(value, list):
        return False, ("result_must_be_array",)
    errors: list[str] = []
    if task == "memory-select":
        known = {
            item.get("memory_id")
            for item in request.get("candidates", [])
            if isinstance(item, Mapping) and isinstance(item.get("memory_id"), str)
        }
        if any(not isinstance(x, str) for x in value):
            errors.append("selected_ids_must_be_strings")
        if any(x not in known for x in value if isinstance(x, str)):
            errors.append("selected_id_not_in_candidates")
        if len(value) != len(set(value)):
            errors.append("duplicate_selected_id")
        return not errors, tuple(errors)

    required = {
        "memory-write": ("memory_id", "content", "event_status", "evidence_ids", "confidence"),
        "memory-compress": ("memory_id", "content", "source_ids", "event_status", "uncertainty"),
    }[task]
    known_sources = {
        source
        for item in request.get("memories", [])
        if isinstance(item, Mapping)
        for source in [item.get("memory_id"), *(item.get("evidence_ids", []) or [])]
        if isinstance(source, str)
    }
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            errors.append(f"item_{index}_must_be_object")
            continue
        missing = [key for key in required if key not in item]
        errors.extend(f"item_{index}_missing_{key}" for key in missing)
        if task == "memory-write" and item.get("event_status") not in {"proposed", "completed", "failed", "cancelled", "reported"}:
            errors.append(f"item_{index}_invalid_event_status")
        if task == "memory-compress":
            sources = item.get("source_ids")
            if not isinstance(sources, list) or any(source not in known_sources for source in sources):
                errors.append(f"item_{index}_invalid_source_ids")
    return not errors, tuple(errors)


class MemoryTaskEngine:
    """Run a memory task through one shared model with one active decode row."""

    def __init__(self, model: Any, tokenizer: Any, *, compute_label: str = "unknown"):
        self.model = model
        self.tokenizer = tokenizer
        self.compute_label = compute_label

    def run(self, task: str, request: Mapping[str, Any], *, max_tokens: int | None = None) -> MemoryTaskResult:
        if task not in MEMORY_TASKS:
            raise ValueError(f"unknown memory task: {task}")
        if not isinstance(request, Mapping):
            raise TypeError("request must be an object")
        if max_tokens is None:
            max_tokens = {"memory-select": 64, "memory-write": 192, "memory-compress": 256}[task]
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        messages = [
            {"role": "system", "content": _task_system(task)},
            {"role": "user", "content": _request_text(task, request)},
        ]
        ids = prompt_ids(self.tokenizer, messages)
        started = time.perf_counter()
        generated = batched_generate(self.model, self.tokenizer, ids, max_tokens=max_tokens, heads=("speech",))
        elapsed = time.perf_counter() - started
        raw = generated["channels"]["speech"]["text"]
        parsed, parse_errors = _extract_json(raw)
        valid, validation_errors = _validate(task, parsed, request)
        stats = dict(generated.get("stats", {}))
        stats.update({"task_wall_s": elapsed, "task_head": "speech", "compute": self.compute_label})
        return MemoryTaskResult(task, raw, parsed, valid and not parse_errors, tuple(parse_errors + validation_errors), stats)


@dataclass
class SessionMemoryCoordinator:
    """Collect turns and run memory-write only when a session ends.

    ``record_turn`` is intentionally model-free. ``end`` is the only method
    that invokes memory-write. The host should audit/bind evidence before
    calling ``compress``; this class never persists or promotes model output.
    """

    engine: MemoryTaskEngine
    idle_seconds: float = 300.0
    clock: Callable[[], float] = time.monotonic
    turns: list[Mapping[str, Any]] = field(default_factory=list)
    last_activity: float | None = None
    ended: bool = False
    last_write: MemoryTaskResult | None = None

    def record_turn(self, turn: Mapping[str, Any]) -> None:
        if self.ended:
            raise RuntimeError("session already ended")
        self.turns.append(dict(turn))
        self.last_activity = self.clock()

    def idle_due(self, now: float | None = None) -> bool:
        if self.ended or not self.turns or self.last_activity is None:
            return False
        current = self.clock() if now is None else now
        return current - self.last_activity >= self.idle_seconds

    def end(self, reason: str = "explicit") -> MemoryTaskResult | None:
        if self.ended:
            return self.last_write
        self.ended = True
        if not self.turns:
            return None
        self.last_write = self.engine.run("memory-write", {"end_reason": reason, "turns": list(self.turns)})
        return self.last_write

    def compress_after_audit(self, audited_memories: Sequence[Mapping[str, Any]]) -> MemoryTaskResult:
        if not self.ended:
            raise RuntimeError("end session and audit memory-write candidates first")
        return self.engine.run("memory-compress", {"memories": list(audited_memories)})
