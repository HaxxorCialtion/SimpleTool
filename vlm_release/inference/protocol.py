"""Backend-independent generative SimpleTool protocol; no model or executor imports.

Pure string schemas use literal unquoted text (the legacy SimpleTool contract).
All other schemas, including unions, use JSON. ``<|null|>`` means omitted,
whereas JSON ``null`` is an explicit value. Schema property insertion order
defines the six argument slots. Branch prefixes never include the selected tool.
"""
from dataclasses import dataclass
import json
import math
from typing import Mapping

import jsonschema

MAX_ARGUMENT_HEADS = 6
HEADS = ("function",) + tuple(f"arg{i}" for i in range(1, MAX_ARGUMENT_HEADS + 1))
SPECIAL = tuple(f"<{slash}{head}>" for head in ("content",) + HEADS
                for slash in ("", "/")) + ("<|null|>",)
NULL = "<|null|>"
MODES = ("direct", "adaptive")


class ProtocolError(ValueError):
    """Malformed generated output, incompatible schema, or invalid target."""


@dataclass(frozen=True)
class Branch:
    name: str
    prefix: str
    target: str
    weight: float


def _mode(mode):
    if mode not in MODES:
        raise ProtocolError(f"unknown mode: {mode}")


def _literal(text, *, allow_null=False):
    if not isinstance(text, str):
        raise ProtocolError("protocol values must be strings")
    if allow_null and text == NULL:
        return text
    if any(token in text for token in SPECIAL):
        raise ProtocolError("nested/reserved protocol token")
    return text


def _closed(raw, head, *, allow_null=False):
    end = f"</{head}>"
    if not isinstance(raw, str) or not raw.endswith(end):
        raise ProtocolError(f"missing or wrong closing tag: {head}")
    return _literal(raw[:-len(end)], allow_null=allow_null)


def content_prefix(prompt: str, mode: str):
    """Return content request prefix, or None for forced direct decoding."""
    _mode(mode)
    return prompt + "<content>" if mode == "adaptive" else None


def head_prefixes(prompt: str, mode: str, content_output=None, heads=None):
    """Accept only generated content, including its closing tag; no tool target."""
    _mode(mode)
    if mode == "direct":
        if content_output is not None:
            raise ProtocolError("direct mode cannot consume content")
        shared = prompt
    else:
        _closed(content_output, "content")
        shared = prompt + "<content>" + content_output
    names = HEADS if heads is None else tuple(heads)
    return {head: shared + f"<{head}>" for head in names}


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ProtocolError("nonfinite JSON number")
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError("JSON object keys must be strings")
            _finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite(item)


def _reject_constant(value):
    raise ProtocolError(f"nonfinite JSON constant: {value}")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _no_refs(schema):
    # Network/local reference resolution is deliberately not an output parser's job.
    if isinstance(schema, dict):
        if "$ref" in schema or "nullable" in schema:
            raise ProtocolError("normalize nullable and inline $ref before using protocol")
        for value in schema.values():
            _no_refs(value)
    elif isinstance(schema, list):
        for value in schema:
            _no_refs(value)


def tool_schemas(tools):
    """Validate dynamic OpenAI-style tools; support zero through six slots."""
    if not isinstance(tools, (list, tuple)) or not tools:
        raise ProtocolError("tools must be a nonempty sequence")
    result = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            raise ProtocolError("expected function tool")
        fn = tool.get("function", {})
        name = fn.get("name")
        if not isinstance(name, str) or not name or name in result:
            raise ProtocolError("missing/duplicate tool name")
        _literal(name)
        schema = fn.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ProtocolError("parameters must have object type")
        _no_refs(schema)
        try:
            jsonschema.Draft7Validator.check_schema(schema)
        except jsonschema.SchemaError as exc:
            raise ProtocolError(f"invalid schema: {name}") from exc
        props = schema.get("properties", {})
        if len(props) > 6 or not set(schema.get("required", [])) <= set(props):
            raise ProtocolError("too many slots or required property without slot")
        result[name] = schema
    return result


def active_heads(tools):
    """Return function plus the maximum argument slots in this request."""
    schemas = tool_schemas(tools)
    width = max((len(schema.get("properties", {})) for schema in schemas.values()), default=0)
    return ("function",) + tuple(f"arg{i}" for i in range(1, width + 1))


def _validate(arguments, schema):
    if not isinstance(arguments, dict):
        raise ProtocolError("arguments must be an object")
    if not set(arguments) <= set(schema.get("properties", {})):
        raise ProtocolError("argument has no declared slot")
    _finite(arguments)
    try:
        jsonschema.Draft7Validator(schema).validate(arguments)
    except jsonschema.ValidationError as exc:
        raise ProtocolError(f"schema validation: {exc.message}") from exc


def strict_decode(predictions: Mapping[str, str], tools):
    """Parse only the active generated continuations without repair or fallback."""
    heads = active_heads(tools)
    if set(predictions) != set(heads):
        raise ProtocolError(f"expected heads: {', '.join(heads)}")
    values = {head: _closed(predictions[head], head, allow_null=head != "function")
              for head in heads}
    schemas = tool_schemas(tools)
    name = values["function"]
    if name not in schemas:
        raise ProtocolError("unknown tool")
    schema = schemas[name]
    props = list(schema.get("properties", {}).items())
    arguments = {}
    for index in range(len(heads) - 1):
        raw = values[f"arg{index + 1}"]
        if index >= len(props):
            if raw != NULL:
                raise ProtocolError("non-null unused slot")
            continue
        key, spec = props[index]
        if raw == NULL:
            continue  # Required omissions fail schema validation, never defaulted.
        if isinstance(spec, dict) and spec.get("type") == "string":
            value = raw
        else:
            try:
                value = json.loads(raw, parse_constant=_reject_constant,
                                   object_pairs_hook=_unique_pairs)
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"invalid JSON argument: {key}") from exc
        arguments[key] = value
    _validate(arguments, schema)
    return name, arguments


def encode_call(function: str, arguments: dict, tools):
    """Return training bodies; closing tags are added by training_branches."""
    schemas = tool_schemas(tools)
    if function not in schemas:
        raise ProtocolError("unknown tool")
    schema = schemas[function]
    _validate(arguments, schema)
    values = dict.fromkeys(HEADS, NULL)
    values["function"] = function
    for index, (name, spec) in enumerate(schema.get("properties", {}).items(), 1):
        if name in arguments:
            value = arguments[name]
            raw = value if isinstance(spec, dict) and spec.get("type") == "string" else json.dumps(
                value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            values[f"arg{index}"] = _literal(raw)
    return values


def training_branches(row, mode: str):
    """Normalized branch weights sum to one; mask every prefix in the trainer.

    Ground-truth content is teacher-forced only in adaptive training. Function
    and argument labels never appear in another head's prefix.
    """
    _mode(mode)
    values = encode_call(row["function"], row["arguments"], row["tools"])
    if "head_values" in row and row["head_values"] != values:
        raise ProtocolError("head_values disagree with arguments/schema")
    branches = []
    active = active_heads(row["tools"])
    if mode == "adaptive":
        content = _literal(row.get("content", "")) + "</content>"
        branches.append(Branch("content", content_prefix(row["prompt"], mode), content,
                               1 / (len(active) + 1)))
        prefixes = head_prefixes(row["prompt"], mode, content, active)
        weight = 1 / (len(active) + 1)
    else:
        prefixes = head_prefixes(row["prompt"], mode, heads=active)
        weight = 1 / len(active)
    branches.extend(Branch(head, prefixes[head], values[head] + f"</{head}>", weight)
                    for head in active)
    return branches
