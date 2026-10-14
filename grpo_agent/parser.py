from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

_FENCE = re.compile(r"^\s*```[a-zA-Z0-9]*\s*(.*?)\s*```\s*$", re.DOTALL)


def strip_code_fence(text: str) -> tuple[str, bool]:
    """Remove a surrounding markdown fence. Only used for scoring, never for execution."""
    if not isinstance(text, str):
        return "", False
    match = _FENCE.match(text)
    if match:
        return match.group(1).strip(), True
    return text.strip(), False


@dataclass(frozen=True)
class ParsedAction:
    kind: str
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    final: str | None = None


def parse_action(text: str) -> ParsedAction:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("model output is empty")
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("action must be a JSON object")
    if set(value) == {"final"} and isinstance(value["final"], str):
        return ParsedAction("final", final=value["final"])
    if set(value) != {"tool", "arguments"}:
        raise ValueError("tool action must contain exactly 'tool' and 'arguments'")
    if not isinstance(value["tool"], str) or not isinstance(value["arguments"], dict):
        raise ValueError("tool must be a string and arguments must be an object")
    return ParsedAction("tool", tool=value["tool"], arguments=value["arguments"])


def parse_plan(text: str) -> list[ParsedAction]:
    """Parse a GRPO completion as a JSON array of tool actions and optional final."""
    try:
        values = json.loads(text.strip())
    except (json.JSONDecodeError, AttributeError) as exc:
        raise ValueError("completion must be valid JSON") from exc
    if not isinstance(values, list) or not values:
        raise ValueError("completion must be a non-empty JSON array")
    return [parse_action(json.dumps(value)) for value in values]
