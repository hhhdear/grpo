from __future__ import annotations

import json
from typing import Any

from .parser import strip_code_fence

TOOL_ARGUMENTS = {
    "query_order": {"order_id"},
    "query_stock": {"product_id"},
    "modify_order": {"order_id", "quantity"},
    "verify_order": {"order_id"},
}

DEFAULT_FORMAT_WEIGHTS = {"structure": 0.4, "known_tool": 0.2, "valid_arguments": 0.2, "fence_penalty": 0.3}


def _is_final(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {"final"} and isinstance(value["final"], str)


def _check_action(value: Any) -> tuple[bool, bool, bool]:
    """Return (structure_ok, tool_known, arguments_valid) for one decoded action."""
    if _is_final(value):
        return True, True, True
    if not isinstance(value, dict) or set(value) != {"tool", "arguments"}:
        return False, False, False
    tool, arguments = value["tool"], value["arguments"]
    if not isinstance(tool, str) or not isinstance(arguments, dict):
        return False, False, False
    expected = TOOL_ARGUMENTS.get(tool)
    if expected is None:
        return True, False, False
    if set(arguments) != expected:
        return True, True, False
    if tool == "modify_order":
        quantity = arguments["quantity"]
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            return True, True, False
    if any(not isinstance(argument, str) or not argument.strip()
           for key, argument in arguments.items() if key != "quantity"):
        return True, True, False
    return True, True, True


def format_diagnosis(text: str) -> dict[str, Any]:
    """Measure how close a completion is to the tool protocol, without executing it."""
    cleaned, fenced = strip_code_fence(text)
    empty = {"parsed": False, "fenced": fenced, "actions": 0, "structure_ratio": 0.0,
             "tool_ratio": 0.0, "argument_ratio": 0.0}
    try:
        value = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return empty
    items = value if isinstance(value, list) else [value]
    if not items:
        return empty
    structure = tools = arguments = 0
    for item in items:
        ok_structure, ok_tool, ok_arguments = _check_action(item)
        structure += int(ok_structure)
        tools += int(ok_tool)
        arguments += int(ok_arguments)
    total = len(items)
    return {"parsed": True, "fenced": fenced, "actions": total, "structure_ratio": structure / total,
            "tool_ratio": tools / total, "argument_ratio": arguments / total}


def format_credit(text: str, weights: dict[str, float] | None = None) -> float:
    """Partial credit for protocol compliance. Breaks reward ties inside a GRPO group."""
    if not weights:
        return 0.0
    diagnosis = format_diagnosis(text)
    return (float(weights.get("structure", 0.4)) * diagnosis["structure_ratio"] +
            float(weights.get("known_tool", 0.2)) * diagnosis["tool_ratio"] +
            float(weights.get("valid_arguments", 0.2)) * diagnosis["argument_ratio"] -
            float(weights.get("fence_penalty", 0.3)) * int(diagnosis["fenced"]))
