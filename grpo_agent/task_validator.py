from __future__ import annotations

import re
from typing import Any, Iterable

from .environment import ALLOWED_TO_MODIFY
from .policies import oracle_policy
from .rollout import run_rollout
from .task_generator import SCENARIOS

STRING_FIELDS = ("task_id", "prompt", "order_id", "product_id", "initial_order_status", "expected_outcome")
INT_FIELDS = ("initial_quantity", "available_stock", "requested_quantity")
# Words that would hand the answer to the model without any tool call.
LEAK_WORDS = ("pending", "processing", "shipped", "insufficient", "modified", "rejected", "expected_outcome")


def check_format(task: dict[str, Any]) -> list[str]:
    problems = []
    for field in STRING_FIELDS:
        value = task.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"format:{field}")
    for field in INT_FIELDS:
        value = task.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            problems.append(f"format:{field}")
    if task.get("scenario") not in SCENARIOS:
        problems.append("format:scenario")
    if task.get("expected_outcome") not in {"modified", "rejected"}:
        problems.append("format:expected_outcome")
    failures = task.get("transient_failures", {})
    if not isinstance(failures, dict) or any(isinstance(v, bool) or not isinstance(v, int) for v in failures.values()):
        problems.append("format:transient_failures")
    return problems


def check_business_consistency(task: dict[str, Any]) -> list[str]:
    """The expected outcome must follow from the business rules, not from the generator's word."""
    problems = []
    if task.get("expected_outcome") is None:
        return ["business:missing_expected_outcome"]
    invalid_id = str(task.get("order_id", "")).startswith("BAD-")
    modifiable = (task["initial_order_status"] in ALLOWED_TO_MODIFY
                  and task["available_stock"] >= task["requested_quantity"]
                  and not invalid_id)
    expected = task["expected_outcome"]
    if expected == "modified" and not modifiable:
        problems.append("business:expected_modified_but_blocked")
    if expected == "rejected" and modifiable:
        problems.append("business:expected_rejected_but_modifiable")
    if task["initial_quantity"] == task["requested_quantity"]:
        problems.append("business:no_op_request")
    if invalid_id != (task["scenario"] == "invalid_order"):
        problems.append("business:invalid_id_scenario_mismatch")
    if (task["scenario"] == "transient_failure") != bool(task.get("transient_failures")):
        problems.append("business:failure_flag_mismatch")
    return problems


def check_answer_leak(task: dict[str, Any]) -> list[str]:
    problems = []
    prompt = str(task.get("prompt", ""))
    numbers = {int(match) for match in re.findall(r"\d+", prompt)}
    lowered = prompt.lower()
    # The requested quantity legitimately appears in the prompt; the hidden state must not.
    if task["available_stock"] != task["requested_quantity"] and task["available_stock"] in numbers:
        problems.append("leak:available_stock")
    if task["initial_quantity"] != task["requested_quantity"] and task["initial_quantity"] in numbers:
        problems.append("leak:initial_quantity")
    for word in LEAK_WORDS:
        if word in lowered:
            problems.append(f"leak:{word}")
    return problems


def check_solvable(task: dict[str, Any], weights: dict[str, float], max_turns: int, max_tool_calls: int) -> list[str]:
    """Derive solvability from the environment itself: the oracle must actually reach the goal."""
    try:
        result = run_rollout(task, oracle_policy, weights, max_turns=max_turns, max_tool_calls=max_tool_calls)
    except Exception as exc:  # noqa: BLE001 - a malformed task must be reported, not raised
        return [f"solvable:rollout_error:{type(exc).__name__}"]
    if not result.verification["task_success"]:
        return [f"solvable:oracle_failed:{result.termination}"]
    return []


def validate_task(task: dict[str, Any], weights: dict[str, float], max_turns: int = 6, max_tool_calls: int = 5) -> dict:
    format_problems = check_format(task)
    business_problems = [] if format_problems else check_business_consistency(task)
    leak_problems = [] if format_problems else check_answer_leak(task)
    solvable_problems = ([] if format_problems or business_problems or leak_problems
                         else check_solvable(task, weights, max_turns, max_tool_calls))
    problems = format_problems + business_problems + leak_problems + solvable_problems
    return {"ok": not problems, "problems": problems}


def validate_tasks(tasks: Iterable[dict[str, Any]], weights: dict[str, float], max_turns: int = 6,
                   max_tool_calls: int = 5) -> dict:
    """Validate a whole split and break the outcome down by gate."""
    gate_counts: dict[str, int] = {}
    accepted = 0
    rejected = 0
    for task in tasks:
        report = validate_task(task, weights, max_turns, max_tool_calls)
        if report["ok"]:
            accepted += 1
            continue
        rejected += 1
        for problem in report["problems"]:
            gate = problem.split(":", 1)[0]
            gate_counts[gate] = gate_counts.get(gate, 0) + 1
    total = accepted + rejected
    return {"accepted": accepted, "rejected": rejected, "total": total,
            "pass_rate": accepted / total if total else 0.0, "rejected_by_gate": gate_counts}
