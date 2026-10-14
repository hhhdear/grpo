from __future__ import annotations

import json

TOOLS = [
    {"name": "query_order", "arguments": {"order_id": "string"}},
    {"name": "query_stock", "arguments": {"product_id": "string"}},
    {"name": "modify_order", "arguments": {"order_id": "string", "quantity": "positive integer"}},
    {"name": "verify_order", "arguments": {"order_id": "string"}},
]

# Fixed demonstration identifiers. They never match a real task, so the examples
# teach the protocol shape instead of leaking an answer.
EXAMPLE_ORDER_ID = "ORD-000001"
EXAMPLE_PRODUCT_ID = "SKU-100"
EXAMPLE_QUANTITY = 3

NO_FENCE_RULE = ("Return only the JSON value. No markdown code fences, no prose before or after it. "
                 "A tool action uses exactly the keys \"tool\" and \"arguments\"; a finish step uses "
                 "exactly the key \"final\".")


def _dump(value) -> str:
    return json.dumps(value, separators=(",", ":"))


def _success_plan() -> str:
    return _dump([
        {"tool": "query_order", "arguments": {"order_id": EXAMPLE_ORDER_ID}},
        {"tool": "query_stock", "arguments": {"product_id": EXAMPLE_PRODUCT_ID}},
        {"tool": "modify_order", "arguments": {"order_id": EXAMPLE_ORDER_ID, "quantity": EXAMPLE_QUANTITY}},
        {"tool": "verify_order", "arguments": {"order_id": EXAMPLE_ORDER_ID}},
        {"final": f"Order {EXAMPLE_ORDER_ID} was changed to {EXAMPLE_QUANTITY} and verified."},
    ])


def _refusal_plan() -> str:
    return _dump([
        {"tool": "query_order", "arguments": {"order_id": EXAMPLE_ORDER_ID}},
        {"final": f"Order {EXAMPLE_ORDER_ID} has status shipped, so no change was made."},
    ])


def _single_step_examples() -> str:
    return "\n".join([
        _dump({"tool": "query_order", "arguments": {"order_id": EXAMPLE_ORDER_ID}}),
        _dump({"final": "Insufficient stock; no change was made."}),
    ])


def system_prompt(plan_mode: bool = False) -> str:
    shape = "a JSON array containing the complete action plan" if plan_mode else "one JSON object for the next action"
    return ("You manage synthetic orders. Never claim a change without tool evidence. Query before changing, obey "
            "status and stock rules, and verify successful changes. Return only " + shape + ". Tools: " +
            json.dumps(TOOLS, separators=(",", ":")))


def fewshot_block(plan_mode: bool = False) -> str:
    """Demonstrations of the exact wire format, appended after the system rules."""
    if plan_mode:
        return (NO_FENCE_RULE + "\nExamples:\nUser: Change order X to 3 units if allowed.\nAssistant: " +
                _success_plan() + "\nUser: Change order Y to 9 units if allowed.\nAssistant: " + _refusal_plan())
    return NO_FENCE_RULE + "\nExamples:\nAssistant: " + _single_step_examples()


def build_messages(user_prompt: str, transcript: list[dict] | None = None, fewshot: bool = True) -> list[dict[str, str]]:
    system = system_prompt(False) + ("\n" + fewshot_block(False) if fewshot else "")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}]
    for turn in transcript or []:
        messages.append({"role": "assistant", "content": turn["model_output"]})
        messages.append({"role": "tool", "content": json.dumps(turn["tool_result"], ensure_ascii=False)})
    return messages


def build_training_prompt(user_prompt: str, fewshot: bool = True) -> str:
    head = system_prompt(True) + ("\n" + fewshot_block(True) if fewshot else "")
    return head + "\nUser: " + user_prompt + "\nAssistant:"
