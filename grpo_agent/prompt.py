from __future__ import annotations

import json

TOOLS = [
    {"name": "query_order", "arguments": {"order_id": "string"}},
    {"name": "query_stock", "arguments": {"product_id": "string"}},
    {"name": "modify_order", "arguments": {"order_id": "string", "quantity": "positive integer"}},
    {"name": "verify_order", "arguments": {"order_id": "string"}},
]


def system_prompt(plan_mode: bool = False) -> str:
    shape = "a JSON array containing the complete action plan" if plan_mode else "one JSON object for the next action"
    return ("You manage synthetic orders. Never claim a change without tool evidence. Query before changing, obey "
            "status and stock rules, and verify successful changes. Return only " + shape + ". Tools: " +
            json.dumps(TOOLS, separators=(",", ":")))


def build_messages(user_prompt: str, transcript: list[dict] | None = None) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt(False)}, {"role": "user", "content": user_prompt}]
    for turn in transcript or []:
        messages.append({"role": "assistant", "content": turn["model_output"]})
        messages.append({"role": "tool", "content": json.dumps(turn["tool_result"], ensure_ascii=False)})
    return messages


def build_training_prompt(user_prompt: str) -> str:
    return system_prompt(True) + "\nUser: " + user_prompt + "\nAssistant:"
