from __future__ import annotations

import json
import re

from .environment import OrderEnvironment


def oracle_policy(messages: list[dict[str, str]], env: OrderEnvironment) -> str:
    """Deterministic test policy; never used as an LLM substitute in claimed evaluations."""
    task, history = env.task, env.history
    oid = task["order_id"]
    if not history or (history[-1]["result"]["code"] == "temporary_failure"):
        return json.dumps({"tool": "query_order", "arguments": {"order_id": oid}})
    last = history[-1]["result"]
    if not last["ok"]:
        return json.dumps({"final": "The requested operation is not allowed or the order does not exist."})
    if history[-1]["tool"] == "query_order":
        if last["data"]["status"] not in {"pending", "processing"}:
            return json.dumps({"final": "The order status forbids modification."})
        return json.dumps({"tool": "query_stock", "arguments": {"product_id": last["data"]["product_id"]}})
    if history[-1]["tool"] == "query_stock":
        if last["data"]["available_stock"] < task["requested_quantity"]:
            return json.dumps({"final": "Insufficient stock; no change was made."})
        return json.dumps({"tool": "modify_order", "arguments": {"order_id": oid, "quantity": task["requested_quantity"]}})
    if history[-1]["tool"] == "modify_order":
        return json.dumps({"tool": "verify_order", "arguments": {"order_id": oid}})
    return json.dumps({"final": "Completed and verified."})


def make_model_policy(model_config_path: str):
    from .model_loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config_path)

    def policy(messages, env):
        import torch
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=192, do_sample=False)
        return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return policy
