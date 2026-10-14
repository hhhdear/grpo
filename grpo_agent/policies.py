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


def make_batch_model_policy(model_config_path: str, temperature: float = 0.0, top_p: float = 1.0,
                            max_new_tokens: int = 192):
    """Same policy, but scores a list of message sets in one forward pass per turn.

    Used to run the k sampled trajectories of success@k as one batch: the trajectories stay
    independent and observe exactly what the sequential loop would give them, they are only
    generated together.
    """
    from .model_loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config_path)
    tokenizer.padding_side = "left"

    def policy(batch_messages):
        import torch
        texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                 for messages in batch_messages]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        sampling = temperature > 0.0
        kwargs = {"max_new_tokens": max_new_tokens, "do_sample": sampling}
        if sampling:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        if tokenizer.pad_token_id is not None:
            kwargs["pad_token_id"] = tokenizer.pad_token_id
        with torch.no_grad():
            output = model.generate(**inputs, **kwargs)
        generated = output[:, inputs["input_ids"].shape[1]:]
        return [tokenizer.decode(row, skip_special_tokens=True).strip() for row in generated]
    return policy


def make_model_policy(model_config_path: str, temperature: float = 0.0, top_p: float = 1.0, max_new_tokens: int = 192):
    """Greedy by default. `temperature > 0` switches to sampling, which is what success@k needs."""
    from .model_loader import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_config_path)

    def policy(messages, env):
        import torch
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        sampling = temperature > 0.0
        kwargs = {"max_new_tokens": max_new_tokens, "do_sample": sampling}
        if sampling:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        if tokenizer.pad_token_id is not None:
            kwargs["pad_token_id"] = tokenizer.pad_token_id
        with torch.no_grad():
            output = model.generate(**inputs, **kwargs)
        return tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return policy
