from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

ALLOWED_TO_MODIFY = {"pending", "processing"}


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    tool: str
    data: dict[str, Any]
    error: str | None = None
    code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OrderEnvironment:
    def __init__(self, task: dict[str, Any]):
        self.task = deepcopy(task)
        actual_id = task.get("actual_order_id") or (
            f"ORD-{int(task['seed']) % 1000000:06d}" if task["scenario"] == "invalid_order" else task["order_id"]
        )
        self.orders = {actual_id: {"order_id": actual_id, "status": task["initial_order_status"],
                                   "product_id": task["product_id"], "quantity": task["initial_quantity"]}}
        self.stock = {task["product_id"]: task["available_stock"]}
        self.failures = dict(task.get("transient_failures", {}))
        self.history: list[dict[str, Any]] = []
        self.invalid_actions = 0
        self.verified = False

    def snapshot(self) -> dict[str, Any]:
        return {"orders": deepcopy(self.orders), "stock": deepcopy(self.stock), "verified": self.verified}

    @staticmethod
    def _string(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value

    def execute(self, tool: str, arguments: dict[str, Any]) -> ToolResult:
        before = self.snapshot()
        if not isinstance(arguments, dict):
            result = ToolResult(False, tool, {}, "arguments must be an object", "invalid_arguments")
        elif tool not in {"query_order", "query_stock", "modify_order", "verify_order"}:
            result = ToolResult(False, tool, {}, f"unknown tool: {tool}", "unknown_tool")
        elif self.failures.get(tool, 0) > 0:
            self.failures[tool] -= 1
            result = ToolResult(False, tool, {}, "temporary tool failure; retry is allowed", "temporary_failure")
        else:
            try:
                result = getattr(self, f"_{tool}")(**arguments)
            except TypeError as exc:
                result = ToolResult(False, tool, {}, f"invalid tool signature: {exc}", "invalid_arguments")
            except ValueError as exc:
                result = ToolResult(False, tool, {}, str(exc), "invalid_arguments")
        if not result.ok and result.code not in {"temporary_failure", "not_found"}:
            self.invalid_actions += 1
        self.history.append({"tool": tool, "arguments": deepcopy(arguments), "result": result.to_dict(),
                             "state_before": before, "state_after": self.snapshot()})
        return result

    def _query_order(self, order_id: Any) -> ToolResult:
        order_id = self._string(order_id, "order_id")
        if order_id not in self.orders:
            return ToolResult(False, "query_order", {}, "order not found", "not_found")
        return ToolResult(True, "query_order", deepcopy(self.orders[order_id]))

    def _query_stock(self, product_id: Any) -> ToolResult:
        product_id = self._string(product_id, "product_id")
        if product_id not in self.stock:
            return ToolResult(False, "query_stock", {}, "product not found", "not_found")
        return ToolResult(True, "query_stock", {"product_id": product_id, "available_stock": self.stock[product_id]})

    def _modify_order(self, order_id: Any, quantity: Any) -> ToolResult:
        order_id = self._string(order_id, "order_id")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            raise ValueError("quantity must be a positive integer")
        order = self.orders.get(order_id)
        if order is None:
            return ToolResult(False, "modify_order", {}, "order not found", "not_found")
        if order["status"] not in ALLOWED_TO_MODIFY:
            return ToolResult(False, "modify_order", {}, f"status {order['status']} cannot be modified", "forbidden_status")
        available = self.stock[order["product_id"]]
        if quantity > available:
            return ToolResult(False, "modify_order", {"available_stock": available}, "insufficient stock", "insufficient_stock")
        order["quantity"] = quantity
        self.verified = False
        return ToolResult(True, "modify_order", deepcopy(order))

    def _verify_order(self, order_id: Any) -> ToolResult:
        order_id = self._string(order_id, "order_id")
        order = self.orders.get(order_id)
        if order is None:
            return ToolResult(False, "verify_order", {}, "order not found", "not_found")
        matches = order["quantity"] == self.task["requested_quantity"]
        self.verified = matches
        return ToolResult(True, "verify_order", {**deepcopy(order), "matches_requested_quantity": matches})

    def verify_task(self) -> dict[str, Any]:
        expected = self.task["expected_outcome"]
        requested_id = self.task["order_id"]
        order = self.orders.get(requested_id)
        changed = bool(order and order["quantity"] != self.task["initial_quantity"])
        if expected == "modified":
            success = bool(order and order["quantity"] == self.task["requested_quantity"] and self.verified)
            compliant = self.invalid_actions == 0
        else:
            # A no-op is not automatically a correct refusal. The agent must
            # gather environment evidence for the particular rejection reason.
            successful_results = [item for item in self.history if item["result"]["ok"]]
            observed_codes = {item["result"]["code"] for item in self.history}
            scenario = self.task["scenario"]
            if scenario == "invalid_order":
                evidence = "not_found" in observed_codes
            elif scenario in {"shipped", "combined"}:
                evidence = any(item["tool"] == "query_order" and
                               item["result"]["data"].get("status") not in ALLOWED_TO_MODIFY
                               for item in successful_results)
            elif scenario == "insufficient_stock":
                evidence = any(item["tool"] == "query_stock" and
                               item["result"]["data"].get("available_stock", 0) < self.task["requested_quantity"]
                               for item in successful_results)
            else:
                evidence = False
            success = not changed and evidence
            compliant = not changed and self.invalid_actions == 0
        calls = [(item["tool"], item["arguments"]) for item in self.history]
        redundant = len(calls) - len({repr(item) for item in calls})
        return {"task_success": success, "rule_compliance": compliant, "verification": self.verified,
                "invalid_actions": self.invalid_actions, "redundant_calls": max(0, redundant),
                "tool_calls": len(self.history), "changed": changed, "expected_outcome": expected}
