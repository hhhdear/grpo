from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class RewardBreakdown:
    task_success: float
    rule_compliance: float
    verification: float
    invalid_action: float
    redundant_calls: float
    execution_cost: float

    @property
    def total(self) -> float:
        return sum(asdict(self).values())

    def to_dict(self) -> dict[str, float]:
        return {**asdict(self), "total": self.total}


def calculate_reward(verification: dict[str, Any], weights: dict[str, float]) -> RewardBreakdown:
    missing = set(RewardBreakdown.__dataclass_fields__) - weights.keys()
    if missing:
        raise ValueError(f"Missing reward weights: {sorted(missing)}")
    return RewardBreakdown(
        weights["task_success"] if verification["task_success"] else 0.0,
        weights["rule_compliance"] if verification["rule_compliance"] else 0.0,
        weights["verification"] if verification["verification"] else 0.0,
        weights["invalid_action"] * verification["invalid_actions"],
        weights["redundant_calls"] * verification["redundant_calls"],
        weights["execution_cost"] * verification["tool_calls"],
    )
