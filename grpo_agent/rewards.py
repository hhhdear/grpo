from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


# Share of the progress credit each completed intermediate step is worth. Sums to 1.0, so
# the configured `progress_credit` weight is the maximum a trajectory can earn here.
PROGRESS_SHARES = {"queried_order": 0.2, "queried_stock": 0.2, "modified_ok": 0.4, "verified_called": 0.2}


@dataclass(frozen=True)
class RewardBreakdown:
    task_success: float
    rule_compliance: float
    verification: float
    invalid_action: float
    redundant_calls: float
    execution_cost: float
    progress_credit: float

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
        _progress_credit(verification, weights["progress_credit"]),
    )


def _progress_credit(verification: dict[str, Any], weight: float) -> float:
    """Partial credit for completed intermediate steps of a multi-step task."""
    progress = verification.get("progress") or {}
    done = sum(share * bool(progress.get(key)) for key, share in PROGRESS_SHARES.items())
    return weight * done
