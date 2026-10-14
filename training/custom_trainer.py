"""Stable extension seams around TRL, without vendoring third-party trainer code."""
from __future__ import annotations

from typing import Any, Callable


class TrajectoryResearchHooks:
    """Hooks for a future, explicitly specified credit-assignment experiment.

    `transform_trajectory` may add per-step research signals without mutating the
    environment record. `aggregate_reward` maps those signals to the scalar
    completion reward expected by stock GRPO. Replacing GRPO advantages or loss
    requires a separately tested trainer subclass tied to a pinned TRL version.
    """

    def transform_trajectory(self, trajectory: dict[str, Any]) -> dict[str, Any]:
        return trajectory

    def aggregate_reward(self, trajectory: dict[str, Any], default_reward: float) -> float:
        return default_reward


def safe_extension_points() -> tuple[str, ...]:
    return ("completion replay", "reward components", "trajectory logging", "metrics callbacks")
