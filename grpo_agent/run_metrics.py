from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

REWARD_TERM_KEYS = ("task_success", "rule_compliance", "verification", "invalid_action",
                    "redundant_calls", "execution_cost", "format_credit", "overlong_penalty")

# TRL log key -> canonical name used by every ablation run, so that six runs stay comparable.
TRL_FIELD_MAP = {
    "reward": "reward",
    "reward_std": "reward_std",
    "frac_reward_zero_std": "frac_reward_zero_std",
    "rewards/reward_func/mean": "reward_func_mean",
    "entropy": "entropy",
    "kl": "kl",
    "loss": "loss",
    "grad_norm": "grad_norm",
    "learning_rate": "learning_rate",
    "completions/mean_length": "completion_mean_length",
    "completions/clipped_ratio": "clipped_ratio",
    "completions/mean_terminated_length": "completion_terminated_length",
    "clip_ratio/low_mean": "clip_ratio_low",
    "clip_ratio/high_mean": "clip_ratio_high",
    "clip_ratio/region_mean": "clip_ratio_region",
    "frac_zero_var_filtered": "frac_zero_var_filtered",
    "num_tokens": "num_tokens",
    "step_time": "step_time",
    "epoch": "epoch",
}

CANONICAL_FIELDS = ("step", "loss", "grad_norm", "learning_rate", "reward", "reward_std",
                    "frac_reward_zero_std", "frac_zero_var_filtered", "entropy", "kl",
                    "clip_ratio_low", "clip_ratio_high", "clip_ratio_region",
                    "completion_mean_length", "clipped_ratio", "completion_terminated_length",
                    "trajectory_tool_calls", "trajectory_turns", "trajectory_invalid_actions",
                    "invalid_action_rate", "num_tokens", "step_time", "tokens_per_second")


class RewardLedger:
    """Collects per-completion reward breakdowns and folds them into per-step group means.

    TRL calls every reward function exactly once per optimizer step with the whole group
    batch, so one `close_batch()` corresponds to one step.
    """

    def __init__(self) -> None:
        self._pending: list[dict[str, float]] = []
        self.batches: list[dict[str, float]] = []

    def add(self, breakdown: dict[str, float], verification: dict[str, Any]) -> None:
        row = {key: float(breakdown.get(key, 0.0)) for key in REWARD_TERM_KEYS}
        row["total"] = float(breakdown.get("total", 0.0))
        calls = float(verification.get("tool_calls", 0) or 0)
        invalid = float(verification.get("invalid_actions", 0) or 0)
        row["tool_calls"] = calls
        row["invalid_actions"] = invalid
        # Watched whenever the invalid-action penalty is loosened: a cheap penalty can
        # invite the model to spam calls instead of learning to call correctly.
        row["invalid_action_rate"] = invalid / calls if calls else 0.0
        self._pending.append(row)

    def close_batch(self) -> None:
        if not self._pending:
            return
        size = len(self._pending)
        batch = {key: sum(row[key] for row in self._pending) / size for key in self._pending[0]}
        batch["group_size"] = float(size)
        self.batches.append(batch)
        self._pending = []


    def step(self, index: int) -> dict[str, float]:
        if 0 <= index < len(self.batches):
            return self.batches[index]
        return {}


def build_step_rows(log_history: Iterable[dict[str, Any]], ledger: RewardLedger) -> list[dict[str, Any]]:
    """Normalize TRL's log history plus the reward ledger into one row per optimizer step."""
    rows: list[dict[str, Any]] = []
    step = 0
    for entry in log_history:
        if "loss" not in entry and "reward" not in entry:
            # Skip the final runtime summary emitted by Trainer.
            continue
        row: dict[str, Any] = {"step": step}
        for source, target in TRL_FIELD_MAP.items():
            if source in entry:
                row[target] = entry[source]
        ledger_row = ledger.step(step)
        for key in REWARD_TERM_KEYS:
            if key in ledger_row:
                row[f"reward_term/{key}"] = ledger_row[key]
        if ledger_row:
            # In plan mode one tool call is one trajectory turn.
            row["trajectory_tool_calls"] = ledger_row.get("tool_calls")
            row["trajectory_turns"] = ledger_row.get("tool_calls")
            row["trajectory_invalid_actions"] = ledger_row.get("invalid_actions")
            row["invalid_action_rate"] = ledger_row.get("invalid_action_rate")
            row["group_size"] = ledger_row.get("group_size")
        tokens = row.get("num_tokens")
        elapsed = row.get("step_time")
        if tokens is not None and elapsed:
            row["tokens_per_second"] = float(tokens) / float(elapsed)
        rows.append(row)
        step += 1
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate step rows and report which requested metrics were never observed."""
    if not rows:
        return {"steps": 0, "missing_metrics": sorted(CANONICAL_FIELDS)}

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows if row.get(key) is not None]

    summary: dict[str, Any] = {"steps": len(rows)}
    means: dict[str, float] = {}
    for key in CANONICAL_FIELDS:
        if key == "step":
            continue
        observed = values(key)
        if observed:
            means[key] = sum(observed) / len(observed)
    summary["mean"] = means
    summary["first"] = {key: rows[0][key] for key in rows[0] if key != "step"}
    summary["last"] = {key: rows[-1][key] for key in rows[-1] if key != "step"}
    missing = [key for key in CANONICAL_FIELDS if key != "step" and key not in means
               and not any(key in row for row in rows)]
    summary["missing_metrics"] = sorted(missing)
    return summary


def write_metrics(rows: list[dict[str, Any]], path: str | Path) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(rows)
    summary_path = destination.with_name(destination.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
