from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable


def aggregate_metrics(records: Iterable[dict]) -> dict:
    rows = list(records)
    if not rows:
        return {"count": 0}
    per_scenario = defaultdict(list)
    per_scenario_at_k = defaultdict(list)
    for row in rows:
        per_scenario[row["scenario"]].append(bool(row["verification"]["task_success"]))
        sampled = row.get("sampled_successes")
        if sampled:
            per_scenario_at_k[row["scenario"]].append(any(sampled))
    calls = sum(row["verification"]["tool_calls"] for row in rows)
    invalid = sum(row["verification"]["invalid_actions"] for row in rows)
    modified = [row for row in rows if row["verification"]["expected_outcome"] == "modified"]
    metrics = {
        "count": len(rows),
        "task_success_rate": mean(bool(r["verification"]["task_success"]) for r in rows),
        "rule_compliance_rate": mean(bool(r["verification"]["rule_compliance"]) for r in rows),
        "tool_selection_accuracy": mean(r.get("tool_selection_accuracy", 1.0 if r["verification"]["invalid_actions"] == 0 else 0.0) for r in rows),
        "tool_argument_accuracy": mean(r.get("tool_argument_accuracy", 1.0 if r["verification"]["invalid_actions"] == 0 else 0.0) for r in rows),
        "average_tool_calls": calls / len(rows),
        "invalid_tool_call_rate": invalid / calls if calls else 0.0,
        "post_modify_verification_rate": mean(bool(r["verification"]["verification"]) for r in modified) if modified else 0.0,
        "average_execution_cost": calls / len(rows),
        "average_reward": mean(r["rewards"]["total"] for r in rows),
        "success_rate_by_scenario": {key: mean(values) for key, values in sorted(per_scenario.items())},
    }
    sampled = [row["sampled_successes"] for row in rows if row.get("sampled_successes")]
    if sampled:
        k = len(sampled[0])
        # success@1 stays greedy; success@k is "at least one of k samples solved it".
        metrics["success_at_1"] = metrics["task_success_rate"]
        metrics[f"success_at_{k}"] = mean(any(sample) for sample in sampled)
        metrics["sample_size_k"] = k
        metrics["mean_sample_success_rate"] = mean(mean(sample) for sample in sampled)
        metrics[f"success_rate_by_scenario_at_{k}"] = {key: mean(values) for key, values in sorted(per_scenario_at_k.items())}
    return metrics
