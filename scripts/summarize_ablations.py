from __future__ import annotations

"""Collect every arm's training and evaluation metrics into one comparison table.

Nothing here invents numbers: a metric that was never recorded stays blank, so an absent
kl or a missing success@8 shows up as empty rather than as a fabricated zero.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ARMS = ("vanilla", "clip_higher", "token_level", "overlong", "dynamic_sampling", "dapo_full")

TRAIN_FIELDS = ("reward", "reward_std", "frac_reward_zero_std", "frac_zero_var_filtered", "entropy", "kl",
                "grad_norm", "completion_mean_length", "clipped_ratio", "trajectory_tool_calls",
                "step_time", "tokens_per_second", "clip_ratio_low", "clip_ratio_high")
REWARD_TERMS = ("task_success", "rule_compliance", "verification", "invalid_action", "redundant_calls",
                "execution_cost", "format_credit", "overlong_penalty")
EVAL_FIELDS = ("success_at_1", "success_at_8", "mean_sample_success_rate", "task_success_rate",
               "rule_compliance_rate", "average_tool_calls", "invalid_tool_call_rate", "average_reward",
               "post_modify_verification_rate")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def output_dir(arm: str) -> Path:
    import yaml

    cfg = yaml.safe_load((ROOT / f"configs/ablations/{arm}.yaml").read_text(encoding="utf-8"))
    return ROOT / cfg["training"]["output_dir"]


def collect(arm: str, samples: int) -> dict:
    summary = load(output_dir(arm) / "metrics" / "train_metrics_summary.json")
    evaluation = load(ROOT / "logs" / "evaluation" / f"ablation_{arm}_metrics.json")
    means = summary.get("mean", {})
    metrics = evaluation.get("metrics", {})
    row = {"steps": summary.get("steps")}
    for field in TRAIN_FIELDS:
        row[field] = means.get(field)
    for term in REWARD_TERMS:
        row[f"reward_term/{term}"] = means.get(f"reward_term/{term}")
    for field in EVAL_FIELDS:
        row[field] = metrics.get(field)
    row[f"success_at_{samples}"] = metrics.get(f"success_at_{samples}", row.get("success_at_8"))
    for scenario, rate in (metrics.get(f"success_rate_by_scenario_at_{samples}")
                           or metrics.get("success_rate_by_scenario") or {}).items():
        row[f"scenario/{scenario}"] = rate
    row["missing_metrics"] = summary.get("missing_metrics", [])
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--include-base", action="store_true")
    args = parser.parse_args()
    table = {arm: collect(arm, args.samples) for arm in ARMS}
    if args.include_base:
        base = load(ROOT / "logs" / "evaluation" / "base_metrics.json").get("metrics", {})
        table["base_model"] = {**{field: None for field in TRAIN_FIELDS},
                               **{f"reward_term/{t}": None for t in REWARD_TERMS},
                               **{field: base.get(field) for field in EVAL_FIELDS}}
    keys: list[str] = []
    for row in table.values():
        for key in row:
            if key not in keys and key != "missing_metrics":
                keys.append(key)
    columns = ["metric"] + list(table)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for key in keys:
        cells = [key]
        for arm in table:
            value = table[arm].get(key)
            cells.append("" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value)))
        lines.append("| " + " | ".join(cells) + " |")
    markdown = "\n".join(lines)
    print(markdown)
    destination = ROOT / "logs" / "ablation_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "logs" / "ablation_summary.md").write_text(markdown + "\n", encoding="utf-8")
    print(f"\nsaved {destination}")
    for arm, row in table.items():
        if row.get("missing_metrics"):
            print(f"note {arm}: never observed {', '.join(row['missing_metrics'])}")


if __name__ == "__main__":
    main()
