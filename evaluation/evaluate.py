from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grpo_agent.config import load_config
from grpo_agent.metrics import aggregate_metrics
from grpo_agent.policies import make_model_policy, oracle_policy
from grpo_agent.rollout import append_trace, run_rollout
from grpo_agent.task_generator import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--policy", choices=("oracle", "model"), default="model")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    tasks = read_jsonl(cfg["data"]["test_path"])
    if args.limit is not None:
        tasks = tasks[:args.limit]
    policy = oracle_policy if args.policy == "oracle" else make_model_policy(cfg["model_config"])
    output_dir = Path(cfg["paths"]["evaluation_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = Path(cfg["paths"]["trace_dir"]) / f"evaluate_{args.policy}.jsonl"
    if trace.exists():
        trace.unlink()
    rows = []
    for task in tasks:
        result = run_rollout(task, policy, cfg["reward"], **cfg["rollout"])
        append_trace(result, trace)
        rows.append({"scenario": task["scenario"], **result.to_dict()})
    report = {"policy": args.policy, "metrics": aggregate_metrics(rows), "records": rows}
    destination = output_dir / f"{args.policy}_metrics.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
