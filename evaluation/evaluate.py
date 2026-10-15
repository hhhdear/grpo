from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grpo_agent.config import load_config
from grpo_agent.metrics import aggregate_metrics
from grpo_agent.policies import make_batch_model_policy, make_model_policy, oracle_policy
from grpo_agent.rollout import append_trace, run_rollout, run_rollout_batch
from grpo_agent.task_generator import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--policy", choices=("oracle", "model"), default="model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples", type=int, default=1,
                        help="extra sampled attempts per task; >1 adds success@k on top of greedy success@1")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--tag", default=None,
                        help="names the output file so ablation arms do not overwrite each other")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tasks = read_jsonl(cfg["data"]["test_path"])
    if args.limit is not None:
        tasks = tasks[:args.limit]
    checkpoint = Path(cfg["training"]["output_dir"])
    # Evaluate the trained adapter. Falling back to the base model here would silently
    # report an untrained policy's numbers as if they belonged to the run.
    model_dir = str(checkpoint) if (checkpoint / "adapter_config.json").is_file() else None
    if model_dir is None:
        print(f"WARNING: no trained adapter in {checkpoint}; evaluating the base model", file=sys.stderr)
    policy = oracle_policy if args.policy == "oracle" else make_model_policy(cfg["model_config"], model_dir=model_dir)
    output_dir = Path(cfg["paths"]["evaluation_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.policy
    trace = Path(cfg["paths"]["trace_dir"]) / f"evaluate_{tag}.jsonl"
    if trace.exists():
        trace.unlink()
    format_weights = cfg["reward"].get("format")
    # The oracle is deterministic, so sampling it k times would only repeat the same answer.
    # Sampling runs as a batched rollout: one generation per turn for all k trajectories.
    sampler = None
    if args.policy == "model" and args.samples > 1:
        sampler = make_batch_model_policy(cfg["model_config"], temperature=args.temperature, model_dir=model_dir)
    rows = []
    for task in tasks:
        result = run_rollout(task, policy, cfg["reward"], format_weights=format_weights, **cfg["rollout"])
        append_trace(result, trace)
        row = {"scenario": task["scenario"], **result.to_dict()}
        if sampler is not None:
            batch = run_rollout_batch(task, sampler, cfg["reward"], args.samples,
                                      format_weights=format_weights, **cfg["rollout"])
            row["sampled_successes"] = batch["successes"]
            row["sampled_tool_calls"] = batch["tool_calls"]
            row["sampled_invalid_actions"] = batch["invalid_actions"]
            row["sampled_terminations"] = batch["terminations"]
        rows.append(row)
    report = {"policy": args.policy, "samples": args.samples, "temperature": args.temperature,
              "metrics": aggregate_metrics(rows), "records": rows}
    destination = output_dir / f"{tag}_metrics.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
