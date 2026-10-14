from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grpo_agent.config import load_config
from grpo_agent.policies import oracle_policy
from grpo_agent.rollout import run_rollout
from grpo_agent.task_generator import generate_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results = [run_rollout(task.to_dict(), oracle_policy, cfg["reward"], **cfg["rollout"])
               for task in generate_tasks(12, int(cfg["seed"]))]
    successes = sum(result.verification["task_success"] for result in results)
    print(f"CPU environment/tool smoke: {successes}/{len(results)} successful")
    if successes != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
