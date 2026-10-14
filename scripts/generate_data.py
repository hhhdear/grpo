from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grpo_agent.config import load_config
from grpo_agent.task_generator import generate_tasks, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data, seed = cfg["data"], int(cfg["seed"])
    specs = [("train", 0), ("validation", 10_000_000), ("test", 20_000_000)]
    for name, offset in specs:
        tasks = generate_tasks(int(data[f"{name}_size"]), seed + offset)
        write_jsonl(tasks, data[f"{name}_path"])
        print(f"wrote {len(tasks)} tasks to {data[f'{name}_path']}")


if __name__ == "__main__":
    main()
