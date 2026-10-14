from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grpo_agent.config import load_config
from grpo_agent.task_generator import Task, combination_signature, generate_task, write_jsonl
from grpo_agent.task_validator import validate_task


def build_split(count: int, seed: int, weights: dict, rollout_cfg: dict, exclude: set) -> tuple[list[Task], set, int, int]:
    """Generate `count` tasks that are unique on their parameter combination and pass validation.

    Combination uniqueness is what makes the held-out set a genuine held-out set, so a
    candidate that repeats an already-taken combination is discarded rather than emitted.
    """
    if count <= 0:
        return [], set(exclude), 0, 0
    accepted: list[Task] = []
    seen = set(exclude)
    rejected = 0
    attempts = 0
    max_attempts = max(count * 80, 2000)
    while len(accepted) < count and attempts < max_attempts:
        task = generate_task(seed + attempts * 104729, attempts, wide=True)
        attempts += 1
        signature = combination_signature(task)
        if signature in seen:
            continue
        if not validate_task(task.to_dict(), weights, **rollout_cfg)["ok"]:
            rejected += 1
            continue
        seen.add(signature)
        accepted.append(task)
    if len(accepted) < count:
        raise RuntimeError(f"only {len(accepted)}/{count} tasks survived dedup and validation after {attempts} attempts")
    # Return only the signatures this split introduced; `seen` also carries the excluded ones.
    return accepted, {combination_signature(task) for task in accepted}, rejected, attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    data, seed = cfg["data"], int(cfg["seed"])
    weights = {key: value for key, value in cfg["reward"].items() if isinstance(value, (int, float))}
    rollout_cfg = {"max_turns": int(cfg["rollout"]["max_turns"]), "max_tool_calls": int(cfg["rollout"]["max_tool_calls"])}
    specs = [("train", 0), ("validation", 10_000_000), ("test", 20_000_000)]
    manifest: dict = {"seed": seed, "wide_parameter_space": True, "reward_weights": weights,
                      "rollout": rollout_cfg, "splits": {}}
    taken: set = set()
    for name, offset in specs:
        size = int(data[f"{name}_size"])
        tasks, signatures, rejected, attempts = build_split(size, seed + offset, weights, rollout_cfg, taken)
        path = data[f"{name}_path"]
        write_jsonl(tasks, path)
        overlap = signatures & taken
        manifest["splits"][name] = {
            "path": path, "count": len(tasks),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "scenario_distribution": dict(sorted(Counter(task.scenario for task in tasks).items())),
            "unique_combinations": len({combination_signature(task) for task in tasks}),
            "validation_rejected": rejected, "generation_attempts": attempts,
            "overlapping_combinations_with_previous_splits": len(overlap),
        }
        taken |= signatures
        print(f"{name}: {len(tasks)} tasks, {rejected} rejected by validation, "
              f"{len(overlap)} combinations overlapping earlier splits -> {path}")
    manifest_path = Path(data["output_dir"]) / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote dataset manifest {manifest_path}")


if __name__ == "__main__":
    main()
