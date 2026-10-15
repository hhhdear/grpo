from __future__ import annotations

"""Gate a reward change through a small run before it is allowed into the real arms.

A reward function can always be gamed in a new way, so every change is checked on a short
run against explicit thresholds instead of waiting 500 steps to discover a new hack.

Exit code 0 means every check passed; 1 means at least one failed.
"""
import argparse
import json
import sys
from pathlib import Path

# The first vanilla run (500 steps) scored: task_success last=0.000, 92.4% of steps had
# all-zero variance, entropy pinned at 0.0105. Those are the numbers to beat.
ZERO_VAR_BASELINE = 0.924
ENTROPY_BASELINE = 0.0105


def load(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def column(rows: list[dict], key: str) -> list[float]:
    return [float(row.get(key, 0) or 0) for row in rows]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", help="path to train_metrics.jsonl")
    parser.add_argument("--tail", type=int, default=20, help="how many trailing steps define 'now'")
    parser.add_argument("--min-task-success", type=float, default=0.30)
    parser.add_argument("--max-zero-var", type=float, default=0.50)
    parser.add_argument("--min-entropy", type=float, default=0.05)
    parser.add_argument("--max-invalid-rate", type=float, default=0.50)
    args = parser.parse_args()

    rows = load(args.metrics)
    if not rows:
        print(f"no step rows in {args.metrics}")
        return 1
    count = len(rows)
    tail = rows[-args.tail:]
    half = max(1, count // 2)

    success = column(rows, "reward_term/task_success")
    entropy = column(rows, "entropy")
    zero_var = column(rows, "frac_reward_zero_std")
    invalid_rate = column(rows, "invalid_action_rate")
    invalid_abs = column(rows, "trajectory_invalid_actions")
    calls = column(rows, "trajectory_tool_calls")
    reward = column(rows, "reward")

    tail_success = mean(column(tail, "reward_term/task_success"))
    zero_share = sum(1 for value in zero_var if value >= 1.0) / count
    last_entropy = entropy[-1] if entropy else 0.0
    mean_invalid_rate = mean(invalid_rate)

    # A zero-variance group is only bad news when the group is also wrong. If every
    # candidate solved the task the group carries no gradient either, but the run is
    # healthy and the task is simply too easy. Split the two, otherwise "the model
    # learned it" and "the model is stuck" look identical.
    full_success = max(success) if success else 0.0
    solved_zero = sum(1 for row in rows
                      if float(row.get("frac_reward_zero_std", 0) or 0) >= 1.0
                      and float(row.get("reward_term/task_success", 0) or 0) >= full_success - 1e-6)
    stuck_zero = sum(1 for row in rows
                     if float(row.get("frac_reward_zero_std", 0) or 0) >= 1.0
                     and float(row.get("reward_term/task_success", 0) or 0) < full_success - 1e-6)
    stuck_share = stuck_zero / count

    print(f"steps={count}  reward first={reward[0]:.3f} mid={reward[half]:.3f} last={reward[-1]:.3f}")
    print(f"task_success   first={success[0]:.3f} mid={success[half]:.3f} last={success[-1]:.3f} "
          f"mean={mean(success):.3f} max={max(success):.3f}")
    print(f"entropy        first={entropy[0]:.4f} mid={entropy[half]:.4f} last={last_entropy:.4f}")
    print(f"tool_calls     first={calls[0]:.2f} mid={calls[half]:.2f} last={calls[-1]:.2f} "
          f"mean={mean(calls):.2f} max={max(calls):.2f}")
    print(f"invalid_calls  mean_abs={mean(invalid_abs):.3f} mean_rate={mean_invalid_rate:.3f}")
    print(f"zero-variance  {zero_share:.1%} of steps (previous run: {ZERO_VAR_BASELINE:.1%}) "
          f"-> solved={solved_zero / count:.1%} stuck={stuck_share:.1%}")
    print()

    checks = [
        ("task_success lifted off zero", tail_success >= args.min_task_success,
         f"last {args.tail} steps mean={tail_success:.3f}, want >= {args.min_task_success:.2f}"),
        ("policy is not stuck on zero-variance", stuck_share < args.max_zero_var,
         f"zero-variance AND unsolved steps={stuck_share:.1%}, want < {args.max_zero_var:.0%}"),
        ("entropy not pinned near zero", last_entropy >= args.min_entropy,
         f"last entropy={last_entropy:.4f}, want >= {args.min_entropy:.3f} (was {ENTROPY_BASELINE})"),
        ("invalid calls not flooding", mean_invalid_rate <= args.max_invalid_rate,
         f"mean invalid_action_rate={mean_invalid_rate:.3f}, want <= {args.max_invalid_rate:.2f}"),
    ]
    failed = 0
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failed += not ok
    print()
    if failed:
        print(f"{failed} check(s) failed -> do not release to the 500-step arms")
        return 1
    print("all checks passed -> safe to run the real arms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
