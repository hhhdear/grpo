from __future__ import annotations

"""Tail live metrics streams and shout the moment a run starts to go wrong.

The training script prints one line per logged step, but nobody can stare at a terminal for
hours. This watches the same stream on disk and applies the failure modes that are cheap to
fix early and expensive to discover late:

  * entropy collapse          - the policy stopped exploring, GRPO has no gradient left
  * zero-variance while wrong - the group is identical *and* unsolved: no signal, no progress
  * nothing ever solved       - the run is not learning the task at all
  * invalid-call flood        - usually a reward the model learned to game
  * stalled stream            - the process died (OOM) or is wedged

Exit code 0 means the run(s) finished without an alert; 1 means at least one alert fired.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Thresholds mirror scripts/check_smoke.py so "the live watcher complained" and "the gate
# failed after the run" mean the same thing. The first vanilla run ended at entropy 0.0105
# with 92% zero-variance steps, which is what these numbers are calibrated against: entropy
# below 0.05 means the policy has effectively stopped exploring even before it hits 0.
DEFAULT_MIN_ENTROPY = 0.05
DEFAULT_MAX_ZERO_VAR = 0.90
DEFAULT_MAX_ZERO_VAR_MEAN = 0.80
DEFAULT_MAX_INVALID_RATE = 0.50


def load_new(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    """Read rows appended since `offset`, stopping at a half-written line."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return [], offset
    rows: list[dict[str, Any]] = []
    consumed = 0
    for line in lines[offset:]:
        if not line.strip():
            consumed += 1
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # torn write; retry on the next poll
        consumed += 1
    return rows, offset + consumed


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


class Run:
    """One metrics stream plus the alert state kept for it."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.name = self.path.parent.parent.name or self.path.stem
        self.offset = 0
        self.rows: list[dict[str, Any]] = []
        self.last_new = time.time()
        self.last_alert: dict[str, float] = {}
        self.done = False

    def poll(self) -> list[dict[str, Any]]:
        rows, self.offset = load_new(self.path, self.offset)
        if rows:
            self.last_new = time.time()
            self.rows.extend(rows)
        return rows

    def arm(self) -> str:
        return self.name.removeprefix("ablation_")

    def stop_command(self) -> str:
        return f'pkill -f "train_grpo.py --config configs/ablations/{self.arm()}.yaml"'

    def cooling(self, rule: str, cooldown: float) -> bool:
        return time.time() - self.last_alert.get(rule, 0.0) < cooldown


def evaluate(run: Run, args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """Return (severity, rule, message) for every condition currently firing."""
    rows = run.rows
    if not rows:
        return []
    tail = rows[-args.window:]
    alerts: list[tuple[str, str, str]] = []
    steps = len(rows)
    success = [float(row.get("reward_term/task_success", 0) or 0) for row in rows]
    best = max(success) if success else 0.0
    tail_success = mean(tail, "reward_term/task_success")
    entropy = mean(tail, "entropy")
    invalid = mean(tail, "invalid_action_rate")
    step_time = mean(tail, "step_time")
    zero_tail = [float(row.get("frac_reward_zero_std", 0) or 0) for row in tail]
    # Zero variance is only fatal when the group is also unsolved: an identical group that
    # all succeeded means the task is easy, not that learning stopped.
    stuck = sum(1 for value, score in zip(zero_tail, [float(row.get("reward_term/task_success", 0) or 0)
                                                      for row in tail])
                if value >= 1.0 and score < best - 1e-6) / len(tail)

    if steps >= args.min_steps and entropy is not None and entropy < args.min_entropy:
        alerts.append(("ALERT", "entropy_collapse",
                       f"entropy {entropy:.4f} over last {len(tail)} steps < {args.min_entropy} "
                       f"- policy stopped exploring, GRPO has no signal left"))
    if steps >= args.progress_steps and best <= 0.0:
        alerts.append(("ALERT", "never_solved",
                       f"task_success is still 0.0 after {steps} steps - the run is not learning the task"))
    zero_mean = mean(tail, "frac_reward_zero_std")
    if steps >= args.min_steps and zero_mean is not None and zero_mean > args.max_zero_var_mean:
        alerts.append(("WARN", "thin_signal",
                       f"{zero_mean:.0%} of rewards had zero variance over the last {len(tail)} steps "
                       f"- GRPO is running on almost no gradient signal"))
    if steps >= args.min_steps and stuck > args.max_zero_var and tail_success is not None:
        alerts.append(("ALERT", "zero_variance_stuck",
                       f"{stuck:.0%} of last {len(tail)} steps had zero reward variance AND stayed "
                       f"unsolved (best={best:.2f}) - no gradient and no progress"))
    if steps >= args.min_steps and invalid is not None and invalid > args.max_invalid_rate:
        alerts.append(("ALERT", "invalid_flood",
                       f"invalid_action_rate {invalid:.2f} > {args.max_invalid_rate} over last "
                       f"{len(tail)} steps - the model is spamming unusable calls"))
    if step_time is not None and step_time > args.max_step_time:
        alerts.append(("WARN", "slow_steps", f"{step_time:.1f}s/step over last {len(tail)} steps"))
    if steps >= args.progress_steps:
        early = rows[: max(1, len(rows) // 3)]
        early_reward, tail_reward = mean(early, "reward"), mean(tail, "reward")
        if early_reward is not None and tail_reward is not None and tail_reward < early_reward:
            alerts.append(("WARN", "reward_not_improving",
                           f"reward {tail_reward:.3f} now vs {early_reward:.3f} early - going backwards"))
    if not run.done and time.time() - run.last_new > args.stall:
        alerts.append(("ALERT", "stalled",
                       f"no new step for {time.time() - run.last_new:.0f}s - process likely died (OOM?) "
                       f"or is wedged"))
    return alerts


def line_for(run: Run, args: argparse.Namespace) -> str:
    rows = run.rows
    if not rows:
        return f"[{run.name}] waiting for first step..."
    tail = rows[-args.window:]
    last = rows[-1]
    return (f"[{time.strftime('%H:%M:%S')}] {run.name} step={int(last.get('step', 0)):>4}"
            f" reward={fmt(mean(tail, 'reward'), 3, plus=True)}"
            f" succ={fmt(mean(tail, 'reward_term/task_success'), 2)}"
            f" best={fmt(max((float(r.get('reward_term/task_success', 0) or 0) for r in rows)), 2)}"
            f" calls={fmt(mean(tail, 'trajectory_tool_calls'), 2)}"
            f" inv={fmt(mean(tail, 'invalid_action_rate'), 2)}"
            f" zero={fmt(mean(tail, 'frac_reward_zero_std'), 2)}"
            f" ent={fmt(mean(tail, 'entropy'), 3)}"
            f" ({len(rows)} rows)")


def fmt(value: float | None, digits: int, plus: bool = False) -> str:
    return "-" if value is None else f"{value:{'+' if plus else ''}.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="live_metrics.jsonl files to watch")
    parser.add_argument("--interval", type=float, default=15.0, help="seconds between polls")
    parser.add_argument("--window", type=int, default=20, help="trailing steps that define 'now'")
    parser.add_argument("--min-steps", type=int, default=30, help="steps before judging a run")
    parser.add_argument("--progress-steps", type=int, default=100, help="steps before expecting success")
    parser.add_argument("--min-entropy", type=float, default=DEFAULT_MIN_ENTROPY)
    parser.add_argument("--max-zero-var", type=float, default=DEFAULT_MAX_ZERO_VAR)
    parser.add_argument("--max-zero-var-mean", type=float, default=DEFAULT_MAX_ZERO_VAR_MEAN)
    parser.add_argument("--max-invalid-rate", type=float, default=DEFAULT_MAX_INVALID_RATE)
    parser.add_argument("--max-step-time", type=float, default=60.0)
    parser.add_argument("--stall", type=float, default=600.0, help="seconds without a new step")
    parser.add_argument("--cooldown", type=float, default=180.0, help="seconds between repeat alerts")
    parser.add_argument("--kill", action="store_true",
                        help="stop the training process when a severe alert first fires")
    parser.add_argument("--replay", action="store_true",
                        help="judge the rows already on disk once, then exit (no waiting)")
    args = parser.parse_args()

    runs = [Run(path) for path in args.paths]
    alerted = 0
    poll = 0
    while True:
        poll += 1
        for run in runs:
            new = run.poll()
            # A restarted arm still carries the previous run's manifest, so "the file exists"
            # is not evidence that this run finished. Training writes the manifest after the
            # live stream is closed, so compare timestamps instead of just checking existence.
            manifest = run.path.parent.parent / "run_manifest.json"
            if (run.rows and not run.done and manifest.exists()
                    and manifest.stat().st_mtime >= run.path.stat().st_mtime):
                run.done = True
                print(f"[{run.name}] finished ({len(run.rows)} steps)", flush=True)
            if new or poll == 1:
                print(line_for(run, args), flush=True)
            for severity, rule, message in evaluate(run, args):
                if run.cooling(rule, args.cooldown):
                    continue
                run.last_alert[rule] = time.time()
                alerted += 1
                print(f"[{severity} {time.strftime('%H:%M:%S')}] {run.name}: {message}", flush=True)
                if severity == "ALERT":
                    if args.kill and not run.done:
                        subprocess.run(["bash", "-lc", run.stop_command()], check=False)
                        print(f"  stopped: {run.stop_command()}", flush=True)
                    else:
                        print(f"  to stop it: {run.stop_command()}", flush=True)
        if args.replay:
            break
        if all(run.done for run in runs):
            print("all watched runs finished", flush=True)
            break
        time.sleep(args.interval)
    return 1 if alerted else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nwatcher stopped", flush=True)
        sys.exit(130)
