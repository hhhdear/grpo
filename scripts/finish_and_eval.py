from __future__ import annotations

"""Wait for arms to finish training, then evaluate them one after another.

Evaluation uses the same GPU as training, so running it right after each arm would slow
every other arm still training. This defers evaluation until the listed arms are all done.

An arm counts as finished only when its run manifest is newer than its live metrics file:
a re-run arm still carries the previous run's manifest, which would otherwise look finished.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def finished(arm: str) -> bool:
    out = ROOT / "checkpoints" / f"ablation_{arm}"
    manifest = out / "run_manifest.json"
    live = out / "metrics" / "live_metrics.jsonl"
    return live.exists() and manifest.exists() and manifest.stat().st_mtime >= live.stat().st_mtime


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--poll", type=float, default=60.0)
    args = parser.parse_args()

    pending = list(args.arms)
    while pending:
        done = [arm for arm in pending if finished(arm)]
        for arm in done:
            print(f"[{time.strftime('%H:%M:%S')}] {arm}: training finished", flush=True)
        pending = [arm for arm in pending if arm not in done]
        if pending:
            print(f"[{time.strftime('%H:%M:%S')}] waiting for: {' '.join(pending)}", flush=True)
            time.sleep(args.poll)

    failures = []
    for arm in args.arms:
        command = [sys.executable, "evaluation/evaluate.py", "--config", f"configs/ablations/{arm}.yaml",
                   "--samples", str(args.samples), "--tag", f"ablation_{arm}"]
        if args.limit:
            command += ["--limit", str(args.limit)]
        print(f"[{time.strftime('%H:%M:%S')}] >>> {' '.join(command)}", flush=True)
        env = {**os.environ, "MODEL_PATH": str(ROOT / "checkpoints" / f"ablation_{arm}")}
        result = subprocess.run(command, cwd=ROOT, env=env)
        print(f"[{time.strftime('%H:%M:%S')}] {arm}: exit={result.returncode}", flush=True)
        if result.returncode != 0:
            failures.append(arm)
    if failures:
        print("evaluation failed for: " + ", ".join(failures), flush=True)
        return 1
    print("all evaluations finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
