from __future__ import annotations

"""Run the six ablation arms end to end: train, then evaluate each checkpoint.

Arms already finished are skipped by looking for their metrics file, so the script can be
restarted after an interruption without redoing hours of training.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ARMS = ("vanilla", "clip_higher", "token_level", "overlong", "dynamic_sampling", "dapo_full")
ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], log_path: Path, extra_env: dict[str, str] | None = None,
        echo: bool = True) -> None:
    """Run a command, keeping the log file and echoing stdout to the console.

    An arm trains for hours and the only cheap moment to react to a failing run is while it
    is happening, so per-step metrics are teed to the terminal instead of being captured.
    """
    env = {**os.environ, **(extra_env or {})}
    print(f">>> {' '.join(command)} (log: {log_path})", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            if echo:
                print(line, end="", flush=True)
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"command failed ({returncode}): {' '.join(command)}; see {log_path}")


def output_dir(config: str) -> str:
    import yaml

    return yaml.safe_load(Path(config).read_text(encoding="utf-8"))["training"]["output_dir"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N test tasks")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="retrain even when a metrics file exists; required after any "
                             "change to the training configuration, otherwise old results "
                             "from a different configuration are silently reused")
    parser.add_argument("--quiet", action="store_true",
                        help="keep child output in the log files instead of echoing it")
    args = parser.parse_args()

    for arm in args.arms:
        config = f"configs/ablations/{arm}.yaml"
        out_dir = ROOT / output_dir(config)
        metrics_file = out_dir / "metrics" / "train_metrics.jsonl"
        if not args.skip_training and (args.force or not metrics_file.exists()):
            run([sys.executable, "training/train_grpo.py", "--config", config],
                ROOT / "logs" / "ablation" / f"train_{arm}.log", echo=not args.quiet)
        else:
            print(f">>> skipping training for {arm}: {metrics_file} exists", flush=True)
        command = [sys.executable, "evaluation/evaluate.py", "--config", config,
                   "--samples", str(args.samples), "--tag", f"ablation_{arm}"]
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        run(command, ROOT / "logs" / "ablation" / f"eval_{arm}.log",
            {"MODEL_PATH": str(out_dir)}, echo=not args.quiet)
    print("all requested arms finished", flush=True)


if __name__ == "__main__":
    main()
