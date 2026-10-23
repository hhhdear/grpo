from __future__ import annotations

"""Stream per-step metrics to the console and to disk while a run is still training.

A collapse is only cheap to react to while it is happening: the canonical metrics file is
written after `max_steps`, by which point the GPU hours are already spent. Every logged step
is therefore printed as one compact line and appended to `metrics/live_metrics.jsonl`, which
`scripts/watch_metrics.py` tails while the run is in flight.

The grounded reward terms live in `RewardLedger` and are only aligned with TRL's log entry at
the end of a run, so the callback pairs the newest ledger batch with the newest log entry.
TRL calls the reward function once per optimizer step, which makes that pairing 1:1.
"""
import json
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from grpo_agent.run_metrics import REWARD_TERM_KEYS, TRL_FIELD_MAP, RewardLedger

LEDGER_FIELDS = (("tool_calls", "trajectory_tool_calls"),
                 ("invalid_actions", "trajectory_invalid_actions"),
                 ("invalid_action_rate", "invalid_action_rate"),
                 ("group_size", "group_size"))


def _fmt(value: float | None, digits: int, width: int, plus: bool = False) -> str:
    """Render a metric, or a dash when the run has not reported it yet.

    A missing metric must never look like a zero: "entropy is absent" and "entropy is 0.0"
    are very different situations and only the second one means the policy is dead.
    """
    if value is None:
        return "-".rjust(width)
    spec = f"{{:{'+' if plus else ''}.{digits}f}}"
    return spec.format(value).rjust(width)


def _num(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if value is not None else None


def format_step_line(row: dict[str, Any]) -> str:
    return (f"[step {int(row.get('step', 0)):>4}]"
            f" reward={_fmt(_num(row, 'reward'), 3, 7, plus=True)}"
            f" succ={_fmt(_num(row, 'reward_term/task_success'), 2, 5)}"
            f" calls={_fmt(_num(row, 'trajectory_tool_calls'), 2, 5)}"
            f" inv={_fmt(_num(row, 'invalid_action_rate'), 2, 5)}"
            f" zero={_fmt(_num(row, 'frac_reward_zero_std'), 2, 5)}"
            f" ent={_fmt(_num(row, 'entropy'), 3, 6)}"
            f" len={_fmt(_num(row, 'completion_mean_length'), 0, 5)}"
            f" {float(row.get('wall_time', 0) or 0):.0f}s")


class LiveMetricsCallback(TrainerCallback):
    """Append every logged step to a jsonl stream and echo it to stdout."""

    def __init__(self, ledger: RewardLedger, path: str | Path) -> None:
        self.ledger = ledger
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # "w", not "a": one file belongs to one run. Appending would mix the previous run's
        # steps into this one's window and the watcher would judge stale numbers.
        self.handle = self.path.open("w", encoding="utf-8")
        self.started = time.time()
        self.rows: list[dict[str, Any]] = []

    def on_log(self, args: Any, state: Any, control: Any, logs: dict | None = None, **kwargs: Any) -> None:
        logs = logs or {}
        row: dict[str, Any] = {"step": int(state.global_step),
                               "wall_time": round(time.time() - self.started, 1)}
        for source, target in TRL_FIELD_MAP.items():
            value = logs.get(source)
            if value is not None:
                try:
                    row[target] = float(value)
                except (TypeError, ValueError):
                    continue
        batch = self.ledger.step(len(self.ledger.batches) - 1)
        for key in REWARD_TERM_KEYS:
            if key in batch:
                row[f"reward_term/{key}"] = float(batch[key])
        for source, target in LEDGER_FIELDS:
            if source in batch:
                row[target] = float(batch[source])
        self.rows.append(row)
        self.handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.handle.flush()
        print(format_step_line(row), flush=True)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        print(f"[live] {len(self.rows)} step rows -> {self.path}", flush=True)
        self.handle.close()
