from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

from .environment import OrderEnvironment
from .parser import parse_action
from .prompt import build_messages
from .rewards import calculate_reward

Policy = Callable[[list[dict[str, str]], OrderEnvironment], str]


@dataclass
class RolloutResult:
    task_id: str
    prompt: str
    turns: list[dict]
    final_text: str
    termination: str
    verification: dict
    rewards: dict

    def to_dict(self) -> dict:
        return asdict(self)


def run_rollout(task: dict, policy: Policy, weights: dict[str, float], max_turns: int = 6,
                max_tool_calls: int = 5) -> RolloutResult:
    env, turns, final_text, termination = OrderEnvironment(task), [], "", "max_turns"
    for _ in range(max_turns):
        output = policy(build_messages(task["prompt"], turns), env)
        try:
            action = parse_action(output)
        except ValueError as exc:
            env.invalid_actions += 1
            turns.append({"model_output": output, "parse_error": str(exc)})
            termination = "parse_error"
            break
        if action.kind == "final":
            final_text, termination = action.final or "", "final"
            break
        if len(env.history) >= max_tool_calls:
            termination = "max_tool_calls"
            break
        result = env.execute(action.tool or "", action.arguments or {})
        turns.append({"model_output": output, "tool": action.tool, "arguments": action.arguments,
                      "tool_result": result.to_dict(), "execution": env.history[-1]})
    verification = env.verify_task()
    rewards = calculate_reward(verification, weights).to_dict()
    return RolloutResult(task["task_id"], task["prompt"], turns, final_text, termination, verification, rewards)


def append_trace(result: RolloutResult, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
