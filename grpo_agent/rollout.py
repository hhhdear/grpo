from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

from .environment import OrderEnvironment
from .format_reward import format_credit
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
                max_tool_calls: int = 5, format_weights: dict[str, float] | None = None) -> RolloutResult:
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
    if format_weights is not None:
        outputs = [turn["model_output"] for turn in turns if "model_output" in turn]
        credit = (sum(format_credit(text, format_weights) for text in outputs) / len(outputs)) if outputs else 0.0
        rewards["format_credit"] = credit
        rewards["total"] += credit
    return RolloutResult(task["task_id"], task["prompt"], turns, final_text, termination, verification, rewards)


def run_rollout_batch(task: dict, policy: Callable[[list], list[str]], weights: dict[str, float], samples: int,
                      max_turns: int = 6, max_tool_calls: int = 5,
                      format_weights: dict[str, float] | None = None) -> dict:
    """Run `samples` independent trajectories for one task, one batched generation per turn.

    Semantically identical to calling `run_rollout` `samples` times: each trajectory owns its
    environment and history, and a trajectory that finishes early simply stops being included
    in later batches.
    """
    envs = [OrderEnvironment(task) for _ in range(samples)]
    histories: list[list[dict]] = [[] for _ in range(samples)]
    outputs: list[list[str]] = [[] for _ in range(samples)]
    terminations = ["max_turns"] * samples
    active = list(range(samples))
    for _ in range(max_turns):
        if not active:
            break
        generated = policy([build_messages(task["prompt"], histories[index]) for index in active])
        still_active = []
        for slot, index in enumerate(active):
            text = generated[slot]
            outputs[index].append(text)
            env = envs[index]
            try:
                action = parse_action(text)
            except ValueError:
                env.invalid_actions += 1
                histories[index].append({"model_output": text, "parse_error": "invalid"})
                terminations[index] = "parse_error"
                continue
            if action.kind == "final":
                terminations[index] = "final"
                continue
            if len(env.history) >= max_tool_calls:
                terminations[index] = "max_tool_calls"
                continue
            env.execute(action.tool or "", action.arguments or {})
            histories[index].append({"model_output": text, "tool": action.tool, "arguments": action.arguments,
                                     "tool_result": env.history[-1]["result"], "execution": env.history[-1]})
            still_active.append(index)
        active = still_active
    verifications = [env.verify_task() for env in envs]
    rewards = []
    for index, verification in enumerate(verifications):
        breakdown = calculate_reward(verification, weights).to_dict()
        if format_weights is not None:
            produced = outputs[index]
            credit = (sum(format_credit(text, format_weights) for text in produced) / len(produced)) if produced else 0.0
            breakdown["format_credit"] = credit
            breakdown["total"] += credit
        rewards.append(breakdown)
    return {"task_id": task["task_id"], "successes": [bool(v["task_success"]) for v in verifications],
            "verifications": verifications, "terminations": terminations, "rewards": rewards,
            "tool_calls": [v["tool_calls"] for v in verifications],
            "invalid_actions": [v["invalid_actions"] for v in verifications]}


def append_trace(result: RolloutResult, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
