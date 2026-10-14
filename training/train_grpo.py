from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grpo_agent.config import load_config, load_yaml, model_source
from grpo_agent.environment import OrderEnvironment
from grpo_agent.parser import parse_plan
from grpo_agent.prompt import build_training_prompt
from grpo_agent.rewards import calculate_reward
from grpo_agent.task_generator import read_jsonl


def replay_completion(completion: str, task: dict, weights: dict, max_calls: int) -> tuple[float, dict]:
    env = OrderEnvironment(task)
    try:
        actions = parse_plan(completion)
        for action in actions:
            if action.kind == "final":
                break
            if len(env.history) >= max_calls:
                env.invalid_actions += 1
                break
            env.execute(action.tool or "", action.arguments or {})
    except ValueError:
        env.invalid_actions += 1
    verification = env.verify_task()
    breakdown = calculate_reward(verification, weights).to_dict()
    return breakdown["total"], {"verification": verification, "rewards": breakdown, "history": env.history}


def dependency_api():
    try:
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("GRPO dependencies are missing; install requirements.txt in an isolated environment") from exc
    return Dataset, LoraConfig, AutoModelForCausalLM, AutoTokenizer, GRPOConfig, GRPOTrainer


def validate_api(GRPOConfig, GRPOTrainer) -> dict[str, list[str]]:
    config_params = set(inspect.signature(GRPOConfig.__init__).parameters)
    trainer_params = set(inspect.signature(GRPOTrainer.__init__).parameters)
    required_config = {"output_dir", "learning_rate", "max_steps", "num_generations"}
    required_trainer = {"model", "reward_funcs", "args", "train_dataset"}
    missing = {"GRPOConfig": sorted(required_config - config_params), "GRPOTrainer": sorted(required_trainer - trainer_params)}
    if any(missing.values()):
        raise RuntimeError(f"Installed TRL API is incompatible: {missing}")
    return {"GRPOConfig": sorted(config_params), "GRPOTrainer": sorted(trainer_params)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    tasks = read_jsonl(cfg["data"]["train_path"])
    if not tasks:
        raise RuntimeError("Training dataset is empty")
    sample_score, _ = replay_completion("[{\"final\":\"no action\"}]", tasks[0], cfg["reward"], cfg["rollout"]["max_tool_calls"])
    print(f"validated {len(tasks)} tasks and grounded reward replay (sample={sample_score:.3f})")
    Dataset, LoraConfig, AutoModel, AutoTokenizer, GRPOConfig, GRPOTrainer = dependency_api()
    api = validate_api(GRPOConfig, GRPOTrainer)
    print("installed TRL API is compatible with the baseline constructor")
    if args.dry_run:
        return
    model_cfg = load_yaml(cfg["model_config"])["model"]
    source, tokenizer_source, allow_download = model_source(model_cfg)
    local_only = not allow_download
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=local_only)
    dataset = Dataset.from_list([{"prompt": build_training_prompt(t["prompt"]), "task_json": json.dumps(t)} for t in tasks])

    def reward_func(completions, task_json, **kwargs):
        scores = []
        for completion, task_text in zip(completions, task_json):
            text = completion if isinstance(completion, str) else completion[0].get("content", "")
            score, _ = replay_completion(text, json.loads(task_text), cfg["reward"], cfg["rollout"]["max_tool_calls"])
            scores.append(score)
        return scores

    tc = cfg["training"]
    accepted = set(inspect.signature(GRPOConfig.__init__).parameters)
    candidates = {
        "output_dir": tc["output_dir"], "learning_rate": tc["learning_rate"], "max_steps": tc["max_steps"],
        "per_device_train_batch_size": tc["per_device_train_batch_size"],
        "gradient_accumulation_steps": tc["gradient_accumulation_steps"], "num_generations": tc["num_generations"],
        "max_prompt_length": tc["max_prompt_length"], "max_completion_length": tc["max_completion_length"],
        "save_steps": tc["save_steps"], "logging_steps": tc["logging_steps"], "temperature": tc["temperature"],
        "seed": cfg["seed"], "report_to": "none",
    }
    unsupported = sorted(key for key in candidates if key not in accepted)
    if unsupported:
        raise RuntimeError(f"Installed GRPOConfig does not support configured fields: {unsupported}")
    train_args = GRPOConfig(**candidates)
    peft_config = None
    if model_cfg.get("use_lora", False):
        lc = model_cfg["lora"]
        peft_config = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                 target_modules=lc["target_modules"], task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=source, reward_funcs=reward_func, args=train_args, train_dataset=dataset,
                          processing_class=tokenizer, peft_config=peft_config)
    def trainable_signature():
        return {name: (float(param.detach().float().sum().cpu()), float(param.detach().float().norm().cpu()))
                for name, param in trainer.model.named_parameters() if param.requires_grad}
    before = trainable_signature()
    if not before:
        raise RuntimeError("Model has no trainable parameters")
    result = trainer.train()
    after = trainable_signature()
    changed = any(before[name] != after[name] for name in before)
    if not changed:
        raise RuntimeError("Trainer ran but no trainable parameter value changed")
    trainer.save_model(tc["output_dir"])
    tokenizer.save_pretrained(tc["output_dir"])
    manifest = {"config": cfg, "train_result": result.metrics, "parameter_update_verified": changed,
                "trainable_parameter_tensors": len(before), "global_step": trainer.state.global_step, "trl_api": api}
    Path(tc["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(tc["output_dir"], "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if trainer.state.global_step < 1:
        raise RuntimeError("Trainer finished without an optimizer step")
    print(f"completed {trainer.state.global_step} real optimizer steps; saved {tc['output_dir']}")


if __name__ == "__main__":
    main()
