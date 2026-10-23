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
from grpo_agent.format_reward import format_credit
from grpo_agent.parser import parse_plan
from grpo_agent.prompt import build_training_prompt
from grpo_agent.rewards import calculate_reward
from grpo_agent.run_metrics import RewardLedger, build_step_rows, write_metrics
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


def truncate_prompt_text(text: str, tokenizer: Any, limit: int) -> tuple[str, bool]:
    """Clip a prompt to `limit` tokens when the installed TRL cannot do it itself."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= limit:
        return text, False
    return tokenizer.decode(ids[:limit]), True


def latest_checkpoint(output_dir: str) -> str | None:
    """Newest checkpoint directory, so a run killed by an environment restart can continue.

    Training costs hours and the machine hosting it can go away without warning, so the
    work already done should not be silently repeated.
    """
    candidates = sorted((path for path in Path(output_dir).glob("checkpoint-*")
                         if path.name.split("-")[-1].isdigit()),
                        key=lambda path: int(path.name.split("-")[-1]))
    return str(candidates[-1]) if candidates else None


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
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="continue from a checkpoint: give a path, or omit it for the newest one")
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
    tc = cfg["training"]
    accepted = set(inspect.signature(GRPOConfig.__init__).parameters)
    prompt_limit = int(tc["max_prompt_length"])
    if "max_prompt_length" in accepted:
        prompt_length_mode = "trl_max_prompt_length"
    else:
        prompt_length_mode = "tokenizer_truncation"
    print(f"prompt length control: mode={prompt_length_mode} limit={prompt_limit}")
    if args.dry_run:
        return
    model_cfg = load_yaml(cfg["model_config"])["model"]
    source, tokenizer_source, allow_download = model_source(model_cfg)
    local_only = not allow_download
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=local_only)
    rows = []
    truncated_prompts = 0
    for task in tasks:
        prompt_text = build_training_prompt(task["prompt"])
        if prompt_length_mode == "tokenizer_truncation":
            prompt_text, clipped = truncate_prompt_text(prompt_text, tokenizer, prompt_limit)
            truncated_prompts += int(clipped)
        rows.append({"prompt": prompt_text, "task_json": json.dumps(task)})
    dataset = Dataset.from_list(rows)
    if prompt_length_mode == "tokenizer_truncation":
        print(f"truncated {truncated_prompts}/{len(rows)} prompts to {prompt_limit} tokens")

    format_weights = cfg["reward"].get("format")
    dapo = cfg.get("dapo", {})
    overlong = dapo.get("overlong_reward_shaping", {})
    overlong_enabled = bool(overlong.get("enabled", False))
    soft_limit = int(overlong.get("soft_limit", 10 ** 9))
    penalty_per_token = float(overlong.get("penalty_per_token", 0.0))
    max_penalty = float(overlong.get("max_penalty", 0.0))
    # Imported here rather than at module scope: transformers is a heavy optional dependency
    # and --dry-run has to keep working without it.
    from training.live_metrics import LiveMetricsCallback

    ledger = RewardLedger()
    live = LiveMetricsCallback(ledger, Path(tc["output_dir"]) / "metrics" / "live_metrics.jsonl")

    def reward_func(completions, task_json, **kwargs):
        scores = []
        for completion, task_text in zip(completions, task_json):
            text = completion if isinstance(completion, str) else completion[0].get("content", "")
            score, details = replay_completion(text, json.loads(task_text), cfg["reward"],
                                               cfg["rollout"]["max_tool_calls"])
            credit = format_credit(text, format_weights)
            # DAPO Overlong Reward Shaping: penalize length past a soft limit instead of
            # silently truncating, so the penalty is shaped rather than simply dropped.
            penalty = 0.0
            if overlong_enabled:
                length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                if length > soft_limit:
                    penalty = min((length - soft_limit) * penalty_per_token, max_penalty)
            ledger.add({**details["rewards"], "format_credit": credit, "overlong_penalty": -penalty},
                       details["verification"])
            scores.append(score + credit - penalty)
        ledger.close_batch()
        return scores

    candidates = {
        "output_dir": tc["output_dir"], "learning_rate": tc["learning_rate"], "max_steps": tc["max_steps"],
        "per_device_train_batch_size": tc["per_device_train_batch_size"],
        "gradient_accumulation_steps": tc["gradient_accumulation_steps"], "num_generations": tc["num_generations"],
        "max_completion_length": tc["max_completion_length"],
        "save_steps": tc["save_steps"], "logging_steps": tc["logging_steps"], "temperature": tc["temperature"],
        "seed": cfg["seed"], "report_to": "none",
    }
    if "beta" in tc and "beta" in accepted:
        candidates["beta"] = tc["beta"]
    for key in ("loss_type", "epsilon_high", "mask_truncated_completions"):
        if key in dapo and key in accepted:
            candidates[key] = dapo[key]
    # Generation backend. These keys existed in the YAML but were never forwarded, so
    # vllm_gpu_memory_utilization: 0.15 was dead configuration and generation always ran
    # through plain HF model.generate.
    if tc.get("use_vllm") and "use_vllm" in accepted:
        candidates["use_vllm"] = True
        for key in ("vllm_mode", "vllm_gpu_memory_utilization", "vllm_tensor_parallel_size",
                    "vllm_enable_sleep_mode"):
            if key in tc and key in accepted:
                candidates[key] = tc[key]
    for key in ("use_liger_kernel", "use_transformers_continuous_batching", "torch_compile",
                "gradient_checkpointing", "steps_per_generation"):
        if key in tc and key in accepted:
            candidates[key] = tc[key]
    if prompt_length_mode == "trl_max_prompt_length":
        candidates["max_prompt_length"] = prompt_limit
    unsupported = sorted(key for key in candidates if key not in accepted)
    if unsupported:
        raise RuntimeError(f"Installed GRPOConfig does not support configured fields: {unsupported}")
    train_args = GRPOConfig(**candidates)
    peft_config = None
    if model_cfg.get("use_lora", False):
        lc = model_cfg["lora"]
        peft_config = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                                 target_modules=lc["target_modules"], task_type="CAUSAL_LM")
    sampling = cfg.get("dynamic_sampling", {})
    filter_zero_var = bool(sampling.get("filter_zero_variance_groups", False))
    trainer_kwargs: dict[str, Any] = {}
    trainer_cls = GRPOTrainer
    if filter_zero_var:
        from training.zero_variance_trainer import ZeroVarianceFilteringGRPOTrainer

        trainer_cls = ZeroVarianceFilteringGRPOTrainer
        trainer_kwargs = {"filter_zero_variance_groups": True,
                          "zero_variance_eps": float(sampling.get("zero_variance_eps", 1e-6))}
    # Without this the metrics only exist in the final file, hours after a collapse became
    # visible in the numbers.
    trainer_kwargs["callbacks"] = [live]
    print(f"trainer={trainer_cls.__name__} zero_variance_filter={filter_zero_var} "
          f"beta={candidates.get('beta', 'unset')}")
    trainer = trainer_cls(model=source, reward_funcs=reward_func, args=train_args, train_dataset=dataset,
                          processing_class=tokenizer, peft_config=peft_config, **trainer_kwargs)
    def trainable_signature():
        return {name: (float(param.detach().float().sum().cpu()), float(param.detach().float().norm().cpu()))
                for name, param in trainer.model.named_parameters() if param.requires_grad}
    before = trainable_signature()
    if not before:
        raise RuntimeError("Model has no trainable parameters")
    resume_from = None
    if args.resume:
        # A resumed run can end on a partial batch that no longer divides by num_generations,
        # which TRL rejects, so allow naming an earlier checkpoint explicitly.
        resume_from = latest_checkpoint(tc["output_dir"]) if args.resume == "auto" else args.resume
    if resume_from:
        print(f"resuming from {resume_from}", flush=True)
    result = trainer.train(resume_from_checkpoint=resume_from)
    after = trainable_signature()
    changed = any(before[name] != after[name] for name in before)
    if not changed:
        raise RuntimeError("Trainer ran but no trainable parameter value changed")
    trainer.save_model(tc["output_dir"])
    tokenizer.save_pretrained(tc["output_dir"])
    step_rows = build_step_rows(trainer.state.log_history, ledger)
    metrics_summary = write_metrics(step_rows, Path(tc["output_dir"]) / "metrics" / "train_metrics.jsonl")
    if metrics_summary["missing_metrics"]:
        print("WARNING metrics never observed this run: " + ", ".join(metrics_summary["missing_metrics"]))
        if "kl" in metrics_summary["missing_metrics"]:
            print("  kl is absent because beta=0 disables the reference model and the KL term")
    manifest = {"config": cfg, "train_result": result.metrics, "parameter_update_verified": changed,
                "metrics_summary": metrics_summary,
                "trainable_parameter_tensors": len(before), "global_step": trainer.state.global_step, "trl_api": api,
                "prompt_length_control": {"requested": prompt_limit, "mode": prompt_length_mode,
                                          "truncated_prompts": truncated_prompts}}
    Path(tc["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(tc["output_dir"], "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if trainer.state.global_step < 1:
        raise RuntimeError("Trainer finished without an optimizer step")
    print(f"completed {trainer.state.global_step} real optimizer steps; saved {tc['output_dir']}")


if __name__ == "__main__":
    main()
