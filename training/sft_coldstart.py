from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grpo_agent.config import load_config, load_yaml, model_source
from grpo_agent.policies import oracle_policy
from grpo_agent.prompt import build_messages, build_training_prompt
from grpo_agent.rollout import run_rollout
from grpo_agent.task_generator import read_jsonl


def build_samples(tasks: list[dict], cfg: dict, tokenizer=None) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Cold-start data from the deterministic oracle.

    Two views of the same trajectory are produced: the whole plan (what GRPO
    optimizes) and one step at a time (what the rollout-based evaluator asks for).
    """
    plan_samples: list[dict[str, str]] = []
    step_samples: list[dict[str, str]] = []
    skipped = 0
    for task in tasks:
        result = run_rollout(task, oracle_policy, cfg["reward"], **cfg["rollout"])
        if not result.verification["task_success"]:
            skipped += 1
            continue
        actions = [json.loads(turn["model_output"]) for turn in result.turns]
        actions.append({"final": result.final_text})
        plan_samples.append({"prompt": build_training_prompt(task["prompt"]),
                             "completion": json.dumps(actions, separators=(",", ":"), ensure_ascii=False)})
        if tokenizer is None:
            continue
        for index, turn in enumerate(result.turns):
            step_samples.append({"prompt": _render(build_messages(task["prompt"], result.turns[:index]), tokenizer),
                                 "completion": turn["model_output"]})
        step_samples.append({"prompt": _render(build_messages(task["prompt"], result.turns), tokenizer),
                             "completion": json.dumps({"final": result.final_text}, separators=(",", ":"),
                                                      ensure_ascii=False)})
    samples = plan_samples + step_samples
    stats = {"plan": len(plan_samples), "step": len(step_samples), "kept": len(samples), "skipped": skipped}
    return samples, stats


def _render(messages: list[dict[str, str]], tokenizer) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tasks = read_jsonl(cfg["data"]["train_path"])
    if args.limit is not None:
        tasks = tasks[:args.limit]
    if not tasks:
        raise RuntimeError("Training dataset is empty")
    samples, stats = build_samples(tasks, cfg)
    if not samples:
        raise RuntimeError("Oracle produced no successful trajectories; refusing to train on failures")
    print(f"cold-start SFT plan samples: {stats['plan']} kept, {stats['skipped']} skipped")
    print("sample prompt tail:", samples[0]["prompt"][-160:].replace("\n", " "))
    print("sample completion:", samples[0]["completion"])
    if args.dry_run:
        return

    import torch
    from torch.utils.data import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    model_cfg = load_yaml(cfg["model_config"])["model"]
    source, tokenizer_source, allow_download = model_source(model_cfg)
    common = {"local_files_only": not allow_download,
              "trust_remote_code": bool(model_cfg.get("trust_remote_code", False))}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"), torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype, **common)
    model.config.use_cache = False

    samples, stats = build_samples(tasks, cfg, tokenizer)
    print(f"cold-start SFT samples: plan={stats['plan']} step={stats['step']} total={stats['kept']}")
    max_length = int(cfg["training"]["max_prompt_length"]) + int(cfg["training"]["max_completion_length"])

    class SFTDataset(Dataset):
        def __init__(self, rows: list[dict[str, str]]):
            self.examples: list[dict[str, list[int]]] = []
            for row in rows:
                prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
                completion_ids = tokenizer(row["completion"], add_special_tokens=False)["input_ids"]
                completion_ids = completion_ids + [tokenizer.eos_token_id]
                self.examples.append({"input_ids": (prompt_ids + completion_ids)[:max_length],
                                      "labels": ([-100] * len(prompt_ids) + completion_ids)[:max_length]})

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            return self.examples[index]

    def collate(features: list[dict[str, list[int]]]) -> dict[str, Any]:
        width = max(len(feature["input_ids"]) for feature in features)
        input_ids, labels, attention = [], [], []
        for feature in features:
            pad = width - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [tokenizer.pad_token_id] * pad)
            labels.append(feature["labels"] + [-100] * pad)
            attention.append([1] * len(feature["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attention)}

    output_dir = Path(args.output_dir) if args.output_dir else Path(cfg["training"]["output_dir"]).parent / "sft_coldstart"
    training_args = TrainingArguments(output_dir=str(output_dir), num_train_epochs=args.epochs,
                                      per_device_train_batch_size=args.batch_size, learning_rate=args.learning_rate,
                                      lr_scheduler_type="constant", logging_steps=5, save_strategy="no",
                                      bf16=True, report_to="none", seed=int(cfg["seed"]))
    trainer = Trainer(model=model, args=training_args, train_dataset=SFTDataset(samples), data_collator=collate)

    def signature() -> dict[str, float]:
        return {name: float(param.detach().float().sum().cpu())
                for name, param in trainer.model.named_parameters() if param.requires_grad}

    before = signature()
    if not before:
        raise RuntimeError("Model has no trainable parameters")
    result = trainer.train()
    after = signature()
    changed = any(before[name] != after[name] for name in before)
    if not changed:
        raise RuntimeError("SFT ran but no trainable parameter value changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.config.use_cache = True
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    manifest = {"config": args.config, "samples": stats, "epochs": args.epochs,
                "learning_rate": args.learning_rate, "batch_size": args.batch_size,
                "train_result": result.metrics, "parameter_update_verified": changed,
                "trainable_parameter_tensors": len(before)}
    (output_dir / "sft_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved cold-start model to {output_dir}")
    print("next step: export MODEL_PATH=" + str(output_dir) +
          " && python training/train_grpo.py --config " + args.config)


if __name__ == "__main__":
    main()
