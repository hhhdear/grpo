from __future__ import annotations

import json
import os
from pathlib import Path

from .config import load_yaml, model_source


def _is_adapter_directory(path: str | Path) -> bool:
    return (Path(path) / "adapter_config.json").is_file()


def load_model_and_tokenizer(config_path: str, model_dir: str | None = None):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing optional dependency: install transformers and accelerate") from exc
    cfg = load_yaml(config_path)["model"]
    source, tokenizer_source, allow_download = model_source(cfg)
    if model_dir:
        # Evaluate the artifact that was actually trained, not the base model it started from.
        source = tokenizer_source = model_dir
    if (cfg.get("local_path") or os.environ.get("MODEL_PATH")) and not Path(source).is_dir():
        raise FileNotFoundError(f"Configured local model directory does not exist: {source}")
    local_only = not allow_download
    dtype_name = cfg.get("dtype", "float32")
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    common = {"local_files_only": local_only, "trust_remote_code": bool(cfg.get("trust_remote_code", False))}
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **common)
        if _is_adapter_directory(source):
            # A GRPO run saves a LoRA adapter, not a full checkpoint.
            from peft import PeftModel

            adapter_config = json.loads((Path(source) / "adapter_config.json").read_text(encoding="utf-8"))
            base = adapter_config.get("base_model_name_or_path")
            if not base:
                raise ValueError(f"Adapter config in {source} has no base_model_name_or_path")
            base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype,
                                                              device_map=cfg.get("device_map", "auto"), **common)
            model = PeftModel.from_pretrained(base_model, source)
        else:
            model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype,
                                                         device_map=cfg.get("device_map", "auto"), **common)
    except Exception as exc:
        mode = "offline/local cache" if local_only else "download-enabled"
        raise RuntimeError(f"Failed to load model '{source}' in {mode} mode: {exc}") from exc
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
