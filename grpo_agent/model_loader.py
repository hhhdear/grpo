from __future__ import annotations

import os
from pathlib import Path

from .config import load_yaml, model_source


def load_model_and_tokenizer(config_path: str):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Missing optional dependency: install transformers and accelerate") from exc
    cfg = load_yaml(config_path)["model"]
    source, tokenizer_source, allow_download = model_source(cfg)
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
        model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype,
                                                     device_map=cfg.get("device_map", "auto"), **common)
    except Exception as exc:
        mode = "offline/local cache" if local_only else "download-enabled"
        raise RuntimeError(f"Failed to load model '{source}' in {mode} mode: {exc}") from exc
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
