from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {source}")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = load_yaml(path)
    required = {"seed", "data", "rollout", "training", "reward", "paths"}
    missing = sorted(required - cfg.keys())
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}")
    output_root = os.environ.get("GRPO_OUTPUT_ROOT")
    if output_root:
        root = Path(output_root)
        cfg["training"]["output_dir"] = str(root / "checkpoints" / cfg.get("experiment", "run"))
        cfg["paths"]["trace_dir"] = str(root / "logs" / "traces")
        cfg["paths"]["evaluation_dir"] = str(root / "logs" / "evaluation")
    return cfg


def model_source(config: dict[str, Any]) -> tuple[str, str, bool]:
    """Resolve model/tokenizer locations without machine-specific code edits."""
    local = os.environ.get("MODEL_PATH") or config.get("local_path")
    model_id = os.environ.get("MODEL_ID") or config.get("name")
    if not local and not model_id:
        raise ValueError("Set MODEL_PATH, MODEL_ID, model.local_path, or model.name")
    source = local or model_id
    tokenizer = os.environ.get("TOKENIZER_PATH") or config.get("tokenizer_path") or source
    flag = os.environ.get("ALLOW_MODEL_DOWNLOAD")
    allow_download = bool(config.get("allow_download", False)) if flag is None else flag.lower() in {
        "1", "true", "yes", "on"
    }
    return str(source), str(tokenizer), allow_download
