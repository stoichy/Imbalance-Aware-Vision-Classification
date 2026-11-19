"""Utility helpers for reproducibility, configuration loading, device selection, and logging."""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "auto") -> torch.device:
    """Resolve the compute device from a preference string ('auto', 'cpu', 'cuda', 'mps')."""
    pref = preferred.lower().strip()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but no CUDA device is available.")
        return torch.device("cuda")
    if pref == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested, but MPS is unavailable.")
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device specification: {preferred!r}")


def setup_logging(
    level: str = "INFO",
    log_file: str | Path | None = None,
    name: str = "imbalanced_vision",
) -> logging.Logger:
    """Configure structured console (and optional file) logging."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand `${VAR:-default}` placeholders in strings inside loaded YAML."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            lambda m: os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else ""),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file and expand environment variable placeholders."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML configuration not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in YAML file: {yaml_path}")
    return _expand_env_vars(data)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of `base` updated recursively with `override`."""
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def save_json(data: Any, path: str | Path) -> None:
    """Serialize data as formatted JSON, creating parent directories if needed."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def load_json(path: str | Path) -> Any:
    """Load JSON data from disk."""
    in_path = Path(path)
    if not in_path.exists():
        raise FileNotFoundError(f"JSON file not found: {in_path}")
    with in_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total, trainable, and frozen parameter counts for a PyTorch module."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }

