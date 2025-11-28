from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preference: str) -> torch.device:
    preference = preference.lower()
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA devices are available")
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device preference '{preference}'")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    metrics: Dict[str, Any],
    config: Dict[str, Any],
    filename: str = "checkpoint.pt",
) -> Path:
    ensure_dir(output_dir)
    checkpoint_path = output_dir / filename
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": config,
    }
    torch.save(payload, checkpoint_path)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    LOGGER.info("Checkpoint saved to %s", checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    map_location: Optional[torch.device] = None,
) -> Dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state"):
        scheduler.load_state_dict(payload["scheduler_state"])
    LOGGER.info("Loaded checkpoint from %s", checkpoint_path)
    return payload


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def summarize_config(config: Dict[str, Any]) -> str:
    return json.dumps(config, indent=2, default=str)
