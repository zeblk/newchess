from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from .features import NUM_FEATURE_PLANES

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    input_planes: int = NUM_FEATURE_PLANES
    channels: int = 128
    residual_blocks: int = 6
    policy_channels: Optional[int] = None
    value_hidden_size: int = 256
    dropout: float = 0.1


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    name: Optional[str] = "cosine"  # options: cosine, step, none
    t_max: Optional[int] = 20
    min_lr: float = 1e-6
    gamma: float = 0.1
    step_size: int = 10


@dataclass
class TrainingConfig:
    batch_size: int = 256
    epochs: int = 30
    val_split: float = 0.1
    num_workers: int = 4
    max_positions: Optional[int] = None
    amp: bool = True
    grad_clip_norm: Optional[float] = 1.0
    output_dir: Path = Path("artifacts/default")
    checkpoint_every: int = 1
    log_every: int = 50


@dataclass
class LoggingConfig:
    use_tqdm: bool = True


@dataclass
class PathsConfig:
    dataset: Path = Path("data/stockfish_positions.jsonl")


@dataclass
class ExperimentConfig:
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        instance = cls()
        instance.update_from_dict(data)
        return instance

    def update_from_dict(self, data: Dict[str, Any]) -> "ExperimentConfig":
        for field_info in dataclasses.fields(self):
            name = field_info.name
            if name not in data:
                continue
            value = data[name]
            current = getattr(self, name)
            if dataclasses.is_dataclass(current) and isinstance(value, dict):
                _merge_dataclass(current, value)
            else:
                setattr(self, name, value)
        self._post_init_adjustments()
        return self

    def _post_init_adjustments(self) -> None:
        # Ensure paths are Path objects
        self.training.output_dir = Path(self.training.output_dir)
        self.paths.dataset = Path(self.paths.dataset)

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _merge_dataclass(instance: Any, updates: Dict[str, Any]) -> None:
    for field_info in dataclasses.fields(instance):
        key = field_info.name
        if key not in updates:
            continue
        value = updates[key]
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(instance, key, value)


def load_config(path: Path | str) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    if not isinstance(raw, dict):
        raise ValueError("Configuration file must define a mapping at the root level")

    config = ExperimentConfig.from_dict(raw)
    LOGGER.debug("Loaded configuration: %s", config.as_dict())
    return config
