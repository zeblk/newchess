from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import chess
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from . import features as feature_utils
from .config import ExperimentConfig
from .move_encoding import legal_move_mask, move_to_index


@dataclass
class PositionRecord:
    fen: str
    best_move: str


class PositionDataset(Dataset):
    """Dataset for reading sharded JSONL files with O(1) random access.
    
    Builds an in-memory index of (file_path, byte_offset) for every line
    in all .jsonl files within the given directory.
    """

    def __init__(
        self,
        data_path: Path | str,
        max_positions: Optional[int] = None,
        include_legal_mask: bool = False,
    ) -> None:
        self.path = Path(data_path)
        self.include_legal_mask = include_legal_mask
        self.index: List[tuple[Path, int]] = []
        
        if self.path.is_dir():
            self._build_index_from_dir(self.path, max_positions)
        elif self.path.is_file():
            self._build_index_from_file(self.path, max_positions)
        else:
            raise FileNotFoundError(f"Dataset path not found: {self.path}")

    def _build_index_from_dir(self, directory: Path, max_positions: Optional[int]) -> None:
        files = sorted(directory.glob("*.jsonl"))
        if not files:
            raise ValueError(f"No .jsonl files found in {directory}")
        
        for file_path in files:
            if max_positions is not None and len(self.index) >= max_positions:
                break
            self._build_index_from_file(file_path, max_positions)

    def _build_index_from_file(self, file_path: Path, max_positions: Optional[int]) -> None:
        with file_path.open("rb") as f:
            while True:
                if max_positions is not None and len(self.index) >= max_positions:
                    break
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                # Optional: verify line is valid JSON here or just assume it is
                if line.strip():
                    self.index.append((file_path, offset))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        file_path, offset = self.index[index]
        with file_path.open("r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
            payload = json.loads(line)

        fen = payload["fen"]
        best_move = payload["best_move"]
        
        board = chess.Board(fen)
        features_tensor = feature_utils.board_to_tensor(board)
        target_index = move_to_index(best_move)
        
        sample: Dict[str, torch.Tensor | str] = {
            "features": features_tensor,
            "target": torch.tensor(target_index, dtype=torch.long),
            "fen": fen,
        }
        if self.include_legal_mask:
            sample["legal_mask"] = legal_move_mask(board)
        return sample


def default_collate(batch: List[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[str]]:
    features = torch.stack([sample["features"] for sample in batch])
    targets = torch.stack([sample["target"] for sample in batch]).view(-1)
    fens = [str(sample["fen"]) for sample in batch]
    collated: Dict[str, torch.Tensor | List[str]] = {
        "features": features,
        "targets": targets,
        "fens": fens,
    }
    if "legal_mask" in batch[0]:
        masks = torch.stack([sample["legal_mask"] for sample in batch])
        collated["legal_mask"] = masks
    return collated


@dataclass
class RLPositionRecord:
    fen: str
    action: str
    value: float


class RLDataset(Dataset):
    """Dataset for reading sharded JSONL files from self-play."""

    def __init__(
        self,
        data_path: Path | str,
        max_positions: Optional[int] = None,
        include_legal_mask: bool = False,
    ) -> None:
        self.path = Path(data_path)
        self.include_legal_mask = include_legal_mask
        self.index: List[tuple[Path, int]] = []
        
        if self.path.is_dir():
            self._build_index_from_dir(self.path, max_positions)
        elif self.path.is_file():
            self._build_index_from_file(self.path, max_positions)
        else:
            # It's possible the file doesn't exist yet if we haven't generated data
            LOGGER.warning(f"Dataset path not found: {self.path}. Assuming empty dataset.")

    def _build_index_from_dir(self, directory: Path, max_positions: Optional[int]) -> None:
        files = sorted(directory.glob("*.jsonl"))
        for file_path in files:
            if max_positions is not None and len(self.index) >= max_positions:
                break
            self._build_index_from_file(file_path, max_positions)

    def _build_index_from_file(self, file_path: Path, max_positions: Optional[int]) -> None:
        with file_path.open("rb") as f:
            while True:
                if max_positions is not None and len(self.index) >= max_positions:
                    break
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    self.index.append((file_path, offset))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str | float]:
        file_path, offset = self.index[index]
        with file_path.open("r", encoding="utf-8") as f:
            f.seek(offset)
            line = f.readline()
            payload = json.loads(line)

        fen = payload["fen"]
        action = payload["action"]
        value = payload["value"]
        
        board = chess.Board(fen)
        features_tensor = feature_utils.board_to_tensor(board)
        action_index = move_to_index(action)
        
        sample: Dict[str, torch.Tensor | str | float] = {
            "features": features_tensor,
            "action_target": torch.tensor(action_index, dtype=torch.long),
            "value_target": torch.tensor(value, dtype=torch.float32),
            "fen": fen,
        }
        if self.include_legal_mask:
            sample["legal_mask"] = legal_move_mask(board)
        return sample


def rl_collate(batch: List[Dict[str, torch.Tensor | str | float]]) -> Dict[str, torch.Tensor | List[str]]:
    features = torch.stack([sample["features"] for sample in batch])
    action_targets = torch.stack([sample["action_target"] for sample in batch]).view(-1)
    value_targets = torch.stack([sample["value_target"] for sample in batch]).view(-1)
    fens = [str(sample["fen"]) for sample in batch]
    
    collated: Dict[str, torch.Tensor | List[str]] = {
        "features": features,
        "action_targets": action_targets,
        "value_targets": value_targets,
        "fens": fens,
    }
    if "legal_mask" in batch[0]:
        masks = torch.stack([sample["legal_mask"] for sample in batch])
        collated["legal_mask"] = masks
    return collated


def create_dataloaders(
    config: ExperimentConfig,
    include_legal_mask: bool = False,
    is_rl: bool = False,
) -> tuple[DataLoader, Optional[DataLoader]]:
    
    if is_rl:
        dataset_cls = RLDataset
        collate_fn = rl_collate
    else:
        dataset_cls = PositionDataset
        collate_fn = default_collate

    dataset = dataset_cls(
        data_path=config.paths.dataset,
        max_positions=config.training.max_positions,
        include_legal_mask=include_legal_mask,
    )

    if len(dataset) == 0:
        return DataLoader(dataset), None

    val_fraction = config.training.val_split
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("training.val_split must be in the range [0.0, 1.0)")

    if val_fraction == 0.0:
        train_dataset: Dataset = dataset
        val_dataset: Optional[Dataset] = None
    else:
        val_size = max(1, int(len(dataset) * val_fraction))
        train_size = len(dataset) - val_size
        if train_size <= 0:
             # Fallback for very small datasets
             train_dataset = dataset
             val_dataset = None
        else:
            generator = torch.Generator().manual_seed(config.seed)
            train_subset, val_subset = random_split(
                dataset,
                lengths=[train_size, val_size],
                generator=generator,
            )
            train_dataset = train_subset
            val_dataset = val_subset

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader: Optional[DataLoader]
    if val_dataset is None:
        val_loader = None
    else:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return train_loader, val_loader
