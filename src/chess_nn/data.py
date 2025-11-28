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
    """Dataset wrapping FEN positions paired with Stockfish moves."""

    def __init__(
        self,
        data_path: Path | str,
        max_positions: Optional[int] = None,
        include_legal_mask: bool = False,
    ) -> None:
        self.path = Path(data_path)
        self.include_legal_mask = include_legal_mask
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        self.records = self._load_records(self.path, max_positions)

    @staticmethod
    def _load_records(path: Path, max_positions: Optional[int]) -> List[PositionRecord]:
        records: List[PositionRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload: Dict[str, str] = json.loads(line)
                fen = payload.get("fen")
                move = payload.get("best_move")
                if fen is None or move is None:
                    raise ValueError("Each dataset entry must contain 'fen' and 'best_move'")
                records.append(PositionRecord(fen=fen, best_move=move))
                if max_positions is not None and len(records) >= max_positions:
                    break
        if not records:
            raise ValueError(f"No records loaded from dataset at {path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        record = self.records[index]
        board = chess.Board(record.fen)
        features_tensor = feature_utils.board_to_tensor(board)
        target_index = move_to_index(record.best_move)
        sample: Dict[str, torch.Tensor | str] = {
            "features": features_tensor,
            "target": torch.tensor(target_index, dtype=torch.long),
            "fen": record.fen,
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


def create_dataloaders(
    config: ExperimentConfig,
    include_legal_mask: bool = False,
) -> tuple[DataLoader, Optional[DataLoader]]:
    dataset = PositionDataset(
        data_path=config.paths.dataset,
        max_positions=config.training.max_positions,
        include_legal_mask=include_legal_mask,
    )

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
            raise ValueError(
                "Validation split too large for dataset. Reduce training.val_split or provide more data."
            )
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
        collate_fn=default_collate,
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
            collate_fn=default_collate,
            pin_memory=True,
        )

    return train_loader, val_loader
