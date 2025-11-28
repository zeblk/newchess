from __future__ import annotations

from typing import Iterable, List

import chess
import torch

FILES = "abcdefgh"
RANKS = "12345678"
PROMOTION_PIECES = "qrbn"


def _generate_all_uci_moves() -> List[str]:
    moves: List[str] = []
    for from_file in FILES:
        for from_rank in RANKS:
            from_sq = f"{from_file}{from_rank}"
            for to_file in FILES:
                for to_rank in RANKS:
                    to_sq = f"{to_file}{to_rank}"
                    if from_sq == to_sq:
                        continue
                    base = f"{from_sq}{to_sq}"
                    moves.append(base)
                    if from_rank in ("7", "2") and to_rank in ("8", "1"):
                        for promo in PROMOTION_PIECES:
                            moves.append(f"{base}{promo}")
    return moves


ALL_UCI_MOVES: List[str] = _generate_all_uci_moves()
MOVE_TO_INDEX = {uci: idx for idx, uci in enumerate(ALL_UCI_MOVES)}
NUM_MOVES = len(ALL_UCI_MOVES)


def move_to_index(move: chess.Move | str) -> int:
    uci = move if isinstance(move, str) else move.uci()
    try:
        return MOVE_TO_INDEX[uci]
    except KeyError as exc:  # pragma: no cover - guard against malformed data
        raise KeyError(f"Move '{uci}' not found in encoding table") from exc


def index_to_move(index: int) -> chess.Move:
    if index < 0 or index >= NUM_MOVES:
        raise IndexError(f"Move index {index} outside encoding range [0, {NUM_MOVES})")
    return chess.Move.from_uci(ALL_UCI_MOVES[index])


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(NUM_MOVES, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move_to_index(move)] = True
    return mask


def filter_valid_indices(indices: Iterable[int]) -> List[int]:
    return [idx for idx in indices if 0 <= idx < NUM_MOVES]
