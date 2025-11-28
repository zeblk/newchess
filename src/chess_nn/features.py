from __future__ import annotations

from typing import Iterable, List

import chess
import numpy as np
import torch

WHITE = chess.WHITE
BLACK = chess.BLACK

PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]

PIECE_PLANES = {
    (WHITE, chess.PAWN): 0,
    (WHITE, chess.KNIGHT): 1,
    (WHITE, chess.BISHOP): 2,
    (WHITE, chess.ROOK): 3,
    (WHITE, chess.QUEEN): 4,
    (WHITE, chess.KING): 5,
    (BLACK, chess.PAWN): 6,
    (BLACK, chess.KNIGHT): 7,
    (BLACK, chess.BISHOP): 8,
    (BLACK, chess.ROOK): 9,
    (BLACK, chess.QUEEN): 10,
    (BLACK, chess.KING): 11,
}

SIDE_TO_MOVE_PLANE = 12
CASTLING_PLANES = {
    "white_k": 13,
    "white_q": 14,
    "black_k": 15,
    "black_q": 16,
}
EN_PASSANT_PLANE = 17

NUM_FEATURE_PLANES = EN_PASSANT_PLANE + 1


def square_to_coords(square: int) -> tuple[int, int]:
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    return 7 - rank, file  # Row major: rank 8 at index 0


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    array = np.zeros((NUM_FEATURE_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        plane_idx = PIECE_PLANES[(piece.color, piece.piece_type)]
        row, col = square_to_coords(square)
        array[plane_idx, row, col] = 1.0

    array[SIDE_TO_MOVE_PLANE, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    array[CASTLING_PLANES["white_k"], :, :] = (
        1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    )
    array[CASTLING_PLANES["white_q"], :, :] = (
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    )
    array[CASTLING_PLANES["black_k"], :, :] = (
        1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    )
    array[CASTLING_PLANES["black_q"], :, :] = (
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    )

    if board.ep_square is not None:
        row, col = square_to_coords(board.ep_square)
        array[EN_PASSANT_PLANE, row, col] = 1.0

    return torch.from_numpy(array)


def batch_board_to_tensor(boards: Iterable[chess.Board]) -> torch.Tensor:
    tensors: List[torch.Tensor] = [board_to_tensor(board) for board in boards]
    return torch.stack(tensors, dim=0)
