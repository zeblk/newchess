from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import chess
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn

from .features import board_to_tensor
from .move_encoding import move_to_index, index_to_move, legal_move_mask

LOGGER = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    output_path: Path | str = "data/selfplay.jsonl"
    games: int = 100
    max_moves_per_game: int = 512
    temperature: float = 1.0
    device: str = "cpu"
    append: bool = True


class SelfPlayEngine:
    def __init__(self, model: nn.Module, device: torch.device, temperature: float = 1.0):
        self.model = model
        self.device = device
        self.temperature = temperature
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def select_move(self, board: chess.Board) -> chess.Move:
        if board.is_game_over():
            raise ValueError("Game is over")

        # Prepare input
        x = board_to_tensor(board).unsqueeze(0).to(self.device)
        
        # Get policy logits
        logits, _ = self.model(x)  # Ignore value for move selection
        logits = logits.squeeze(0)

        # Mask illegal moves
        legal_mask = legal_move_mask(board).to(self.device)
        logits = logits.masked_fill(~legal_mask, float("-inf"))

        # Apply temperature
        if self.temperature == 0:
            move_idx = torch.argmax(logits).item()
        else:
            probs = F.softmax(logits / self.temperature, dim=0)
            move_idx = torch.multinomial(probs, 1).item()

        # Decode move
        # Decode move
        move = index_to_move(move_idx)
        
        # Fallback if something went wrong (shouldn't happen with correct masking)
        if move not in board.legal_moves:
             # Just pick a random legal move
             return random.choice(list(board.legal_moves))
             
        return move


def generate_game(engine: SelfPlayEngine, max_moves: int = 512) -> list[dict]:
    board = chess.Board()
    game_history = []
    
    for _ in range(max_moves):
        if board.is_game_over(claim_draw=True):
            break
            
        fen = board.fen()
        move = engine.select_move(board)
        
        game_history.append({
            "fen": fen,
            "move": move.uci(),
            "turn": board.turn, # True for White, False for Black
        })
        
        board.push(move)
        
    # Determine result
    # 1 for White win, -1 for Black win, 0 for Draw
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        # Game reached max moves without result, treat as draw? Or just discard?
        # Let's treat as draw for now.
        result_val = 0.0
    else:
        if outcome.winner == chess.WHITE:
            result_val = 1.0
        elif outcome.winner == chess.BLACK:
            result_val = -1.0
        else:
            result_val = 0.0
            
    # Assign rewards to history
    # Value target is always from the perspective of the player to move in that position
    for step in game_history:
        # If it was White's turn, and White won (1.0), reward is 1.0
        # If it was Black's turn, and White won (1.0), reward is -1.0
        step_turn = step["turn"]
        if step_turn == chess.WHITE:
            step["value"] = result_val
        else:
            step["value"] = -result_val
            
    return game_history


def generate_selfplay_data(config: SelfPlayConfig, model: nn.Module) -> int:
    accelerator = Accelerator() # Automatically detects best device
    device = accelerator.device
    engine = SelfPlayEngine(model, device, temperature=config.temperature)
    
    out_path = Path(config.output_path)
    if out_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        shard_name = f"selfplay_{uuid.uuid4().hex[:8]}.jsonl"
        target_file = out_path / shard_name
    else:
        target_file = out_path
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
    mode = "a" if config.append else "w"
    
    total_positions = 0
    with target_file.open(mode, encoding="utf-8") as f:
        for i in range(config.games):
            game_data = generate_game(engine, config.max_moves_per_game)
            for step in game_data:
                payload = {
                    "fen": step["fen"],
                    "action": step["move"],
                    "value": step["value"]
                }
                f.write(json.dumps(payload) + "\n")
                total_positions += 1
            
            if (i + 1) % 10 == 0:
                LOGGER.info(f"Generated {i + 1}/{config.games} games")
                
    LOGGER.info(f"Finished generating {config.games} games with {total_positions} positions.")
    return total_positions
