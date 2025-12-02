#!/usr/bin/env python3
"""
CLI entry point for RL training.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

import argparse
import logging

from chess_nn.config import load_config
from chess_nn.rl_train import run_rl_training
from chess_nn.selfplay import SelfPlayConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train chess agent via RL (Self-Play)")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of RL iterations (generate -> train loop)",
    )
    parser.add_argument(
        "--games-per-iter",
        type=int,
        default=100,
        help="Number of self-play games to generate per iteration",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=512,
        help="Max moves per game",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for self-play",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load base config
    config = load_config(args.config)
    
    # Setup RL config
    rl_config = SelfPlayConfig(
        output_path="data/rl_selfplay/", # Directory for shards
        games=args.games_per_iter,
        max_moves_per_game=args.max_moves,
        temperature=args.temperature,
        device=config.device,
        append=False, # We clear directory each iter in rl_train.py
    )
    
    run_rl_training(config, rl_config, iterations=args.iterations)


if __name__ == "__main__":
    main()
