#!/usr/bin/env python3
"""
CLI wrapper to run the Stockfish self-play data generator.
Usage example: 
    python3 scripts/generate_data.py --engine /home/zeblk/newchess/stockfish-ubuntu-x86-64-avx2 --threads 8

"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

import argparse
import logging

from chess_nn.stockfish_data import SelfPlayConfig, generate_selfplay_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stockfish self-play dataset")
    parser.add_argument("--engine", type=Path, default=Path("./stockfish-ubuntu-x86-64-avx2").resolve())
    parser.add_argument("--output", type=Path, default=Path("data/selfplay/"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--max-moves", type=int, default=512)
    parser.add_argument("--time", type=float, default=0.1)
    parser.add_argument("--random-prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--append", action="store_true", default=True, help="Append to output file when set (default: True)")
    parser.add_argument("--threads", type=int, default=1, help="Number of CPU threads to request from Stockfish")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker processes")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = SelfPlayConfig(
        engine_path=args.engine,
        output_path=args.output,
        games=args.games,
        max_moves_per_game=args.max_moves,
        time_limit=args.time,
        random_move_prob=args.random_prob,
        threads=args.threads,
        seed=args.seed,
        append=args.append,
        workers=args.workers,
    )
    generate_selfplay_data(cfg)


if __name__ == "__main__":
    main()
