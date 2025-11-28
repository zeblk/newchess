from __future__ import annotations

import argparse
from pathlib import Path

from chess_nn.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chess policy network trainer")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(config_path=args.config)


if __name__ == "__main__":
    main()
