from __future__ import annotations

import json
import logging
import random
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
import chess.engine

LOGGER = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    engine_path: Path | str = "./stockfish-ubuntu-x86-64-avx2"
    output_path: Path | str = "data/stockfish_positions.jsonl"
    games: int = 100
    max_moves_per_game: int = 512
    time_limit: float = 0.25  # seconds per move for Stockfish
    random_move_prob: float = 0.1  # probability to play a random move instead of engine's best
    threads: int = 1  # number of CPU threads to request from the engine
    seed: int | None = None
    append: bool = True


def _iter_selfplay_positions(
    engine: chess.engine.SimpleEngine, config: SelfPlayConfig
) -> Iterator[tuple[str, str]]:
    rng = random.Random(config.seed)

    # Support config.games <= 0 meaning "run until externally stopped" (infinite self-play)
    if config.games <= 0:
        game_iter: Iterator[int] = itertools.count(0)
        total_games_display = "<infinite>"
    else:
        game_iter = range(config.games)
        total_games_display = str(config.games)

    for game_idx in game_iter:
        board = chess.Board()
        move_count = 0
        LOGGER.info("Starting self-play game %d/%s", game_idx + 1, total_games_display)
        while not board.is_game_over(claim_draw=True) and move_count < config.max_moves_per_game:
            # Ask Stockfish for its preferred move
            try:
                result = engine.play(board, chess.engine.Limit(time=config.time_limit))
                best_move = result.move
            except Exception as exc:  # pragma: no cover - guard if engine fails
                LOGGER.exception("Engine failed to produce a move: %s", exc)
                break

            # Record the position and the engine's preferred move (label)
            fen = board.fen()
            best_uci = best_move.uci()
            yield fen, best_uci

            # Decide what to play on the board: mostly engine's move, sometimes a random legal move
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            if rng.random() < config.random_move_prob:
                chosen_move = rng.choice(legal_moves)
            else:
                chosen_move = best_move

            board.push(chosen_move)
            move_count += 1

        LOGGER.info(
            "Finished game %d/%s after %d moves (result=%s)",
            game_idx + 1,
            total_games_display,
            move_count,
            board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*",
        )


def generate_selfplay_data(config: SelfPlayConfig) -> int:
    """Run Stockfish self-play and append (fen, best_move) entries to output_path.

    Returns the number of positions written.
    """
    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if config.append else "w"

    # Ensure engine binary exists
    engine_path = Path(config.engine_path)
    if not engine_path.exists():
        raise FileNotFoundError(f"Stockfish engine not found at: {engine_path}")

    written = 0
    # Use the UCI engine interface provided by python-chess
    with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
        # Optionally configure engine threads or other options here
        try:
            # Request the configured number of engine threads (some builds may ignore this)
            engine.configure({"Threads": int(config.threads)})
        except Exception:
            # Some Stockfish builds may not accept configuration; ignore failures
            pass

        with out_path.open(mode, encoding="utf-8") as handle:
            for fen, best_uci in _iter_selfplay_positions(engine, config):
                payload = {"fen": fen, "best_move": best_uci}
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
                handle.flush()
                written += 1
    LOGGER.info("Wrote %d positions to %s", written, out_path)
    return written


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate self-play positions with Stockfish")
    parser.add_argument("--engine", type=Path, default=Path("./stockfish-ubuntu-x86-64-avx2"))
    parser.add_argument("--output", type=Path, default=Path("data/stockfish_positions.jsonl"))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--max-moves", type=int, default=512)
    parser.add_argument("--time", type=float, default=0.25)
    parser.add_argument("--random-prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--append", action="store_true", default=True)
    parser.add_argument("--threads", type=int, default=1, help="Number of CPU threads to request from Stockfish")
    args = parser.parse_args()

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
    )
    generate_selfplay_data(cfg)
