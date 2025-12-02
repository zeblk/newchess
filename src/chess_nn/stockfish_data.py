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

import multiprocessing
import os
import shutil
import tempfile

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
    workers: int = 1  # number of parallel worker processes


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


def _worker_generate(config: SelfPlayConfig, worker_id: int, num_games: int) -> int:
    """Worker function to generate a subset of games."""
    # Create a unique shard file for this worker
    # Format: part_{worker_id}_{random_hex}.jsonl
    import uuid
    shard_name = f"part_{worker_id}_{uuid.uuid4().hex[:8]}.jsonl"
    shard_path = Path(config.output_path) / shard_name
    
    # Adjust config for this worker
    worker_config = SelfPlayConfig(
        engine_path=config.engine_path,
        output_path=shard_path,
        games=num_games,
        max_moves_per_game=config.max_moves_per_game,
        time_limit=config.time_limit,
        random_move_prob=config.random_move_prob,
        threads=1,  # Force 1 thread per worker
        seed=config.seed + worker_id if config.seed is not None else None,
        append=False, # New file for this shard
        workers=1,
    )

    try:
        written = generate_selfplay_data(worker_config)
        return written
    except Exception as e:
        LOGGER.exception(f"Worker {worker_id} failed: {e}")
        return 0


def generate_selfplay_data(config: SelfPlayConfig) -> int:
    """Run Stockfish self-play and write positions to output_path.
    
    If workers > 1, output_path must be a directory.
    If workers = 1, output_path can be a file or directory (if dir, a shard is created).
    """
    out_path = Path(config.output_path)
    
    # If workers > 1, dispatch to workers
    if config.workers > 1 and config.games > 0:
        if out_path.exists() and not out_path.is_dir():
             raise NotADirectoryError(f"Output path {out_path} must be a directory for parallel generation.")
        out_path.mkdir(parents=True, exist_ok=True)

        LOGGER.info(f"Starting {config.workers} workers for {config.games} games...")
        games_per_worker = config.games // config.workers
        remainder = config.games % config.workers
        
        tasks = []
        for i in range(config.workers):
            n_games = games_per_worker + (1 if i < remainder else 0)
            if n_games > 0:
                tasks.append((config, i, n_games))

        total_written = 0
        with multiprocessing.Pool(processes=config.workers) as pool:
            results = pool.starmap(_worker_generate, tasks)
            total_written = sum(results)

        LOGGER.info("Parallel generation finished. Wrote %d positions to %s", total_written, out_path)
        return total_written

    # Single process execution
    # If output_path is a directory, create a default shard file
    if out_path.is_dir() or (not out_path.suffix and not out_path.exists()):
        out_path.mkdir(parents=True, exist_ok=True)
        import uuid
        shard_name = f"part_0_{uuid.uuid4().hex[:8]}.jsonl"
        target_file = out_path / shard_name
    else:
        target_file = out_path
        target_file.parent.mkdir(parents=True, exist_ok=True)

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

        with target_file.open(mode, encoding="utf-8") as handle:
            for fen, best_uci in _iter_selfplay_positions(engine, config):
                payload = {"fen": fen, "best_move": best_uci}
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
                handle.flush()
                written += 1
    LOGGER.info("Wrote %d positions to %s", written, target_file)
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
