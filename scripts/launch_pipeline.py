#!/usr/bin/env python3
"""Launch data generator (asynchronously) and PyTorch training together.

This script ensures the project source (src/) is on PYTHONPATH for the child
processes, optionally checks for key dependencies, can auto-install the
project in editable mode, and runs the generator and trainer concurrently.

Usage examples:
  # Simple run (from repository root):
  python3 scripts/launch_pipeline.py --engine ./stockfish-ubuntu-x86-64-avx2 --output data/stockfish_positions.jsonl --games 0

  # Auto-install project deps first (editable install)
  python3 scripts/launch_pipeline.py --auto-install

The script terminates the generator when training exits.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Optional, Tuple

LOGGER = logging.getLogger("launch_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch generator + training together")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the YAML configuration file for training",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=Path("./stockfish-ubuntu-x86-64-avx2"),
        help="Path to the Stockfish binary",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/stockfish_positions.jsonl"),
        help="Path to the dataset file to append to",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=0,
        help="Number of self-play games to generate (<=0 keeps running until training completes)",
    )
    parser.add_argument(
        "--max-moves",
        type=int,
        default=512,
        help="Maximum number of plies to play per game",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=0.25,
        help="Time budget in seconds for Stockfish per move",
    )
    parser.add_argument(
        "--random-prob",
        type=float,
        default=0.1,
        help="Probability of substituting a random legal move during self-play",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for move substitution (optional)",
    )
    parser.add_argument(
        "--no-append",
        dest="append",
        action="store_false",
        help="Overwrite the dataset file before generation instead of appending",
    )
    parser.set_defaults(append=True)
    parser.add_argument(
        "--generator-log",
        type=Path,
        default=None,
        help="Optional path to log generator stdout/stderr",
    )
    parser.add_argument(
        "--auto-install",
        action="store_true",
        help="Run `pip install -e .` with the current Python interpreter before launching",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Only check for key dependencies (python-chess, torch) and exit if missing",
    )
    return parser.parse_args()

def build_generator_command(args: argparse.Namespace) -> list[str]:
    engine_path = str(Path(args.engine).resolve())     # <-- resolve to absolute path
    cmd = [
        sys.executable,
        "scripts/generate_data.py",
        "--engine",
        engine_path,
        "--output",
        str(args.output),
        "--games",
        str(args.games),
        "--max-moves",
        str(args.max_moves),
        "--time",
        str(args.time),
        "--random-prob",
        str(args.random_prob),
    ]
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    if not args.append:
        cmd.append("--overwrite")
    return cmd

def launch_generator(cmd: list[str], log_path: Optional[Path]) -> Tuple[subprocess.Popen, Optional[IO[str]]]:
    LOGGER.info("Starting data generator: %s", " ".join(cmd))
    stdout_sink: Optional[IO[str]] = None
    popen_kwargs = {}
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_sink = log_path.open("a", encoding="utf-8")
        popen_kwargs["stdout"] = stdout_sink
        popen_kwargs["stderr"] = subprocess.STDOUT

    # Ensure src/ is on PYTHONPATH for the generator subprocess as well
    env = os.environ.copy()
    src_path = str(Path("src").resolve())
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path + (os.pathsep + prev if prev else "")
    popen_kwargs["env"] = env

    try:
        process = subprocess.Popen(cmd, **popen_kwargs)
    except Exception:
        if stdout_sink is not None:
            stdout_sink.close()
        raise
    return process, stdout_sink


def run_training(train_config: Path) -> int:
    cmd = [sys.executable, "-m", "chess_nn.train", "--config", str(train_config)]
    LOGGER.info("Starting training: %s", " ".join(cmd))
    env = os.environ.copy()
    # Ensure src/ is on PYTHONPATH so "-m chess_nn.train" can import the package
    src_path = str(Path("src").resolve())
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path + (os.pathsep + prev if prev else "")
    result = subprocess.run(cmd, check=False, env=env)
    return result.returncode


def wait_for_dataset(path: Path, timeout: int = 60) -> None:
    """Wait until the dataset file exists and contains at least one non-empty line.

    Raises TimeoutError if the file is not ready within `timeout` seconds.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if path.exists() and path.stat().st_size > 0:
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            LOGGER.info("Dataset available at %s", path)
                            return
        except Exception:
            # ignore transient IO errors and retry
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for dataset at {path}")


def check_dependencies() -> bool:
    """Check for 'chess' and 'torch' imports. Returns True if both available."""
    missing = []
    try:
        import chess  # type: ignore
    except Exception:
        missing.append("python-chess (import name: chess)")
    try:
        import torch  # type: ignore
    except Exception:
        missing.append("torch")
    if missing:
        LOGGER.warning("Missing dependencies: %s", ", ".join(missing))
        LOGGER.info("You can install the project (and deps) with: %s -m pip install -e .", sys.executable)
        return False
    LOGGER.info("Required dependencies present: python-chess, torch")
    return True


def auto_install_editable() -> int:
    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    LOGGER.info("Running auto-install: %s", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.auto_install:
        rc = auto_install_editable()
        if rc != 0:
            LOGGER.error("Auto-install failed with return code %d", rc)
            sys.exit(rc)

    deps_ok = check_dependencies()
    if args.check_deps:
        sys.exit(0 if deps_ok else 2)
    if not deps_ok:
        LOGGER.error("Missing required dependencies. Either install them or run with --auto-install.")
        sys.exit(1)

    generator_cmd = build_generator_command(args)
    generator_process: Optional[subprocess.Popen] = None
    log_handle: Optional[IO[str]] = None

    try:
        generator_process, log_handle = launch_generator(generator_cmd, args.generator_log)
    except FileNotFoundError as exc:
        LOGGER.error("Failed to start data generator: %s", exc)
        sys.exit(1)

    # Wait for the generator to produce at least one dataset record before starting training
    try:
        wait_for_dataset(args.output, timeout=60)
    except TimeoutError as exc:
        LOGGER.error("%s", exc)
        # Stop generator if it was started
        if generator_process is not None and generator_process.poll() is None:
            LOGGER.info("Stopping data generator due to missing dataset ...")
            generator_process.terminate()
            try:
                generator_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                generator_process.kill()
        sys.exit(1)

    return_code = 1
    try:
        return_code = run_training(args.train_config)
    finally:
        if generator_process is not None:
            if generator_process.poll() is None:
                LOGGER.info("Stopping data generator ...")
                generator_process.terminate()
                try:
                    generator_process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    LOGGER.warning("Generator did not exit gracefully; killing it")
                    generator_process.kill()
            else:
                if generator_process.returncode not in (0, None):
                    LOGGER.warning(
                        "Data generator exited early with return code %s",
                        generator_process.returncode,
                    )
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass

    if return_code != 0:
        LOGGER.error("Training exited with return code %d", return_code)
    else:
        LOGGER.info("Training completed successfully")
    sys.exit(return_code)


if __name__ == "__main__":
    main()
