# Data Directory

Place your training data in this directory. Expected format is newline-delimited JSON (`.jsonl`) where each line contains:

- `fen`: A FEN string describing the board state.
- `best_move`: Stockfish's preferred move in UCI notation (e.g., `e2e4`, `e7e8q`).

Example entry:

```
{"fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "best_move": "d2d4"}
```

This repository includes a small self-play data generator that runs a Stockfish binary to produce labeled positions. The generator will play games starting from the standard initial position, with Stockfish playing both sides. Stockfish is given a fixed time budget per move (default 0.25s). In order to increase position diversity, the generator will substitute a random legal move with configurable probability (default 0.1) while still recording Stockfish's preferred move as the training label.

To run the generator, place a Stockfish UCI binary (for example `stockfish-ubuntu-x86-64-avx2`) at the project root and run:

```bash
python scripts/generate_data.py --engine ./stockfish-ubuntu-x86-64-avx2 --output data/stockfish_positions.jsonl --games 100 --time 0.25 --random-prob 0.1
```

The generator appends newline-delimited JSON records to the specified output file. Each run will continue appending to the same file unless you pass `--append` set to false (the CLI defaults to append mode).

Update `configs/default.yaml` (or your custom config) with the path to the dataset file if necessary.
