# Data Directory

Place your training data in this directory. Expected format is newline-delimited JSON (`.jsonl`) where each line contains:

- `fen`: A FEN string describing the board state.
- `best_move`: Stockfish's preferred move in UCI notation (e.g., `e2e4`, `e7e8q`).

Example entry:

```
{"fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "best_move": "d2d4"}
```

Update `configs/default.yaml` (or your custom config) with the path to the dataset file.
