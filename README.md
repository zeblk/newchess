# Chess Policy Network Trainer

This repository contains a PyTorch-based policy network that learns to play chess via supervised learning on a corpus of chess positions annotated with Stockfish's preferred moves. The code leverages the [`python-chess`](https://python-chess.readthedocs.io/) library for board representation and move handling.

## Key Features

- **Policy Network Architecture**: Residual convolutional neural network that outputs a probability distribution over a predefined set of moves.
- **Feature Extraction**: Converts board states (including piece placement, castling rights, en passant squares, and side-to-move) into tensor representations suitable for neural network consumption.
- **Move Encoding**: Deterministic mapping between legal move encodings and categorical indices for supervised learning targets.
- **Supervised Training Loop**: Mini-batch gradient descent with configurable optimizer, learning rate scheduler, and mixed-precision support.
- **Google Cloud Friendly**: Training pipeline detects CUDA devices automatically, making it suitable for GPU-enabled Google Compute Engine instances. Placeholder support for future integration with Google Cloud APIs is provided (no API key required at this stage).

## Repository Structure

```
.
├── README.md
├── pyproject.toml
├── configs/
│   └── default.yaml
├── data/
│   └── README.md
├── scripts/
│   └── launch_gcp_training.py
└── src/
    ├── main.py
    └── chess_nn/
        ├── __init__.py
        ├── config.py
        ├── data.py
        ├── features.py
        ├── gcp.py
        ├── model.py
        ├── move_encoding.py
        ├── train.py
        └── utils.py
```

## Dataset Format

Training data should be provided as a newline-delimited JSON (`.jsonl`) file where each line contains a dictionary with a FEN-encoded board position and Stockfish's preferred move for that position encoded in UCI notation:

```jsonl
{"fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "best_move": "d2d4"}
{"fen": "r1bqk2r/ppppbppp/2n2n2/2B1p3/4P3/2NP1N2/PPP2PPP/R1BQ1RK1 w kq - 2 7", "best_move": "c5e7"}
```

- **`fen`**: Full FEN string describing the board position, side to move, castling availability, en passant target, half-move clock, and full-move number.
- **`best_move`**: Stockfish's preferred move in [UCI format](https://en.wikipedia.org/wiki/Universal_Chess_Interface), including promotion suffixes (e.g., `e7e8q`).

Place datasets under the `data/` directory (ignored by version control) and update the configuration file accordingly.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Configuration

Configuration is handled through YAML files. The default configuration lives at `configs/default.yaml`. Key parameters include:

- **Data settings**: dataset path, validation split ratio, maximum number of positions to load.
- **Model settings**: convolutional channel width, number of residual blocks, dropout rate.
- **Training settings**: batch size, optimizer hyperparameters, gradient clipping, learning rate scheduler, output directory, mixed precision toggle.

You can create additional configuration files for different experiments. Pass the desired config path to the CLI when launching training.

## Training

```bash
python -m chess_nn.train --config configs/default.yaml
```

The training script will:

1. Load the dataset and split it into training and validation subsets.
2. Construct mini-batches with board tensors and target move indices.
3. Train the policy network using categorical cross-entropy loss.
4. Periodically evaluate on the validation set and save the best-performing model checkpoint.

### Mixed Precision Training

If CUDA is available and mixed precision is enabled in the configuration, the training loop will utilize `torch.cuda.amp` for improved performance on GPUs.

### Checkpoints & Logs

Model checkpoints, optimizer state, and configuration snapshots are stored under the `output_dir` specified in the config (defaults to `artifacts/default`).

## Google Cloud Compute Integration

The repository includes a placeholder script (`scripts/launch_gcp_training.py`) and module (`src/chess_nn/gcp.py`) to facilitate future integration with Google Cloud Compute Engine. These components are structured to authenticate using an API key and launch GPU-backed instances for training. The API key is intentionally left blank; populate it via environment variables or secret management tooling when you're ready to enable cloud integration.

## Development

- Format with `black` (optional dependency declared in `pyproject.toml`).
- Run tests (if/when added) via `pytest`.
- Type-check with `mypy` (optional).

## License

This project is released under the MIT License. See `LICENSE` for details.
