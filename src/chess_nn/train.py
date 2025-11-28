from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, _LRScheduler

from .config import ExperimentConfig, load_config
from .data import create_dataloaders
from .model import PolicyNetwork
from .utils import (
    count_parameters,
    ensure_dir,
    resolve_device,
    save_checkpoint,
    set_seed,
    summarize_config,
)

LOGGER = logging.getLogger(__name__)


def build_model(config: ExperimentConfig, device: torch.device) -> PolicyNetwork:
    model = PolicyNetwork(
        input_planes=config.model.input_planes,
        channels=config.model.channels,
        residual_blocks=config.model.residual_blocks,
        dropout=config.model.dropout,
    )
    model.to(device)
    LOGGER.info("Model parameters: %s", count_parameters(model))
    return model


def build_optimizer(model: nn.Module, config: ExperimentConfig) -> Optimizer:
    name = config.optimizer.name.lower()
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
            betas=config.optimizer.betas,
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.optimizer.lr,
            momentum=0.9,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer '{config.optimizer.name}'")


def build_scheduler(
    optimizer: Optimizer,
    config: ExperimentConfig,
) -> Tuple[Optional[_LRScheduler], str]:
    name = (config.scheduler.name or "none").lower()
    if name in {"none", "null", ""}:
        return None, "epoch"
    if name == "cosine":
        t_max = config.scheduler.t_max or config.training.epochs
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
            eta_min=config.scheduler.min_lr,
        )
        return scheduler, "epoch"
    if name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=config.scheduler.step_size,
            gamma=config.scheduler.gamma,
        )
        return scheduler, "epoch"
    raise ValueError(f"Unsupported scheduler '{config.scheduler.name}'")


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    scaler: GradScaler,
    grad_clip: Optional[float],
    use_amp: bool,
    use_tqdm: bool,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    iterable = dataloader
    if use_tqdm:
        try:
            from tqdm.auto import tqdm

            iterable = tqdm(dataloader, desc=f"Epoch {epoch} [train]", leave=False)
        except ImportError:  # pragma: no cover - tqdm optional
            pass

    for batch in iterable:
        inputs = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(inputs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        predictions = torch.argmax(logits.detach(), dim=1)
        total_correct += (predictions == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 20.0 else float("inf")

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "perplexity": perplexity,
    }
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    use_tqdm: bool,
    epoch: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_legal_correct = 0
    total_samples = 0

    iterable = dataloader
    if use_tqdm:
        try:
            from tqdm.auto import tqdm

            iterable = tqdm(dataloader, desc=f"Epoch {epoch} [val]", leave=False)
        except ImportError:  # pragma: no cover
            pass

    for batch in iterable:
        inputs = batch["features"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        legal_mask = batch.get("legal_mask")
        if legal_mask is not None:
            legal_mask = legal_mask.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(inputs)
            loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        predictions = torch.argmax(logits, dim=1)
        total_correct += (predictions == targets).sum().item()

        if legal_mask is not None:
            masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
            safe_mask = torch.isfinite(masked_logits).any(dim=1)
            fallback_logits = logits[~safe_mask]
            masked_logits = masked_logits.clone()
            if fallback_logits.numel() > 0:
                masked_logits[~safe_mask] = fallback_logits
            legal_predictions = torch.argmax(masked_logits, dim=1)
            total_legal_correct += (legal_predictions == targets).sum().item()

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    legal_accuracy = (
        total_legal_correct / max(total_samples, 1) if total_legal_correct > 0 else accuracy
    )
    perplexity = math.exp(avg_loss) if avg_loss < 20.0 else float("inf")

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "legal_accuracy": legal_accuracy,
        "perplexity": perplexity,
    }
    return metrics


def run_training(config_path: Path | str) -> Dict[str, Dict[str, float]]:
    config = load_config(config_path)
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Loaded configuration:\n%s", summarize_config(config.as_dict()))

    set_seed(config.seed)
    device = resolve_device(config.device)
    LOGGER.info("Using device: %s", device)

    train_loader, val_loader = create_dataloaders(
        config, include_legal_mask=True
    )

    model = build_model(config, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler, scheduler_frequency = build_scheduler(optimizer, config)
    use_amp = config.training.amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    history: Dict[str, Dict[str, float]] = {}
    best_val_loss = float("inf")
    output_dir = config.training.output_dir
    ensure_dir(output_dir)

    for epoch in range(1, config.training.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            grad_clip=config.training.grad_clip_norm,
            use_amp=use_amp,
            use_tqdm=config.logging.use_tqdm,
            epoch=epoch,
        )

        LOGGER.info(
            "Epoch %d | train loss %.4f | acc %.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["accuracy"],
        )

        val_metrics: Optional[Dict[str, float]] = None
        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
                use_tqdm=config.logging.use_tqdm,
                epoch=epoch,
            )
            LOGGER.info(
                "Epoch %d | val loss %.4f | acc %.4f | legal acc %.4f",
                epoch,
                val_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["legal_accuracy"],
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics={"train": train_metrics, "val": val_metrics, "epoch": epoch},
                    config=config.as_dict(),
                    filename="best.pt",
                )

        if scheduler is not None and scheduler_frequency == "epoch":
            scheduler.step()

        if epoch % config.training.checkpoint_every == 0:
            save_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics={"train": train_metrics, "val": val_metrics, "epoch": epoch},
                config=config.as_dict(),
                filename=f"checkpoint_epoch_{epoch:03d}.pt",
            )

        history[f"epoch_{epoch:03d}"] = {
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
        }
        if val_loader is not None and val_metrics is not None:
            history[f"epoch_{epoch:03d}"].update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_legal_accuracy": val_metrics["legal_accuracy"],
                }
            )

    save_checkpoint(
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=config.training.epochs,
        metrics={"history": history, "epoch": config.training.epochs},
        config=config.as_dict(),
        filename="last.pt",
    )

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the chess policy network")
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
