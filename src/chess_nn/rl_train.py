from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.optim import Optimizer

from .config import ExperimentConfig
from .data import create_dataloaders
from .model import PolicyNetwork
from .selfplay import SelfPlayConfig, generate_selfplay_data
from .train import build_model, build_optimizer, build_scheduler
from .utils import ensure_dir, resolve_device, save_checkpoint, set_seed

LOGGER = logging.getLogger(__name__)


def train_rl_epoch(
    model: nn.Module,
    dataloader,
    optimizer: Optimizer,
    device: torch.device,
    scaler: GradScaler,
    grad_clip: Optional[float],
    use_amp: bool,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        inputs = batch["features"].to(device, non_blocking=True)
        action_targets = batch["action_targets"].to(device, non_blocking=True)
        value_targets = batch["value_targets"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits, values = model(inputs)
            values = values.squeeze(-1)  # (B, 1) -> (B,)

            # Value Loss: MSE between predicted value and actual game outcome
            value_loss = F.mse_loss(values, value_targets)

            # Policy Loss: CrossEntropy * Advantage
            # Advantage = (Return - Value)
            # Here we use a simple version where we weight the policy gradient by the outcome.
            # Ideally we want to encourage moves that lead to winning.
            # If we use the outcome as the advantage:
            #   If outcome is 1 (Win), we want to increase prob of action.
            #   If outcome is -1 (Loss), we want to decrease prob of action.
            # Using CrossEntropyLoss gives us -log(prob(action)).
            # So we want to minimize: -log(prob(action)) * outcome
            # However, standard REINFORCE uses return.
            # A2C uses (Return - Value_detached).
            
            # Let's use A2C style: Advantage = Target - Value.detach()
            # But here Target is just the game outcome.
            advantage = value_targets - values.detach()
            
            # CrossEntropyLoss computes -log(p_target)
            # We want to minimize -log(p_target) * advantage
            # But CrossEntropyLoss doesn't take sample weights directly in this way easily without reduction='none'
            ce_loss = F.cross_entropy(logits, action_targets, reduction='none')
            policy_loss = (ce_loss * advantage).mean()
            
            # Combine losses
            # We might want to weight them. AlphaZero uses loss = (z - v)^2 + pi^T * log(p) + c * ||theta||^2
            # Here: loss = value_loss + policy_loss
            loss = value_loss + policy_loss

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        batch_size = action_targets.size(0)
        total_loss += loss.item() * batch_size
        total_policy_loss += policy_loss.item() * batch_size
        total_value_loss += value_loss.item() * batch_size
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    avg_policy_loss = total_policy_loss / max(total_samples, 1)
    avg_value_loss = total_value_loss / max(total_samples, 1)

    return {
        "loss": avg_loss,
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss,
    }


def run_rl_training(config: ExperimentConfig, rl_config: SelfPlayConfig, iterations: int) -> None:
    logging.basicConfig(level=logging.INFO)
    set_seed(config.seed)
    device = resolve_device(config.device)
    LOGGER.info("Using device: %s", device)

    # Initialize model
    model = build_model(config, device)
    optimizer = build_optimizer(model, config)
    scheduler, _ = build_scheduler(optimizer, config)
    use_amp = config.training.amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    
    output_dir = config.training.output_dir
    ensure_dir(output_dir)
    
    # RL Loop
    for iteration in range(1, iterations + 1):
        LOGGER.info(f"--- Starting RL Iteration {iteration}/{iterations} ---")
        
        # 1. Generate Self-Play Data
        LOGGER.info("Generating self-play data...")
        # Clean up old data to keep it fresh? Or append?
        # For this simple implementation, let's clear the data directory for each iteration
        # to implement an iterative improvement loop (on-policy-ish).
        data_path = Path(rl_config.output_path).parent
        if data_path.exists():
            shutil.rmtree(data_path)
        data_path.mkdir(parents=True, exist_ok=True)
        
        # Update config to point to the directory
        rl_config.output_path = data_path
        
        # Generate
        generate_selfplay_data(rl_config, model)
        
        # 2. Train on generated data
        LOGGER.info("Training on generated data...")
        # Update dataset path in config to point to generated data
        config.paths.dataset = data_path
        
        train_loader, _ = create_dataloaders(config, include_legal_mask=True, is_rl=True)
        if train_loader is None:
            LOGGER.warning("No data generated, skipping training step.")
            continue
            
        # Train for one or more epochs on this data
        # Usually in AlphaZero we train for a bit. Let's do 1 epoch per iteration for now,
        # or use config.training.epochs.
        for epoch in range(1, config.training.epochs + 1):
            metrics = train_rl_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                grad_clip=config.training.grad_clip_norm,
                use_amp=use_amp,
                epoch=epoch,
            )
            
            LOGGER.info(
                "Iter %d | Epoch %d | loss %.4f | policy_loss %.4f | value_loss %.4f",
                iteration,
                epoch,
                metrics["loss"],
                metrics["policy_loss"],
                metrics["value_loss"],
            )
            
        if scheduler is not None:
            scheduler.step()
            
        # Save checkpoint
        save_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=iteration,
            metrics=metrics,
            config=config.as_dict(),
            filename=f"rl_iter_{iteration:03d}.pt",
        )
        # Also save as latest
        save_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=iteration,
            metrics=metrics,
            config=config.as_dict(),
            filename="latest.pt",
        )
