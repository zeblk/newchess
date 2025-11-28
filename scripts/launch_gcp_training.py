#!/usr/bin/env python3
"""Placeholder script for launching training on Google Cloud Compute Engine.

This script is intentionally skeletal. Populate the `GCP_API_KEY` environment
variable and extend the helper functions in `chess_nn.gcp` to integrate with
Google Cloud when ready.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from chess_nn import gcp


LOGGER = logging.getLogger("gcp_launcher")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the chess policy network training job on GCP"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to the training configuration file",
    )
    parser.add_argument(
        "--project",
        type=str,
        required=False,
        help="Google Cloud project ID",
    )
    parser.add_argument(
        "--zone",
        type=str,
        default="us-central1-a",
        help="Compute Engine zone to launch the instance",
    )
    parser.add_argument(
        "--machine-type",
        type=str,
        default="n1-standard-8",
        help="Machine type for the training instance",
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default="nvidia-tesla-t4",
        help="GPU accelerator to attach to the instance",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch plan without executing any API calls",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    api_key = os.environ.get("GCP_API_KEY", "")
    if not api_key:
        LOGGER.warning(
            "GCP_API_KEY environment variable is empty. Set it before launching real jobs."
        )

    launch_plan = gcp.TrainingLaunchPlan(
        project=args.project or "",
        zone=args.zone,
        machine_type=args.machine_type,
        gpu_type=args.gpu_type,
        config_path=str(args.config),
    )

    if args.dry_run:
        LOGGER.info("Dry run enabled. Launch plan: %s", launch_plan)
        return

    client = gcp.GCPTrainingClient(api_key=api_key)
    client.launch_training_instance(plan=launch_plan)


if __name__ == "__main__":
    main()
