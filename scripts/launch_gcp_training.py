#!/usr/bin/env python3
"""Placeholder script for launching training on Google Cloud Compute Engine.

This script is intentionally skeletal. Populate the `GCP_API_KEY` environment
variable and extend the helper functions in `chess_nn.gcp` to integrate with
Google Cloud when ready.

Example usage:
    python3 scripts/launch_gcp_training.py --config configs/default.yaml --project my-gcp-project --zone us-central1-a --dry-run
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from chess_nn import gcp


import subprocess
import sys

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
