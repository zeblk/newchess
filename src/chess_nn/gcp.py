from __future__ import annotations

import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingLaunchPlan:
    project: str
    zone: str
    machine_type: str
    gpu_type: str
    config_path: str


class GCPTrainingClient:
    """Placeholder client for launching training jobs on Google Cloud Compute."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        if not api_key:
            LOGGER.warning(
                "Initialized GCPTrainingClient with an empty API key. All operations will be dry runs."
            )

    def launch_training_instance(self, plan: TrainingLaunchPlan) -> None:
        LOGGER.info(
            "[Dry Run] Launching training instance in project=%s zone=%s machine_type=%s gpu=%s with config %s",
            plan.project,
            plan.zone,
            plan.machine_type,
            plan.gpu_type,
            plan.config_path,
        )
        LOGGER.info(
            "Populate GCPTrainingClient.launch_training_instance with real API calls once credentials are configured."
        )
