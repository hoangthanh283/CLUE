"""Weights & Biases experiment tracking utilities."""

import logging
import os
from typing import Optional

import wandb

logger = logging.getLogger(__name__)


def init_wandb_run(config) -> Optional[object]:
    """Initialize wandb run for experiment tracking.

    Args:
        config: wandb config dict (the 'wandb' section), or the full config dict
                (legacy path — full config will have its 'wandb' key extracted).

    Returns:
        wandb.sdk.wandb_run.Run instance if configured, else None.
    """
    # Accept either the wandb sub-dict or the full config dict (legacy).
    if isinstance(config, dict) and "wandb" in config:
        wandb_config = config.get("wandb", {})
        experiment_name = config.get("experiment_name", "experiment")
    else:
        wandb_config = config if isinstance(config, dict) else {}
        experiment_name = "experiment"

    if not wandb_config.get("use_wandb", False):
        return None

    api_key = wandb_config.get("wandb_api_key") or os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)

    project = wandb_config.get("wandb_project") or os.getenv("WANDB_PROJECT", "cl4ie")
    entity = wandb_config.get("wandb_entity") or os.getenv("WANDB_ENTITY")

    run = wandb.init(
        project=project,
        entity=entity,
        name=experiment_name,
        tags=wandb_config.get("tags", []),
        config=wandb_config,
        reinit=True,
    )
    logger.info(f"wandb run initialized: {run.id}")
    return run
