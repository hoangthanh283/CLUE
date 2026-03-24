"""Neptune experiment tracking utilities."""

import logging
import os
from typing import Optional

import neptune
from neptune.utils import stringify_unsupported

logger = logging.getLogger(__name__)


def init_neptune_run(config) -> Optional[neptune.Run]:
    """Initialize Neptune run for experiment tracking.

    Args:
        config: neptune config dict (the 'neptune' section), or the full config dict
                (legacy path — full config will have its 'neptune' key extracted).

    Returns:
        neptune.Run instance if configured and credentials provided, else None.
    """
    # Accept either the neptune sub-dict or the full config dict (legacy).
    if isinstance(config, dict) and "neptune" in config:
        neptune_config = config.get("neptune", {})
        experiment_name = config.get("experiment_name", "experiment")
    else:
        neptune_config = config if isinstance(config, dict) else {}
        experiment_name = "experiment"

    if not neptune_config.get("use_neptune", False):
        return None

    neptune_project = neptune_config.get("neptune_project") or os.getenv("NEPTUNE_PROJECT")
    neptune_api_token = neptune_config.get("neptune_api_token") or os.getenv("NEPTUNE_API_TOKEN")

    if not neptune_project or not neptune_api_token:
        logger.warning("Neptune tracking requested but credentials not found")
        return None

    run = neptune.init_run(
        project=neptune_project,
        name=experiment_name,
        tags=neptune_config.get("tags", []),
        api_token=neptune_api_token
    )
    # Use stringify_unsupported to handle lists and None values
    run["config"] = stringify_unsupported(neptune_config)
    return run
