"""
Utility functions and helpers
"""

import logging
import os

import yaml


def load_config(config_path):
    """Load configuration from YAML file and merge with environment variables

    This function loads the YAML configuration file specified by `config_path`.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        dict: Configuration dictionary with environment variables merged
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not config.get("neptune"):
        return config

    # Override with environment variables if they exist.
    neptune_project = os.getenv("NEPTUNE_PROJECT")
    neptune_token = os.getenv("NEPTUNE_API_TOKEN")
    if neptune_project and neptune_token:
        config["neptune"]["neptune_project"] = neptune_project
        config["neptune"]["neptune_api_token"] = neptune_token
    return config


def setup_logging(log_dir, experiment_name):
    """Setup logging for experiments"""
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)
