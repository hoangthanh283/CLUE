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

    if not config.get("wandb"):
        return config

    # Override with environment variables if they exist.
    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb_project = os.getenv("WANDB_PROJECT")
    wandb_entity = os.getenv("WANDB_ENTITY")
    if wandb_api_key:
        config["wandb"]["wandb_api_key"] = wandb_api_key
    if wandb_project:
        config["wandb"]["wandb_project"] = wandb_project
    if wandb_entity:
        config["wandb"]["wandb_entity"] = wandb_entity
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
