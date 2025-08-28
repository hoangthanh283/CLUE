"""
Utility functions and helpers
"""

import logging
import os

import yaml


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
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


def save_results(results, output_path):
    """Save experiment results"""
    # TODO: Implement result saving
    pass
