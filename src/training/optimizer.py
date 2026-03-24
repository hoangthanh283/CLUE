"""Shared optimizer builders for training."""

import logging
from typing import Union

import torch.nn as nn
from torch.optim import AdamW

from src.config import TrainingConfig

logger = logging.getLogger(__name__)


def build_adamw_optimizer(model: nn.Module, training_config: Union[TrainingConfig, dict]) -> AdamW:
    """Build AdamW optimizer with weight decay for bias/LayerNorm excluded.

    Args:
        model: Model with parameters to optimize.
        training_config: TrainingConfig dataclass (or legacy dict) with learning_rate and weight_decay.

    Returns:
        AdamW optimizer instance.
    """
    if isinstance(training_config, dict):
        learning_rate = training_config["learning_rate"]
        weight_decay = training_config.get("weight_decay", 0.01)
    else:
        learning_rate = training_config.learning_rate
        weight_decay = training_config.weight_decay

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
    logger.info(f"Setup adamw optimizer with lr={learning_rate}")
    return optimizer
