"""
Base interfaces for Continual Learning strategies.
"""

from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn


class BaseCLStrategy:
    """Base class for Continual Learning strategies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    # Lifecycle hooks
    def before_task(self, model: nn.Module, task_id: int, train_loader: Optional[Iterable] = None):
        pass

    def after_task(self, model: nn.Module, task_id: int, train_loader: Optional[Iterable] = None):
        pass

    # Loss computation
    def compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]
                     ) -> torch.Tensor:
        """Return the loss to backprop for the current batch. Default: model loss."""
        base_loss = outputs.get("loss")
        if base_loss is None:
            raise ValueError("Model outputs must contain 'loss'")
        return base_loss

    # Optional gradient hook for strategies like (A-)GEM
    def on_before_backward(self, model: nn.Module, loss: torch.Tensor):
        """Called right before loss.backward() to adjust gradients if needed."""
        return

    # Optional memory update (ER/GEM)
    def update_memory(self, batch: Dict[str, torch.Tensor]):
        return
