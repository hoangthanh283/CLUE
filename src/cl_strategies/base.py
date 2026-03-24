"""
Base interfaces for Continual Learning strategies.
"""

from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn

from src.config import StrategyConfig


class BaseCLStrategy:
    """Base class for Continual Learning strategies.

    Lifecycle hook calling order per task (guaranteed by ContinualLayoutLMTrainer):
        1. before_task(model, task_id, train_loader)    -- once, before training loop
        2. per batch:
             a. compute_loss(model, batch, outputs)     -- returns scalar loss
             b. on_before_backward(model, loss_scaled)  -- before loss.backward()
             c. loss_scaled.backward()
             d. on_after_backward(model, is_final_step) -- after backward, before optimizer
             e. update_memory(batch)                    -- called every batch
        3. after_task(model, task_id, train_loader)     -- once, after training loop

    Subclasses that override before_task or after_task MUST call super() to maintain
    current_task_id and seen_tasks state.

    Attributes:
        config: Strategy configuration.
        current_task_id: Task ID currently being trained. Set by before_task.
        seen_tasks: List of task IDs whose after_task has completed.
    """

    def __init__(self, config: Union[StrategyConfig, Dict[str, Any]]) -> None:
        self.config = config
        self.current_task_id: int = 0
        self.seen_tasks: List[int] = []

    def before_task(self, model: nn.Module, task_id: int,
                    train_loader: Optional[Iterable] = None) -> None:
        """Called once before the training loop for task_id.

        Subclasses must call super().before_task(model, task_id, train_loader)
        to keep current_task_id in sync.
        """
        self.current_task_id = task_id

    def after_task(self, model: nn.Module, task_id: int,
                   train_loader: Optional[Iterable] = None) -> None:
        """Called once after the training loop for task_id.

        Subclasses must call super().after_task(model, task_id, train_loader)
        to keep seen_tasks in sync.
        """
        if task_id not in self.seen_tasks:
            self.seen_tasks.append(task_id)

    def compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor],
                     outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the loss to backprop for the current batch. Default: model loss."""
        base_loss = outputs.get("loss")
        if base_loss is None:
            raise ValueError("Model outputs must contain 'loss'")
        return base_loss

    def on_before_backward(self, model: nn.Module, loss: torch.Tensor) -> None:
        """Called right before loss.backward() to adjust gradients if needed."""
        return

    def on_after_backward(self, model: nn.Module,
                          is_final_accumulation_step: bool = True) -> None:
        """Called after loss.backward() to project/modify gradients if needed.

        Args:
            model: The model with computed gradients.
            is_final_accumulation_step: True if this is the final gradient accumulation
                                        step (i.e., optimizer.step() will be called next).
        """
        return

    def update_memory(self, batch: Dict[str, torch.Tensor]) -> None:
        """Called every batch before optimizer.step(). Default: no-op."""
        return
