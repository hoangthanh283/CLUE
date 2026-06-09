"""Abstract base class for all continual learning methods.

The lifecycle is:
    for task in scenario.tasks:
        method.before_task(task)     # extend classifier, allocate adapters
        method.train_task(task)      # main training loop
        method.after_task(task)      # update buffer, compute Fisher, freeze adapters
        results = method.evaluate(eval_loaders_seen_so_far)

Subclasses override hooks they need; the rest are no-ops by default.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from doccl.types import EvalMetrics, TaskInfo, TaskState, TrainMetrics


class ContinualMethod(ABC):
    """Base class for all CL methods. Subclasses override what they need."""

    name: str = "base"  # override in subclasses for W&B tagging

    def __init__(self, model: nn.Module, config: dict[str, Any]):
        self.model = model
        self.config = config
        self.state = TaskState()
        self.device = next(model.parameters()).device

    # ─── Lifecycle hooks ───────────────────────────────────────────────────────
    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Called once before training on a new task.

        Common uses:
            - Extend classifier head for class-incremental learning
            - Allocate new LoRA bank for parameter-efficient methods
            - Cache teacher model for distillation methods
        """
        pass

    @abstractmethod
    def train_task(self, task: TaskInfo, train_loader: DataLoader) -> TrainMetrics:
        """Train on the current task. Returns training metrics."""
        ...

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Called once after training on a task.

        Common uses:
            - Update replay buffer (ER, DER++, iCaRL)
            - Compute Fisher information matrix (EWC)
            - Snapshot model as teacher (LwF)
            - Freeze current LoRA bank, allocate next
        """
        pass

    @abstractmethod
    def evaluate(
        self, eval_loaders: dict[int, DataLoader]
    ) -> dict[int, EvalMetrics]:
        """Evaluate on all tasks seen so far.

        Args:
            eval_loaders: {task_id: DataLoader} for all tasks seen up to now.

        Returns:
            {task_id: EvalMetrics}
        """
        ...

    # ─── Helpers ───────────────────────────────────────────────────────────────
    def trainable_parameters(self) -> list[nn.Parameter]:
        """All parameters with requires_grad=True (default optimizer target)."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def state_dict(self) -> dict[str, Any]:
        """Serialize method state for checkpointing.

        Subclasses override to save buffers, prompt pools, Fisher matrices, etc.
        """
        return {
            "name": self.name,
            "config": self.config,
            "model_state": self.model.state_dict(),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """Restore method state from checkpoint."""
        self.model.load_state_dict(sd["model_state"])
