"""Mixin for episodic memory-based continual learning strategies (GEM, A-GEM)."""

from typing import Dict, List

import torch
import torch.nn as nn


class EpisodicMemoryMixin:
    """Mixin providing shared episodic memory tracking for GEM/A-GEM strategies.

    Initializes current_task_id and seen_tasks via cooperative __init__.
    Subclasses must set self.memory = MemoryBuffer(...) before update_memory is called.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_task_id: int = 0
        self.seen_tasks: List[int] = []

    def before_task(self, model: nn.Module, task_id: int, train_loader=None):
        """Mark the start of a new task."""
        self.current_task_id = task_id

    def after_task(self, model: nn.Module, task_id: int, train_loader=None):
        """Mark that we've completed training on a task."""
        if task_id not in self.seen_tasks:
            self.seen_tasks.append(task_id)

    def update_memory(self, batch: Dict[str, torch.Tensor]):
        """Store samples from the current task into episodic memory.

        Uses reservoir sampling to maintain a representative subset across all tasks.
        Stores only first sample from batch to minimize memory usage.
        Subclass must set self.memory = MemoryBuffer(...) before calling this.

        Args:
            batch: Training batch with 'input_ids', 'attention_mask', 'bbox', 'labels',
                   and optional 'token_type_ids'.
        """
        sample_batch = {
            "input_ids": batch["input_ids"][:1],
            "attention_mask": batch["attention_mask"][:1],
            "bbox": batch["bbox"][:1],
            "labels": batch["labels"][:1],
        }
        if "token_type_ids" in batch and batch["token_type_ids"] is not None:
            sample_batch["token_type_ids"] = batch["token_type_ids"][:1]

        self.memory.add_batch(sample_batch, task_id=self.current_task_id)
