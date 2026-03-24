"""Episodic memory utilities for continual learning strategies (GEM, A-GEM)."""

from typing import TYPE_CHECKING, Dict

import torch

if TYPE_CHECKING:
    from src.cl_strategies.memory import MemoryBuffer


class EpisodicMemoryMixin:
    """Deprecated. Functionality absorbed into BaseCLStrategy.

    BaseCLStrategy now provides current_task_id, seen_tasks, before_task, and
    after_task directly. Drop EpisodicMemoryMixin from the inheritance list:

        # Before:
        class GEM(EpisodicMemoryMixin, BaseCLStrategy): ...
        # After:
        class GEM(BaseCLStrategy): ...
    """
    pass


def store_episodic_sample(
    memory: "MemoryBuffer",
    batch: Dict[str, torch.Tensor],
    task_id: int,
) -> None:
    """Store first sample from batch into episodic memory with task_id.

    Stores only the first sample to minimise memory usage while maintaining
    task-aware sampling capability for GEM-style constraint gradients.

    Args:
        memory: MemoryBuffer instance to store the sample in.
        batch: Training batch with 'input_ids', 'attention_mask', 'bbox', 'labels',
               and optional 'token_type_ids'.
        task_id: Task ID to tag the stored sample with.
    """
    sample_batch: Dict[str, torch.Tensor] = {
        "input_ids": batch["input_ids"][:1],
        "attention_mask": batch["attention_mask"][:1],
        "bbox": batch["bbox"][:1],
        "labels": batch["labels"][:1],
    }
    if "token_type_ids" in batch and batch["token_type_ids"] is not None:
        sample_batch["token_type_ids"] = batch["token_type_ids"][:1]
    memory.add_batch(sample_batch, task_id=task_id)
