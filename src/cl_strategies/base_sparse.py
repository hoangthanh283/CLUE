"""Base class for sparse/subnetwork continual learning strategies.

This module provides a foundation for implementing sparse and subnetwork-based
continual learning methods such as PackNet, SupSup, WSN, SMFT, and SparseDocCL.

These methods maintain task-specific masks or weights that allow efficient reuse
of the base model across multiple tasks while preventing interference.
"""

from typing import Any, Dict, Generator, Tuple

import torch
import torch.nn as nn

from src.cl_strategies.base import BaseCLStrategy

# Parameter name prefixes that identify the classifier head.
# Adjust if your model uses a different naming convention.
_CLASSIFIER_PREFIXES = ("classifier.", "cls_head.", "output_head.")


class BaseSparseCLStrategy(BaseCLStrategy):
    """Base class for sparse and subnetwork continual learning strategies.

    Provides common functionality for strategies that use masks or sparse weights:
    - PackNet: Progressive neural network masks
    - SupSup: Superposition of masks
    - WSN: Wafer-scale networks (implicit masking)
    - SMFT: Sparse Mixture of Fine-Tuned Experts
    - SparseDocCL: Sparse document continual learning

    Subclasses should implement:
    - before_task(): Initialize task-specific structures
    - after_task(): Store/finalize task masks
    - compute_loss(): Add sparsity regularization if needed
    - on_after_backward(): Apply gradient masking via mask_gradients()
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.task_masks: Dict[int, Dict[str, torch.Tensor]] = {}
        # current_task_id inherited from BaseCLStrategy

    def get_named_backbone_params(self, model: nn.Module) -> Generator[Tuple[str, nn.Parameter], None, None]:
        """Yield (name, param) for backbone parameters only, excluding classifier head.

        Uses prefix matching against _CLASSIFIER_PREFIXES to avoid accidentally
        excluding attention head parameters whose names contain the word "head".

        Args:
            model: The model to extract backbone parameters from.

        Yields:
            (parameter_name, parameter_tensor) for all non-classifier params.
        """
        for name, param in model.named_parameters():
            if not any(name.startswith(prefix) for prefix in _CLASSIFIER_PREFIXES):
                yield name, param

    def zero_weights_for_task(self, model: nn.Module, task_id: int, device: torch.device):
        """Destructively zero model weights outside the task mask.

        WARNING: This permanently modifies model weights in-place. It is intended
        for single-task evaluation where the model will be reloaded between tasks.
        Do NOT call this during training or when the model needs to serve multiple tasks.

        For non-destructive per-forward-pass masking, apply masks inside a forward
        hook instead:
            handle = model.register_forward_hook(lambda m, i, o: ...)

        Args:
            model: Model whose weights will be zeroed outside task_id's mask.
            task_id: Task to keep active.
            device: Device to move mask tensors to.
        """
        if task_id not in self.task_masks:
            return

        masks = self.task_masks[task_id]
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in masks:
                    param.data.mul_(masks[name].to(device))

    def save_task_mask(self, task_id: int, mask_dict: Dict[str, torch.Tensor]):
        """Store binary mask for a task (call from after_task).

        Args:
            task_id: Task ID to save mask for.
            mask_dict: Maps parameter names to binary (0/1) or boolean masks.
        """
        self.task_masks[task_id] = {
            name: mask.detach().cpu() for name, mask in mask_dict.items()
        }

    def mask_gradients(self, model: nn.Module, frozen_mask: Dict[str, torch.Tensor]):
        """Zero gradients for frozen (masked-out) parameters.

        Call from on_after_backward() to implement sparse gradient updates.
        Gradients for parameters with mask value 0 are zeroed, preserving
        gradient flow only for the active subnetwork.

        Args:
            model: Model with already-computed gradients.
            frozen_mask: Maps parameter names to masks (1 = update, 0 = freeze).
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None and name in frozen_mask:
                    param.grad.mul_(frozen_mask[name].to(param.device))

    def before_task(self, model: nn.Module, task_id: int, train_loader=None) -> None:
        """Set current task ID. Subclasses should call super() and add mask setup."""
        super().before_task(model, task_id, train_loader)

    def after_task(self, model: nn.Module, task_id: int, train_loader=None) -> None:
        """Finalize task masks. Subclasses should override to compute and save masks."""
        super().after_task(model, task_id, train_loader)
