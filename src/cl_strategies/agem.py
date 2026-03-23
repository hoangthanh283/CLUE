"""
Averaged Gradient Episodic Memory (A-GEM) for Continual Learning.

Memory-efficient continual learning strategy that uses averaged constraints instead of
per-task constraints like GEM, providing similar performance with O(1) constraint evaluation.

Reference: Chaudhry et al. (2019). Efficient Lifelong Learning with A-GEM. ICLR.
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from src.cl_strategies.base import BaseCLStrategy
from src.cl_strategies.memory import MemoryBuffer
from src.cl_strategies.utils import get_grad_vector, set_grad_vector


class AGEM(BaseCLStrategy):
    """
    Averaged Gradient Episodic Memory (A-GEM) strategy.

    A-GEM enforces a single averaged inequality constraint:
        g^T · g_ref >= 0

    where g_ref is the average gradient from episodic memory samples across all previous tasks.
    This ensures the average loss over previous tasks does not increase.

    Key advantages over GEM:
    - O(1) constraint evaluation (vs O(t) for GEM)
    - No QP solver required (simple projection)
    - Comparable or better performance than GEM
    - More memory and computationally efficient
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        cl_cfg = config.get("cl_strategy", {})

        # Memory settings optimized for RTX 2060 6GB
        mem_size = int(cl_cfg.get("memory_size", 1000))
        self.memory = MemoryBuffer(mem_size)

        # Reference batch size for computing average gradient
        self.ref_batch_size = int(cl_cfg.get("replay_batch_size", 4))

        # Constraint threshold (should be close to 0 for standard A-GEM)
        # Negative values allow small violations for numerical stability
        self.constraint_threshold = float(cl_cfg.get("constraint_threshold", -1e-6))

        # Memory management
        self.clear_cache_every = int(cl_cfg.get("clear_cache_every", 5))
        self._step_count = 0

        # Optional: Track task information for balanced sampling
        self.use_balanced_sampling = bool(cl_cfg.get("use_balanced_sampling", False))
        self.current_task_id = 0
        self.seen_tasks = []  # Track completed tasks

    def before_task(self, model: nn.Module, task_id: int, train_loader=None):
        """Update current task ID for tracking."""
        self.current_task_id = task_id

    def after_task(self, model: nn.Module, task_id: int, train_loader=None):
        """Mark that we've completed training on a task."""
        if task_id not in self.seen_tasks:
            self.seen_tasks.append(task_id)

    def update_memory(self, batch: Dict[str, torch.Tensor]):
        """
        Store samples from current task into episodic memory.

        Uses reservoir sampling to maintain a representative subset across all tasks.
        Stores only first sample from batch to minimize memory usage.
        """
        # Extract single sample to minimize memory footprint
        sample_batch = {
            "input_ids": batch["input_ids"][:1],
            "attention_mask": batch["attention_mask"][:1],
            "bbox": batch["bbox"][:1],
            "labels": batch["labels"][:1]
        }
        if "token_type_ids" in batch and batch["token_type_ids"] is not None:
            sample_batch["token_type_ids"] = batch["token_type_ids"][:1]

        # Pass task_id for potential future task-aware analysis
        self.memory.add_batch(sample_batch, task_id=self.current_task_id)

    def on_after_backward(self, model: nn.Module, is_final_accumulation_step: bool = True):
        """
        Apply A-GEM constraint: project gradient if it violates averaged constraint.

        This hook is called after loss.backward() has computed gradients. A-GEM projects the
        gradients to satisfy the averaged constraint from memory samples.

        Algorithm:
        1. Get current gradient g (already computed by backward pass)
        2. Compute reference gradient g_ref from memory samples
        3. Check constraint: g^T · g_ref >= 0
        4. If violated, project: g ← g - ((g^T·g_ref) / ||g_ref||²) · g_ref

        Args:
            model: Model with already-computed gradients
            is_final_accumulation_step: Only project on final accumulation step
        """
        # Only project on final gradient accumulation step
        if not is_final_accumulation_step:
            return

        device = next(model.parameters()).device

        # Periodic cache clearing for memory management
        self._step_count += 1
        if self._step_count % self.clear_cache_every == 0:
            torch.cuda.empty_cache()

        # No constraints for first task or if no previous tasks completed
        # This prevents computing constraints from current task against itself
        if self.current_task_id == 0 or len(self.seen_tasks) == 0:
            return

        # Get current task gradient (already computed by backward pass)
        g = get_grad_vector(model)

        # Sample from episodic memory
        # Use adaptive batch size based on memory availability
        effective_batch_size = min(self.ref_batch_size, len(self.memory))
        mem_batch = self.memory.sample(effective_batch_size, device=device)

        if mem_batch is None:
            return

        # Compute reference gradient from memory (averaged across sampled tasks)
        # We need to temporarily store current gradient and compute reference gradient
        g_current = g.clone()
        model.zero_grad(set_to_none=True)
        mem_outputs = model(**mem_batch)
        mem_loss = mem_outputs["loss"]
        mem_loss.backward()
        g_ref = get_grad_vector(model).detach()

        # Restore current gradient
        set_grad_vector(model, g_current)

        # Cleanup memory batch
        del mem_batch, mem_outputs, mem_loss
        torch.cuda.empty_cache()

        # A-GEM constraint check: g^T · g_ref >= 0
        dot_product = torch.dot(g_current, g_ref)

        if dot_product < self.constraint_threshold:
            # Constraint violated: project gradient
            # Projection formula: g ← g - ((g^T·g_ref) / ||g_ref||²) · g_ref
            g_ref_norm_sq = torch.dot(g_ref, g_ref)

            if g_ref_norm_sq > 1e-12:  # Avoid division by zero
                projection_coeff = dot_product / g_ref_norm_sq
                projected_g = g_current - projection_coeff * g_ref
                set_grad_vector(model, projected_g)

                # Cleanup
                del projected_g

        # Final cleanup
        del g, g_current, g_ref, dot_product
        torch.cuda.empty_cache()
