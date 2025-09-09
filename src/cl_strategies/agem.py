"""Averaged GEM (A-GEM)."""

import logging
from typing import Any, Dict

import torch
import torch.nn as nn

from src.cl_strategies.base import BaseCLStrategy
from src.cl_strategies.memory import MemoryBuffer
from src.cl_strategies.utils import get_grad_vector, set_grad_vector

logger = logging.getLogger(__name__)


class AGEM(BaseCLStrategy):
    """Averaged Gradient Episodic Memory (A-GEM) implementation.

    A-GEM is a more efficient version of GEM that uses a single averaged reference
    gradient instead of solving a quadratic programming problem. The key insight is
    to project the current gradient onto the orthogonal space of the average
    reference gradient when they conflict (negative dot product).

    Reference: https://arxiv.org/abs/1812.00420
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        cl_cfg = config.get("cl_strategy", {})
        mem_size = int(cl_cfg.get("memory_size", 2000))
        self.ref_batch_size = int(cl_cfg.get("replay_batch_size", 32))
        self.memory = MemoryBuffer(mem_size)

        # Statistics for monitoring
        self.projection_count = 0
        self.total_steps = 0

    def update_memory(self, batch: Dict[str, torch.Tensor]):
        self.memory.add_batch(batch)

    def on_before_backward(self, model: nn.Module, loss: torch.Tensor):
        """A-GEM gradient projection using proper computation graph handling."""
        device = next(model.parameters()).device
        mem_batch = self.memory.sample(self.ref_batch_size, device=device)
        if mem_batch is None:
            return

        # Step 1: Compute reference gradient from memory batch
        # Clear gradients and compute memory gradient with proper graph management
        model.zero_grad(set_to_none=True)
        mem_outputs = model(**mem_batch)
        mem_loss = mem_outputs["loss"]
        mem_loss.backward(retain_graph=False)
        g_ref = get_grad_vector(model).detach().clone()

        # Step 2: Compute current task gradient
        model.zero_grad(set_to_none=True)
        loss.backward()
        g = get_grad_vector(model)

        # Step 3: Apply A-GEM projection if gradients conflict
        dot_product = torch.dot(g, g_ref)
        self.total_steps += 1

        if dot_product < 0:
            # Project: g' = g - (g·g_ref / ||g_ref||²) * g_ref
            g_ref_norm_sq = g_ref.norm() ** 2 + 1e-12  # Numerical stability
            projection_coeff = dot_product / g_ref_norm_sq
            g_projected = g - projection_coeff * g_ref

            # Verify projection is orthogonal (for debugging)
            new_dot = torch.dot(g_projected, g_ref)
            if abs(new_dot) > 1e-6:
                logger.warning(f"A-GEM projection not orthogonal: {new_dot:.8f}")

            # Set the projected gradient
            set_grad_vector(model, g_projected)
            self.projection_count += 1

            # Log projection for debugging
            logger.debug(f"A-GEM projection applied: dot={dot_product:.6f}, "
                         f"grad_norm: {g.norm():.6f} -> {g_projected.norm():.6f}")
        else:
            logger.debug(f"No A-GEM projection needed: dot={dot_product:.6f}")

    def get_projection_stats(self) -> Dict[str, float]:
        """Get statistics about projection frequency."""
        if self.total_steps == 0:
            return {"projection_rate": 0.0, "total_steps": 0, "projections": 0}
        return {
            "projection_rate": self.projection_count / self.total_steps,
            "total_steps": self.total_steps,
            "projections": self.projection_count
        }
