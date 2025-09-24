"""Gradient Episodic Memory (GEM)"""

from typing import Any, Dict

import torch
import torch.nn as nn

from src.cl_strategies.base import BaseCLStrategy
from src.cl_strategies.memory import MemoryBuffer
from src.cl_strategies.utils import get_grad_vector, set_grad_vector


class GEM(BaseCLStrategy):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        cl_cfg = config.get("cl_strategy", {})
        mem_size = int(cl_cfg.get("memory_size", 50))
        self.memory = MemoryBuffer(mem_size)
        self.replay_batch_size = cl_cfg.get("replay_batch_size", 1)

        # Aggressive memory management
        self.clear_cache_every = 1  # Clear every step.
        self._step_count = 0

    def update_memory(self, batch: Dict[str, torch.Tensor]):
        # Store only single samples to minimize memory
        sample_batch = {
            "input_ids": batch["input_ids"][:1],
            "attention_mask": batch["attention_mask"][:1],
            "bbox": batch["bbox"][:1],
            "labels": batch["labels"][:1]
        }
        if "token_type_ids" in batch and batch["token_type_ids"] is not None:
            sample_batch["token_type_ids"] = batch["token_type_ids"][:1]
        self.memory.add_batch(sample_batch)

    def on_before_backward(self, model: nn.Module, loss: torch.Tensor):
        device = next(model.parameters()).device

        # Aggressive cache clearing to prevent OOM.
        self._step_count += 1
        if self._step_count % self.clear_cache_every == 0:
            torch.cuda.empty_cache()
            
        if len(self.memory) == 0:
            return

        # Use A-GEM projection to prevent QP memory explosion
        self._agem_projection(model, loss, device)
            
    def _agem_projection(self, model: nn.Module, loss: torch.Tensor, device):
        """A-GEM style projection - prevents OOM by avoiding QP solver."""
        # Get single reference sample (minimal memory)
        mem_batch = self.memory.sample(self.replay_batch_size, device=device)
        if mem_batch is None:
            return
        
        # Compute reference gradient.
        model.zero_grad(set_to_none=True)
        mem_out = model(**mem_batch)
        mem_loss = mem_out["loss"]
        mem_loss.backward()
        g_ref = get_grad_vector(model).detach()

        # Immediate cleanup
        del mem_batch, mem_out, mem_loss

        # Compute current gradient
        model.zero_grad(set_to_none=True)
        loss.backward()
        g = get_grad_vector(model)

        # A-GEM projection (memory-efficient, no QP solver)
        dot_product = torch.dot(g, g_ref)
        if dot_product < 0:  # Constraint violation
            # Project: g - (g·g_ref / ||g_ref||²) * g_ref
            g_ref_norm_sq = torch.dot(g_ref, g_ref)
            
            if g_ref_norm_sq > 1e-12:  # Avoid division by zero
                projection_coeff = dot_product / g_ref_norm_sq
                projected_g = g - projection_coeff * g_ref
                set_grad_vector(model, projected_g)
                
                # Cleanup
                del projected_g
                
        # Final cleanup
        del g, g_ref, dot_product
