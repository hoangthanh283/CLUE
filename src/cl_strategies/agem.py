"""Averaged GEM (A-GEM)."""

from typing import Any, Dict

import torch
import torch.nn as nn

from src.cl_strategies.base import BaseCLStrategy
from src.cl_strategies.memory import MemoryBuffer
from src.cl_strategies.utils import get_grad_vector, set_grad_vector


class AGEM(BaseCLStrategy):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        cl_cfg = config.get("cl_strategy", {})
        # Optimized memory settings for RTX 2060 6GB
        mem_size = int(cl_cfg.get("memory_size", 1000))  # Reduced from 2000
        self.ref_batch_size = int(cl_cfg.get("replay_batch_size", 8))  # Reduced from 32
        self.memory = MemoryBuffer(mem_size)
        
        # A-GEM specific optimizations
        self.constraint_threshold = float(cl_cfg.get("constraint_threshold", -1e-3))  # Allow small violations
        self.clear_cache_every = int(cl_cfg.get("clear_cache_every", 5))  # Cache management
        self._step_count = 0

    def update_memory(self, batch: Dict[str, torch.Tensor]):
        """Store samples with memory-efficient approach."""
        # Store only essential samples to minimize memory usage
        if len(batch["input_ids"]) > 1:
            # Take only first sample to minimize memory footprint
            sample_batch = {
                "input_ids": batch["input_ids"][:1],
                "attention_mask": batch["attention_mask"][:1],
                "bbox": batch["bbox"][:1],
                "labels": batch["labels"][:1]
            }
            if "token_type_ids" in batch and batch["token_type_ids"] is not None:
                sample_batch["token_type_ids"] = batch["token_type_ids"][:1]
        else:
            sample_batch = batch
            
        self.memory.add_batch(sample_batch)

    def on_before_backward(self, model: nn.Module, loss: torch.Tensor):
        """Memory-efficient A-GEM gradient projection."""
        device = next(model.parameters()).device
        
        # Periodic cache clearing for memory management
        self._step_count += 1
        if self._step_count % self.clear_cache_every == 0:
            torch.cuda.empty_cache()
        
        # Memory-efficient sampling - use smaller batch size
        effective_batch_size = min(self.ref_batch_size, 4)  # Cap at 4 for RTX 2060
        mem_batch = self.memory.sample(effective_batch_size, device=device)
        if mem_batch is None:
            return

        # Reference gradient from memory (averaged across batch)
        model.zero_grad(set_to_none=True)
        mem_outputs = model(**mem_batch)
        mem_loss = mem_outputs["loss"]
        mem_loss.backward()
        g_ref = get_grad_vector(model).detach()
        
        # Immediate cleanup of memory batch
        del mem_batch, mem_outputs, mem_loss

        # Current gradient
        model.zero_grad(set_to_none=True)
        loss.backward()
        g = get_grad_vector(model)

        # A-GEM constraint check with configurable threshold
        dot = torch.dot(g, g_ref)
        if dot < self.constraint_threshold:  # Constraint violation
            # Gradient projection: g - (g·g_ref / ||g_ref||²) * g_ref
            g_ref_norm_sq = torch.dot(g_ref, g_ref)
            
            if g_ref_norm_sq > 1e-12:  # Avoid division by zero
                projection_coeff = dot / g_ref_norm_sq
                projected_g = g - projection_coeff * g_ref
                set_grad_vector(model, projected_g)
                
                # Cleanup projected gradient
                del projected_g
        
        # Final cleanup
        del g, g_ref
