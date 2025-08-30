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
        mem_size = int(cl_cfg.get("memory_size", 2000))
        self.ref_batch_size = int(cl_cfg.get("replay_batch_size", 32))
        self.memory = MemoryBuffer(mem_size)

    def update_memory(self, batch: Dict[str, torch.Tensor]):
        self.memory.add_batch(batch)

    def on_before_backward(self, model: nn.Module, loss: torch.Tensor):
        device = next(model.parameters()).device
        mem_batch = self.memory.sample(self.ref_batch_size, device=device)
        if mem_batch is None:
            return

        # Reference gradient from memory
        model.zero_grad(set_to_none=True)
        mem_outputs = model(**mem_batch)
        mem_loss = mem_outputs["loss"]
        mem_loss.backward()
        g_ref = get_grad_vector(model).detach()

        # Current gradient
        model.zero_grad(set_to_none=True)
        loss.backward()
        g = get_grad_vector(model)

        dot = torch.dot(g, g_ref)
        if dot < 0:
            proj = g - (dot / (g_ref.norm() ** 2 + 1e-12)) * g_ref
            set_grad_vector(model, proj)
