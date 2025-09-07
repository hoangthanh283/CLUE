"""Experience Replay (ER)."""

from typing import Any, Dict

import torch
import torch.nn as nn

from src.cl_strategies.base import BaseCLStrategy
from src.cl_strategies.memory import MemoryBuffer


class ExperienceReplay(BaseCLStrategy):
    """Experience Replay with reservoir sampling memory.

    L_total = L(current) + lambda * L(memory)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        clcfg = config.get("cl_strategy", {})
        mem_size = int(clcfg.get("memory_size", 2000))
        self.replay_batch_size = int(clcfg.get("replay_batch_size", 32))
        self.replay_weight = float(clcfg.get("replay_weight", 1.0))
        self.memory = MemoryBuffer(mem_size)

    def compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]
                     ) -> torch.Tensor:
        device = next(model.parameters()).device
        base_loss = outputs["loss"]
        mem_batch = self.memory.sample(self.replay_batch_size, device=device)
        if mem_batch is None:
            return base_loss

        mem_outputs = model(**mem_batch)
        mem_loss = mem_outputs["loss"]
        return base_loss + self.replay_weight * mem_loss

    def update_memory(self, batch: Dict[str, torch.Tensor]):
        self.memory.add_batch(batch)
