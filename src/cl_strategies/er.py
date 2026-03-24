"""Experience Replay (ER).

This is a vanilla implementation of Experience Replay following:
    Rolnick et al. (2019) "Experience Replay for Continual Learning"

The algorithm maintains a fixed-size memory buffer using reservoir sampling
and combines the loss from current data with replayed examples:

    L_total = L(current batch) + λ * L(replay batch)

This implementation is suitable as a baseline for continual learning research.
"""

from typing import Any, Dict, Union

import torch
import torch.nn as nn

from src.cl_strategies.base import BaseCLStrategy
from src.cl_strategies.memory import MemoryBuffer
from src.config import ERConfig


class ExperienceReplay(BaseCLStrategy):
    """Vanilla Experience Replay with reservoir sampling memory.

    References:
        Rolnick et al. (2019) "Experience Replay for Continual Learning"
        https://arxiv.org/abs/1811.11682

    Algorithm:
        1. Store training examples in a fixed-size memory buffer using reservoir sampling
        2. During training on new tasks, sample a batch from memory
        3. Combine losses: L_total = L(current) + replay_weight * L(memory)

    This is a basic implementation without class-balancing or other enhancements,
    making it suitable as a vanilla baseline for comparisons.
    """

    def __init__(self, config: Union[ERConfig, Dict[str, Any]]):
        super().__init__(config)
        if isinstance(config, dict):
            clcfg = config.get("cl_strategy", {})
            mem_size = int(clcfg.get("memory_size", 2000))
            self.replay_batch_size = int(clcfg.get("replay_batch_size", 32))
            self.replay_weight = float(clcfg.get("replay_weight", 1.0))
        else:
            mem_size = config.memory_size
            self.replay_batch_size = config.replay_batch_size
            self.replay_weight = config.replay_weight
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
