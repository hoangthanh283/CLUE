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

        # AGENT FIX v2: On GPU with limited VRAM (5.6 GiB), the main forward pass
        # consumes ~5.4 GiB. Even after base_loss.backward() frees activations,
        # the replay batch (replay_batch_size=4, seq_len=512) still OOMs because
        # LayoutLMv3 attention allocates O(batch*seq^2) tensors.
        # Fix: backward base_loss first (frees ~200 MiB of activations), then
        # process each replay sample one-at-a-time to keep peak memory at O(seq^2).
        # Gradients are accumulated across all replay samples and the mean is applied.
        # Net gradient effect is identical to batched replay; only memory layout differs.
        # Assumes gradient_accumulation_steps=1 (true for ER config).
        base_loss.backward(retain_graph=False)
        torch.cuda.empty_cache()

        # Count how many samples are actually in mem_batch (first tensor's batch dim)
        first_val = next(iter(mem_batch.values()))
        n_replay = first_val.shape[0]
        accumulated_mem_loss = torch.tensor(0.0, device=device, requires_grad=False)
        for i in range(n_replay):
            single_sample = {k: v[i:i + 1] for k, v in mem_batch.items()}
            single_out = model(**single_sample)
            single_loss = self.replay_weight * single_out["loss"] / n_replay
            single_loss.backward()
            accumulated_mem_loss = accumulated_mem_loss + single_loss.detach()
            torch.cuda.empty_cache()

        # Return a zero-grad tensor so the outer loop's loss.backward() is a no-op
        # (all replay gradients already accumulated above).
        return torch.tensor(0.0, device=device, requires_grad=True)

    def update_memory(self, batch: Dict[str, torch.Tensor]):
        self.memory.add_batch(batch)
