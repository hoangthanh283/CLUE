"""Experience Replay (ER).

This is a vanilla implementation of Experience Replay following:
    Rolnick et al. (2019) "Experience Replay for Continual Learning"

The algorithm maintains a fixed-size memory buffer using reservoir sampling
and combines the loss from current data with replayed examples:

    L_total = L(current batch) + λ * L(replay batch)

This implementation is suitable as a baseline for continual learning research.
"""

import gc
import logging
from typing import Any, Dict, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

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

        # AGENT FIX v3: On GPU with limited VRAM (5.6 GiB), the main forward pass
        # consumes ~5.4 GiB. Even after base_loss.backward() frees activations,
        # memory fragmentation can cause OOM on replay forward passes after many
        # training steps. Fix: backward base_loss first, then explicitly run GC and
        # empty CUDA cache. Each replay sample is processed individually with a
        # try-except OOM guard — if a sample causes OOM, skip it and log a warning
        # rather than crashing the entire training run. This is a graceful degradation
        # for low-VRAM GPUs; the gradient estimate is noisier but training continues.
        # Assumes gradient_accumulation_steps=1 (true for ER config).
        base_loss.backward(retain_graph=False)
        gc.collect()
        torch.cuda.empty_cache()

        # Count how many samples are actually in mem_batch (first tensor's batch dim)
        first_val = next(iter(mem_batch.values()))
        n_replay = first_val.shape[0]
        n_processed = 0
        for i in range(n_replay):
            single_sample = {k: v[i:i + 1] for k, v in mem_batch.items()}
            try:
                single_out = model(**single_sample)
                single_loss = self.replay_weight * single_out["loss"] / n_replay
                single_loss.backward()
                n_processed += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(
                        "AGENT FIX v3: OOM on replay sample %d/%d — skipping. "
                        "Allocated: %.1f MiB",
                        i, n_replay,
                        torch.cuda.memory_allocated(device) / 1024 ** 2,
                    )
                    torch.cuda.empty_cache()
                else:
                    raise
            finally:
                torch.cuda.empty_cache()

        if n_processed == 0:
            logger.warning("AGENT FIX v3: All %d replay samples skipped (OOM). "
                           "Step has only base_loss gradient.", n_replay)

        # Return a zero-grad tensor so the outer loop's loss.backward() is a no-op
        # (all replay gradients already accumulated above).
        return torch.tensor(0.0, device=device, requires_grad=True)

    def update_memory(self, batch: Dict[str, torch.Tensor]):
        self.memory.add_batch(batch)
