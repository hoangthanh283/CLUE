"""Replay buffer with reservoir sampling.

Used by ER (stores raw samples) and DER++ (also stores logits).
"""
from __future__ import annotations

import random
from typing import Any

import torch


class ReservoirBuffer:
    """Reservoir sampling buffer maintaining a fixed-size i.i.d. sample of all
    examples seen across tasks.

    Storage strategy: lazy — examples stored as dicts of CPU tensors.
    For document data, each example is ~150KB raw image + small text/box tensors.
    Buffer of size 200 ≈ 30MB — fits easily on GPU host.
    """

    def __init__(self, capacity: int = 200, store_logits: bool = False):
        self.capacity = capacity
        self.store_logits = store_logits
        self.buffer: list[dict[str, Any]] = []
        self.n_seen = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add_batch(self, batch: dict[str, torch.Tensor], logits: torch.Tensor | None = None) -> None:
        """Add each example in batch to the buffer via reservoir sampling.

        Args:
            batch: dict of (B, ...) tensors
            logits: (B, L, C) tensor — required if store_logits=True (for DER++)
        """
        B = next(iter(batch.values())).shape[0]
        for i in range(B):
            example = {k: v[i].detach().cpu().clone() for k, v in batch.items() if torch.is_tensor(v)}
            if self.store_logits:
                if logits is None:
                    raise ValueError("store_logits=True requires logits argument")
                example["_logits"] = logits[i].detach().cpu().clone()

            if len(self.buffer) < self.capacity:
                self.buffer.append(example)
            else:
                # Reservoir replacement: keep with probability capacity/n_seen
                idx = random.randint(0, self.n_seen)
                if idx < self.capacity:
                    self.buffer[idx] = example
            self.n_seen += 1

    def sample(self, batch_size: int) -> dict[str, torch.Tensor] | None:
        """Sample a batch from the buffer. Returns None if buffer empty."""
        if not self.buffer:
            return None
        bs = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), bs)
        sampled = [self.buffer[i] for i in indices]

        # Stack into batch
        keys = sampled[0].keys()
        out = {}
        for k in keys:
            out[k] = torch.stack([ex[k] for ex in sampled], dim=0)
        return out

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "store_logits": self.store_logits,
            "buffer": self.buffer,
            "n_seen": self.n_seen,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.capacity = sd["capacity"]
        self.store_logits = sd["store_logits"]
        self.buffer = sd["buffer"]
        self.n_seen = sd["n_seen"]
