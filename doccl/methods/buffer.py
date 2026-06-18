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
            example = {
                k: v[i].detach().cpu().clone() for k, v in batch.items() if torch.is_tensor(v)
            }
            if self.store_logits:
                if logits is None:
                    raise ValueError("store_logits=True requires logits argument")
                example["_logits"] = logits[i].detach().cpu().clone()
                # Record the ORIGINAL logit width (number of classes) for this example.
                # The head grows across CIL tasks, so sampled batches mix widths and
                # get zero-padded to the max in sample(); DER++ must MSE only over each
                # example's true width, NOT against the padding zeros (which would
                # spuriously suppress new-class logits on replayed old-task inputs).
                example["_logit_width"] = torch.tensor(logits[i].shape[-1], dtype=torch.long)

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
            tensors = [ex[k] for ex in sampled]
            # DER++ caches per-example "_logits" of shape (L, C). The classifier head
            # grows across class-incremental tasks, so the buffer can hold logits of
            # different widths C (e.g. 13 from task 0, 25 from task 1). Right-pad each
            # to the max C with zeros before stacking so they form one tensor. The pad
            # columns are NOT neutral, so der.py masks the MSE per example using the
            # stored "_logit_width" (never distilling against the padding zeros).
            if k == "_logits" and len({t.shape[-1] for t in tensors}) > 1:
                max_c = max(t.shape[-1] for t in tensors)
                tensors = [
                    (
                        t
                        if t.shape[-1] == max_c
                        else torch.nn.functional.pad(t, (0, max_c - t.shape[-1]))
                    )
                    for t in tensors
                ]
            out[k] = torch.stack(tensors, dim=0)
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
