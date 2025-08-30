"""
Episodic memory buffer utilities for ER/GEM/A-GEM.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class MemoryItem:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    bbox: torch.Tensor
    labels: torch.Tensor
    token_type_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    image: Optional[torch.Tensor] = None
    pixel_values: Optional[torch.Tensor] = None


class MemoryBuffer:
    """Reservoir-sampling episodic memory buffer."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.items: List[MemoryItem] = []
        self.n_seen = 0

    def __len__(self) -> int:
        return len(self.items)

    def add_batch(self, batch: Dict[str, torch.Tensor]):
        bsz = batch["input_ids"].size(0)
        for i in range(bsz):
            self.n_seen += 1
            item = MemoryItem(
                input_ids=batch["input_ids"][i].detach().cpu(),
                attention_mask=batch["attention_mask"][i].detach().cpu(),
                bbox=batch["bbox"][i].detach().cpu(),
                labels=batch["labels"][i].detach().cpu(),
                token_type_ids=batch.get("token_type_ids", None)[i].detach().cpu()
                if batch.get("token_type_ids", None) is not None
                else None,
                position_ids=batch.get("position_ids", None)[i].detach().cpu()
                if batch.get("position_ids", None) is not None
                else None,
                image=batch.get("image", None)[i].detach().cpu()
                if batch.get("image", None) is not None
                else None,
                pixel_values=batch.get("pixel_values", None)[i].detach().cpu()
                if batch.get("pixel_values", None) is not None
                else None,
            )

            if len(self.items) < self.capacity:
                self.items.append(item)
            else:
                j = random.randint(0, self.n_seen - 1)
                if j < self.capacity:
                    self.items[j] = item

    def sample(self, batch_size: int, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        if len(self.items) == 0:
            return None
        batch_size = min(batch_size, len(self.items))
        samples = random.sample(self.items, batch_size)
        collated: Dict[str, List[torch.Tensor]] = {}
        keys = [
            "input_ids",
            "attention_mask",
            "bbox",
            "labels",
            "token_type_ids",
            "position_ids",
            "image",
            "pixel_values",
        ]
        for k in keys:
            vals = [getattr(it, k) for it in samples if getattr(it, k) is not None]
            if len(vals) == 0:
                continue
            collated[k] = torch.stack(vals, dim=0).to(device)
        return collated
