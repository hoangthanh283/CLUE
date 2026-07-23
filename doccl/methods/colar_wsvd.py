"""CoLaR-WSVD — CoLaR with entity-weighted per-document SVD compression.

CoLaR-Bal changed the replay loss and hit the conservation frontier. This variant changes
the compression objective instead: the document is still compressed whole, but entity-token
rows get a higher reconstruction weight before SVD. Same factor shapes, same byte cost,
different approximation error allocation.
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from doccl.methods.colar import CoLaR
from doccl.methods.latent_replay import LatentReplay

log = logging.getLogger(__name__)

__all__ = ["CoLaRWSVD"]


class CoLaRWSVD(CoLaR):
    """Compressed latent replay with entity-weighted per-doc SVD."""

    name = "colar_wsvd"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.svd_entity_weight = float(config.get("svd_entity_weight", 4.0))
        self._o_label_id = getattr(model, "label_to_id", {}).get("O", 0)

    @staticmethod
    def _weighted_factors(
        hidden: torch.Tensor,
        labels: torch.Tensor,
        rank: int,
        o_label_id: int = 0,
        entity_weight: float = 4.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Weighted low-rank factors for min ||W^(1/2)(H - H_hat)||_F.

        Store ``us`` already divided by sqrt(row_weight), so inherited reconstruction
        ``us @ v`` returns the weighted approximation in original hidden space.
        """
        h = hidden.float()
        labels = labels.flatten().to(h.device)
        weights = torch.ones(h.shape[0], dtype=h.dtype, device=h.device)
        n_text = min(h.shape[0], labels.shape[0])
        entity = (labels[:n_text] != -100) & (labels[:n_text] != o_label_id)
        weights[:n_text][entity] = max(float(entity_weight), 1.0)
        sqrt_w = weights.sqrt().unsqueeze(1)
        u, s, vh = torch.linalg.svd(h * sqrt_w, full_matrices=False)
        r = min(rank, s.shape[0])
        us = (u[:, :r] * s[:r]) / sqrt_w
        return us.to(torch.float16), vh[:r].to(torch.float16)

    def _capture_task(self, train_loader: DataLoader) -> None:
        """Bank raw layer-k docs, then compress new docs with weighted SVD."""
        n_before = len(self.store)
        LatentReplay._capture_task(self, train_loader)
        for d in self.store[n_before:]:
            d["us"], d["v"] = self._weighted_factors(
                d.pop("hidden"),
                d["labels"],
                self.rank_r,
                self._o_label_id,
                self.svd_entity_weight,
            )
        log.info(
            "colar_wsvd: compressed %d docs at rank-%d entity_weight=%.1f (bank ~%.1f MB)",
            len(self.store) - n_before,
            self.rank_r,
            self.svd_entity_weight,
            self.memory_bytes() / 1e6,
        )
