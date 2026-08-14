"""CoLaR — Compressed Latent Replay: per-document rank-r SVD of banked latents.

Consistency-law-prescribed (STATE.md GATE-0 FINAL): whole-document (feature, position,
label) binding is the necessary replay ingredient — marginal memories (class Gaussians,
coresets, carrier scaffolds) all collapse. CoLaR therefore compresses each banked document
AS A WHOLE — a rank-``r`` SVD of its (seq × d) layer-``k`` activation matrix — never pooling
across documents, so the binding survives while the bytes shrink.

Why this can work where SLR could not: the POOLED token-feature space is near-full-rank
(rank-16 = 43% variance), but a SINGLE document's activation matrix is low-rank
(measured, layer-8 LayoutLMv3: r64 = 87.7±3.4%, r128 = 95.2±1.9% variance). Per-doc
factors (u·s, v) at r=64 cost ~(709+768)·64·2 bytes ≈ 0.19 MB vs 1.09 MB raw fp16 — 5.7×;
the dil k4/d50 bank drops ~163 MB → ~28 MB (below the raw d5 bank) while keeping d50's
coverage. Prior art to cite: REMIND (PQ-quantized CNN latents), ACAE-REMIND (auxiliary-AE
compression) — both classification, codebook/AE-based; per-doc SVD in a frozen transformer
space for structured doc-IE, motivated by the consistency law, is the delta.

Everything except the storage format is inherited from ``LatentReplay`` (freeze map,
capture, injection hook, replay loss, doc_selection knob). ``rank_r`` sweeps the
accuracy-vs-bytes Pareto curve; r ≥ min(seq, d) degenerates to (near-)lossless.
"""

from __future__ import annotations

import logging
import random

import torch
from torch.utils.data import DataLoader

from doccl.methods.latent_replay import LatentReplay

log = logging.getLogger(__name__)

__all__ = ["CoLaR"]


class CoLaR(LatentReplay):
    """Latent replay with per-document rank-``r`` compressed storage."""

    name = "colar"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.rank_r = int(config.get("rank_r", 64))

    def _capture_task(self, train_loader: DataLoader) -> None:
        """Bank via the inherited path, then compress each new doc in place:
        hidden (seq, d) → u·s (seq, r) + v (r, d), fp16."""
        n_before = len(self.store)
        super()._capture_task(train_loader)
        for d in self.store[n_before:]:
            h = d.pop("hidden").float()  # (seq, d)
            u, s, vh = torch.linalg.svd(h, full_matrices=False)
            r = min(self.rank_r, s.shape[0])
            d["us"] = (u[:, :r] * s[:r]).to(torch.float16)  # (seq, r)
            d["v"] = vh[:r].to(torch.float16)  # (r, d)
        log.info(
            "colar: compressed %d docs at rank-%d (bank now ~%.1f MB vs ~%.1f MB raw)",
            len(self.store) - n_before,
            self.rank_r,
            self.memory_bytes() / 1e6,
            sum(d["us"].shape[0] * d["v"].shape[1] for d in self.store) * 2 / 1e6,
        )

    def _sample_indices(self) -> list[int]:
        """Indices for one replay batch; variants may stratify this selection."""
        return random.sample(range(len(self.store)), min(self.replay_batch_size, len(self.store)))

    def _stack_replay(self, docs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Reconstruct a specified set of stored documents as one replay batch."""
        replay = {
            "hidden": torch.stack(
                [(doc["us"].float() @ doc["v"].float()).to(torch.float16) for doc in docs]
            ),
            "bbox": torch.stack([doc["bbox"] for doc in docs]),
            "attention_mask": torch.stack([doc["attention_mask"] for doc in docs]),
            "labels": torch.stack([doc["labels"] for doc in docs]),
        }
        if "input_ids" in docs[0]:
            replay["input_ids"] = torch.stack([doc["input_ids"] for doc in docs])
        return replay

    def _sample_replay(self) -> dict[str, torch.Tensor] | None:
        """Sample docs and reconstruct their hiddens from the per-doc factors."""
        if not self.store:
            return None
        return self._stack_replay([self.store[i] for i in self._sample_indices()])

    def memory_bytes(self) -> int:
        total = 0
        for d in self.store:
            total += (d["us"].numel() + d["v"].numel()) * 2  # fp16 factors
            total += (d["bbox"].numel() + d["attention_mask"].numel() + d["labels"].numel()) * 8
            if "input_ids" in d:
                total += d["input_ids"].numel() * 8
        return total
