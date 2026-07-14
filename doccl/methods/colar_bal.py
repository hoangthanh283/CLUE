"""CoLaR-Bal — CoLaR + soft-target (dark-knowledge) replay loss.

Diagnostic: CoLaR under-holds the sparse-entity middle task (SROIE final 76.3 vs
joint 83.4) — the whole dil AA gap. Cause is argmax-CE starvation: ~99% of SROIE
tokens are "O", so hard-label replay CE gives near-zero gradient to the 4 real
entity classes, which then drift. (Coverage is fine here — CoLaR banks the real
docs, unlike PLaR whose SROIE hole was a proxy-coverage artifact.)

Fix: replay against the banked soft logit distribution instead of argmax labels —
every masked token carries boundary information, so minority classes are not washed
out. This is PLaR's soft-CE (`ProxyLatentReplay._replay_forward`) applied to CoLaR's
compressed real-doc bank. Novelty (literature gap, 2026-07): nobody reweights the
*reconstructed* gradient of compressed/SVD latent replay for minority classes.

Reuse, not invent: bank per-doc logits SVD-compressed exactly like `hidden`
(same rank_r), reconstruct the same way, and delegate the soft-CE to PLaR's method.
Falls back to CoLaR's hard-CE when soft is off → byte-identical to CoLaR then.
"""

from __future__ import annotations

import logging
import random

import torch
from torch.utils.data import DataLoader

from doccl.methods.colar import CoLaR
from doccl.methods.proxy_latent_replay import ProxyLatentReplay

log = logging.getLogger(__name__)

__all__ = ["CoLaRBal"]


class CoLaRBal(CoLaR):
    """CoLaR with dark-knowledge (soft-target) replay to protect sparse classes."""

    name = "colar_bal"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.soft_labels = bool(config.get("soft_labels", True))
        self.soft_T = float(config.get("soft_T", 2.0))

    def _capture_task(self, train_loader: DataLoader) -> None:
        """Bank+compress hiddens (CoLaR), then also SVD-compress the banked logits."""
        n_before = len(self.store)
        super()._capture_task(train_loader)  # banks us/v hidden factors
        if not self.soft_labels:
            return
        # The parent (LatentReplay._capture_task) does not bank logits; re-run a
        # cheap label-free forward on the same docs to grab them, then compress.
        # ponytail: recompute logits from the reconstructed hidden — one forward per
        # banked doc, off the hot path (after_task only). Avoids threading logits
        # through the inherited capture loop.
        for d in self.store[n_before:]:
            hidden = (d["us"].float() @ d["v"].float()).unsqueeze(0)  # (1, seq, D)
            logits = self._logits_from_hidden(hidden, d)  # (seq, C)
            u, s, vh = torch.linalg.svd(logits.float(), full_matrices=False)
            r = min(self.rank_r, s.shape[0])
            d["lg_us"] = (u[:, :r] * s[:r]).to(torch.float16)  # (seq, r)
            d["lg_v"] = vh[:r].to(torch.float16)  # (r, C)

    def _logits_from_hidden(self, hidden: torch.Tensor, d: dict) -> torch.Tensor:
        """Inject a reconstructed hidden and read the head's logits (no labels).

        ponytail: mirrors the dummy-ids/pixels/_inject construction in
        ProxyLatentReplay._replay_forward rather than factoring it out — ~8 shared
        lines, not worth a shared helper across two research methods. If a third
        caller appears, extract `_injected_forward(hidden, bbox, mask)`.
        """
        was_training = self.model.training
        self.model.eval()
        bbox = d["bbox"].unsqueeze(0).to(self.device)
        attn = d["attention_mask"].unsqueeze(0).to(self.device)
        processor = getattr(self.model, "processor", None)
        pad_id = processor.tokenizer.pad_token_id if processor is not None else 0
        dummy_ids = torch.full((1, bbox.shape[1]), pad_id, dtype=torch.long, device=self.device)
        dummy_pixels = (
            torch.zeros((1, *self._pixel_shape), device=self.device)
            if self._pixel_shape is not None
            else None
        )
        self._inject = hidden.to(self.device)
        try:
            with torch.no_grad():
                out = self.model(
                    input_ids=dummy_ids, bbox=bbox, pixel_values=dummy_pixels, attention_mask=attn
                )
        finally:
            self._inject = None
            if was_training:
                self.model.train()
        return out.logits[0].detach().cpu()  # (seq, C)

    def _sample_replay(self) -> dict[str, torch.Tensor] | None:
        """CoLaR sample + reconstructed soft logits (when banked)."""
        if not self.store:
            return None
        docs = random.sample(self.store, min(self.replay_batch_size, len(self.store)))
        hidden = torch.stack([(d["us"].float() @ d["v"].float()).to(torch.float16) for d in docs])
        out = {
            "hidden": hidden,
            "bbox": torch.stack([d["bbox"] for d in docs]),
            "attention_mask": torch.stack([d["attention_mask"] for d in docs]),
            "labels": torch.stack([d["labels"] for d in docs]),
        }
        if self.soft_labels and "lg_us" in docs[0]:
            out["logits"] = torch.stack(
                [(d["lg_us"].float() @ d["lg_v"].float()).to(torch.float16) for d in docs]
            )
        return out

    def _replay_forward(self, replay: dict[str, torch.Tensor]):
        """Soft mode → PLaR's temperature-scaled soft-CE vs banked logits. Hard mode
        (soft off / no banked logits) → CoLaR's inherited argmax CE.

        We call PLaR's method as a plain function (passing self) rather than binding it
        onto the class: its internal `super()._replay_forward(...)` fallback must resolve
        against CoLaRBal's MRO (→ CoLaR → LatentReplay), which a stolen unbound method
        would break. Its fallback only fires when soft is off / no logits — which we
        route around here — so the soft path never re-enters that branch.
        """
        if self.soft_labels and "logits" in replay:
            return ProxyLatentReplay._replay_forward(self, replay)
        return super()._replay_forward(replay)

    def memory_bytes(self) -> int:
        total = super().memory_bytes()
        for d in self.store:
            if "lg_us" in d:
                total += (d["lg_us"].numel() + d["lg_v"].numel()) * 2  # fp16
        return total
