"""ER-SBR: raw-document replay x Session-Blocked Readout (AGENT IMPL 2026-11-11, D12 E1 control).

The senior review's "ER + SBR" cell. Identical to CoLaR-SBR (same blocked readout and
supervision plan, same 50-docs-per-session draw from the shuffled loader, same replay batch,
same loss) except for the memory:

- stored: the raw model inputs of each banked document (token ids, boxes, page pixels as fed
  to the model, attention mask, session-masked labels) plus its session id; no activations,
  no compression;
- frozen: nothing; embeddings and every encoder layer stay plastic for all sessions;
- replay: a normal forward of the stored documents through the whole network (gradient
  checkpointing stays on, unlike latent injection).
"""

from __future__ import annotations

import contextlib
import logging

import torch
from torch.utils.data import DataLoader

from doccl.methods.colar_sbr import CoLaRSBR

log = logging.getLogger(__name__)

__all__ = ["ERSBR"]

_RAW_KEYS = ("input_ids", "bbox", "pixel_values", "attention_mask", "labels")


class ERSBR(CoLaRSBR):
    """CoLaR-SBR with a raw-document memory and no frozen trunk."""

    name = "er_sbr"

    def _apply_freeze_map(self) -> None:
        log.info("er_sbr: no freeze map, whole network plastic")

    def _capture_task(self, train_loader: DataLoader) -> None:
        n = 0
        for batch in train_loader:
            for i in range(batch["labels"].shape[0]):
                if n >= self.docs_per_task:
                    break
                doc = {k: batch[k][i].clone() for k in _RAW_KEYS if k in batch}
                doc["session"] = torch.tensor(self._cur_session)
                self.store.append(doc)
                n += 1
            if n >= self.docs_per_task:
                break
        log.info("er_sbr: banked %d raw docs (store=%d)", n, len(self.store))

    def _stack_replay(self, docs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {k: torch.stack([d[k] for d in docs]) for k in docs[0]}

    def no_checkpointing(self):
        return contextlib.nullcontext()  # plain forward: checkpointing is safe and saves VRAM

    def _replay_forward_inner(self, replay: dict[str, torch.Tensor]):
        return self.model(**{k: v.to(self.device) for k, v in replay.items() if k in _RAW_KEYS})

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for d in self.store for t in d.values())
