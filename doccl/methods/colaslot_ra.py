"""CoLaSlot-RF with acquisition-logit-anchored residual refits."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from doccl.methods.colaslot_rf import CoLaSlotRF

__all__ = ["CoLaSlotRA"]


class CoLaSlotRA(CoLaSlotRF):
    """Recover acquisition-time predictions instead of already-satisfied hard labels."""

    name = "colaslot_ra"

    def _capture_task(self, train_loader: DataLoader) -> None:
        before = len(self.store)
        super()._capture_task(train_loader)
        new_docs = self.store[before:]
        was_training = self.model.training
        reads_enabled = self._slot_reads_enabled
        self._slot_reads_enabled = False
        self.model.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(new_docs), self.replay_batch_size):
                    docs = new_docs[start : start + self.replay_batch_size]
                    logits = self._replay_forward(self._stack_replay(docs)).logits.cpu().half()
                    for doc, teacher_logits in zip(docs, logits, strict=True):
                        doc["teacher_logits"] = teacher_logits
        finally:
            self._slot_reads_enabled = reads_enabled
            if was_training:
                self.model.train()

    def _positive_refit_loss(
        self, replay: dict[str, torch.Tensor], docs: list[dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        logits = self._replay_forward(replay).logits
        teacher = torch.stack([doc["teacher_logits"] for doc in docs]).to(
            device=logits.device, dtype=logits.dtype
        )
        valid = replay["attention_mask"].to(logits.device).bool().unsqueeze(-1)
        error = logits[..., : teacher.shape[-1]] - teacher
        return (error.square() * valid).sum() / (valid.sum().clamp_min(1) * teacher.shape[-1])

    def memory_bytes(self) -> int:
        return super().memory_bytes() + sum(
            doc["teacher_logits"].numel() * doc["teacher_logits"].element_size()
            for doc in self.store
            if "teacher_logits" in doc
        )
