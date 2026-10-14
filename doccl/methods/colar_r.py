"""CoLaR-R — CoLaR with label refresh on the stored memory.

Every replay buffer banks labels in the head space of the session that captured the
document, where not-yet-introduced classes are masked to O. Replaying such a document at a
later session teaches "O" for every class learned since, which is why replay sits below
naive on class-incremental CORD. CoLaR's latents live below the frozen boundary, so the
stored document is exactly what the current model would compute on the page: the memory can
be re-annotated without any raw bytes.

refresh: oracle  — unmask the document's true labels for every class now in the head
                   (upper bound on the stale-label cost; needs ``labels_full`` from a CIL
                   remapper built with ``full_head_label_to_id``).
refresh: pseudo  — run the stored latents through the current upper stack after each task
                   and relabel O tokens whose confident prediction is a post-capture class
                   (CPFD's per-class median-entropy rule, applied to the memory instead of
                   to live data; teacher-free, deployable).
refresh: none    — plain CoLaR (control).
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from doccl.methods.colar import CoLaR
from doccl.types import TaskInfo

log = logging.getLogger(__name__)


def oracle_refresh(labels: torch.Tensor, labels_full: torch.Tensor, n_current: int) -> torch.Tensor:
    """Unmask every class whose head index already exists; later classes stay O."""
    known = (labels_full >= 0) & (labels_full < n_current)
    out = torch.where(known, labels_full, torch.zeros_like(labels))
    out[labels == -100] = -100
    return out


def token_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1e-12).log()).sum(-1)


def pseudo_candidates(
    labels: torch.Tensor, probs: torch.Tensor, mask: torch.Tensor, n_capture: int
) -> torch.Tensor:
    """O tokens the current model assigns to a class introduced after this doc was banked."""
    pred = probs.argmax(-1)
    return (labels == 0) & mask.bool() & (pred >= n_capture)


def pseudo_thresholds(
    ents: torch.Tensor, preds: torch.Tensor, n_classes: int, floor: float, quantile: float
) -> torch.Tensor:
    """Per-class entropy threshold = quantile of candidate entropies, clamped to >= floor.
    Classes with no candidates get 0 so nothing is relabelled into them."""
    thr = torch.zeros(n_classes)
    for c in preds.unique().tolist():
        thr[c] = max(ents[preds == c].quantile(quantile).item(), floor)
    return thr


def pseudo_refresh(
    labels: torch.Tensor,
    probs: torch.Tensor,
    mask: torch.Tensor,
    n_capture: int,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    pred = probs.argmax(-1)
    cand = pseudo_candidates(labels, probs, mask, n_capture) & (
        token_entropy(probs) < thresholds[pred]
    )
    return torch.where(cand, pred, labels)


class CoLaRRefresh(CoLaR):
    name = "colar_r"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.refresh = str(config.get("refresh", "oracle"))
        if self.refresh not in ("oracle", "pseudo", "none"):
            raise ValueError(f"colar_r.refresh must be oracle|pseudo|none, got {self.refresh!r}")
        self.entropy_floor = float(config.get("refresh_entropy_floor", 1e-3))
        self.refresh_quantile = float(config.get("refresh_quantile", 0.5))
        self.refresh_log: list[dict] = []

    # ------------------------------------------------------------- capture
    def _capture_task(self, train_loader: DataLoader) -> None:
        n_before = len(self.store)
        super()._capture_task(train_loader)
        for d in self.store[n_before:]:
            if self.refresh == "oracle" and "labels_full" in d:
                d["labels_full"] = d["labels_full"].to(torch.int16)  # <= 61 tags; 2 bytes/token
            else:
                d.pop("labels_full", None)  # pseudo/none never see the true labels

    def memory_bytes(self) -> int:
        return super().memory_bytes() + sum(
            d["labels_full"].numel() * 2 for d in self.store if "labels_full" in d
        )

    # ------------------------------------------------------------- refresh
    def _head_size(self) -> int:
        return len(getattr(self.model, "label_to_id", {}))

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        super().after_task(task, train_loader)
        if self.refresh != "none":
            self._refresh_memory(task.task_id)

    def _refresh_memory(self, task_id: int) -> None:
        n_cur = self._head_size()
        stale = [d for d in self.store if int(d["n_labels"]) < n_cur]
        if not stale:
            return
        changed = (
            self._oracle_pass(stale, n_cur)
            if self.refresh == "oracle"
            else self._pseudo_pass(stale, n_cur)
        )
        for d in stale:
            d["n_labels"] = torch.tensor(n_cur)
        entry = {"task_id": task_id, "docs": len(stale), "tokens_relabelled": changed}
        self.refresh_log.append(entry)
        log.info(
            "colar_r[%s]: refreshed %d docs, %d tokens relabelled",
            self.refresh,
            len(stale),
            changed,
        )

    def _oracle_pass(self, stale: list[dict], n_cur: int) -> int:
        changed = 0
        for d in stale:
            if "labels_full" not in d:
                continue
            new = oracle_refresh(d["labels"], d["labels_full"].long(), n_cur)
            changed += int((new != d["labels"]).sum())
            d["labels"] = new
        return changed

    @torch.no_grad()
    def _memory_probs(self, docs: list[dict]) -> list[torch.Tensor]:
        """Current-model class posteriors for each stored doc, (seq, n_cur) on CPU fp32."""
        was_training = self.model.training
        self.model.eval()
        probs: list[torch.Tensor] = []
        try:
            for i in range(0, len(docs), self.replay_batch_size):
                out = self._replay_forward(self._stack_replay(docs[i : i + self.replay_batch_size]))
                probs.extend(out.logits.float().softmax(-1).cpu())
        finally:
            if was_training:
                self.model.train()
        return probs

    def _pseudo_pass(self, stale: list[dict], n_cur: int) -> int:
        probs = self._memory_probs(stale)
        ents, preds = [], []
        for d, p in zip(stale, probs, strict=True):
            cand = pseudo_candidates(d["labels"], p, d["attention_mask"], int(d["n_labels"]))
            ents.append(token_entropy(p)[cand])
            preds.append(p.argmax(-1)[cand])
        ents_all, preds_all = torch.cat(ents), torch.cat(preds)
        if ents_all.numel() == 0:
            return 0
        thr = pseudo_thresholds(
            ents_all, preds_all, n_cur, self.entropy_floor, self.refresh_quantile
        )
        changed = 0
        for d, p in zip(stale, probs, strict=True):
            new = pseudo_refresh(d["labels"], p, d["attention_mask"], int(d["n_labels"]), thr)
            changed += int((new != d["labels"]).sum())
            d["labels"] = new
        return changed
