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
refresh: sliced  — no relabelling; replay CE over each document's capture-time classes
                   only, so a stale O never pushes a post-capture class down.
refresh: sliced_pseudo — sliced loss during the task, pseudo refresh after it. Plain
                   pseudo relabels ~0 tokens because replaying stale O labels teaches the
                   model "O" on exactly the memory tokens it is later asked to relabel;
                   slicing removes that self-defeating signal so candidates survive.
refresh: oracle_mask — mask-only counterpart of the oracle (R18 E2b): every stored token the
                   oracle would relabel (an O whose true class is in the head now) is set to -100,
                   so it leaves the replay CE as neither "O" nor a positive. Removes the false
                   negatives without adding a label; needs ``labels_full``.
refresh: none    — plain CoLaR (control).

When the oracle refresh runs (AGENT IMPL 2026-11-12, D17). Capture and refresh both sit in
``after_task``; a document banked at session s holds that session's CIL-masked labels.
``refresh_before_task: false`` (default, "refreshed at the boundary", the colar_r_oracle runs):
  the refresh runs after each session's capture, over documents whose capture-time head is
  smaller than the current one. While session t trains, documents banked at sessions <= t-2
  carry the true labels of every class of sessions 0..t-1 (session-t classes are O); the
  documents banked at session t-1 were never refreshed and carry ONLY session t-1's classes
  (everything else O, exactly like current data).
``refresh_before_task: true`` (oracle / oracle_mask, "refreshed before the session", colar_r_oracle_pre):
  additionally, ``before_task`` re-applies the oracle to EVERY stored document using the head
  as expanded for session t. While session t trains, every stored document (sessions 0..t-1)
  carries the true labels of all classes of sessions 0..t; only classes not yet in the head are
  O. What still differs from a perfect-label memory: the current session's own documents still
  label old entities O (that conflict is the point of the control); the memory holds only
  ``docs_per_task`` documents per past session and none of session t; stored latents are the
  rank-``rank_r`` per-document SVD reconstruction, not the exact layer-k activations; labels
  exist on first subwords only (as everywhere).
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812
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


def mask_refresh(labels: torch.Tensor, labels_full: torch.Tensor, n_current: int) -> torch.Tensor:
    """Ignore (-100) exactly the tokens ``oracle_refresh`` would change: a stored O whose true
    class is in the head now. That is not only classes added after capture (n_capture <=
    class < n_current, the R18 spec's range) but also older classes, because a CIL session's
    pages carry only that session's tags and label every earlier entity O."""
    out = labels.clone()
    out[(oracle_refresh(labels, labels_full, n_current) != labels) & (labels != -100)] = -100
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


def sliced_replay_loss(
    logits: torch.Tensor, labels: torch.Tensor, n_capture: torch.Tensor
) -> torch.Tensor:
    """Replay CE over each document's capture-time label space only.

    Classes introduced after a document was banked are removed from its softmax, so a stale
    O label cannot push them down (label-free analogue of the refresh; cf. SS-IL / ER-ACE).
    logits (B, L, C), labels (B, L) with -100 ignored, n_capture (B,) head size at capture.
    """
    n_cls = logits.shape[-1]
    allowed = torch.arange(n_cls, device=logits.device)[None, None, :] < n_capture[:, None, None]
    masked = logits.masked_fill(~allowed, float("-inf"))
    return F.cross_entropy(masked.reshape(-1, n_cls), labels.reshape(-1), ignore_index=-100)


class CoLaRRefresh(CoLaR):
    name = "colar_r"
    MODES = ("oracle", "oracle_mask", "pseudo", "sliced", "sliced_pseudo", "none")
    _TRUE_LABEL_MODES = ("oracle", "oracle_mask")

    def __init__(self, model, config):
        super().__init__(model, config)
        self.refresh = str(config.get("refresh", "oracle"))
        if self.refresh not in self.MODES:
            raise ValueError(f"colar_r.refresh must be one of {self.MODES}, got {self.refresh!r}")
        self.entropy_floor = float(config.get("refresh_entropy_floor", 1e-3))
        self.refresh_quantile = float(config.get("refresh_quantile", 0.5))
        self.refresh_log: list[dict] = []
        self.refresh_before_task = bool(config.get("refresh_before_task", False))
        if self.refresh_before_task and self.refresh not in self._TRUE_LABEL_MODES:
            raise ValueError("colar_r.refresh_before_task needs refresh: oracle or oracle_mask")

    # ------------------------------------------------------------- capture
    def _capture_task(self, train_loader: DataLoader) -> None:
        n_before = len(self.store)
        super()._capture_task(train_loader)
        for d in self.store[n_before:]:
            if self.refresh in self._TRUE_LABEL_MODES and "labels_full" in d:
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

    def _stack_replay(self, docs: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        replay = super()._stack_replay(docs)
        replay["n_labels"] = torch.stack([d["n_labels"] for d in docs])
        return replay

    def _replay_forward(self, replay: dict[str, torch.Tensor]):
        out = super()._replay_forward(replay)
        # AGENT FIX (2026-11-02): `replay_sliced: true` applies the sliced replay CE in ANY
        # refresh mode (oracle/pseudo docs are still stale between refreshes; without it the
        # oracle ceiling collapsed its new session to 1-12 val-F1 on CIL-CORD).
        if self.refresh in ("sliced", "sliced_pseudo") or self.config.get("replay_sliced", False):
            out.loss = sliced_replay_loss(
                out.logits,
                replay["labels"].to(self.device),
                replay["n_labels"].to(self.device),
            )
        return out

    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        super().before_task(task, train_loader)
        if self.refresh_before_task:
            self._refresh_memory(task.task_id, all_docs=True)

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        super().after_task(task, train_loader)
        if self.refresh in ("oracle", "oracle_mask", "pseudo", "sliced_pseudo"):
            self._refresh_memory(task.task_id)

    def _refresh_memory(self, task_id: int, all_docs: bool = False) -> None:
        n_cur = self._head_size()
        # all_docs: also the newest bank, whose capture-time head equals the current one but whose
        # labels are still CIL-masked (oracle only; idempotent on already-true labels)
        stale = [d for d in self.store if all_docs or int(d["n_labels"]) < n_cur]
        if not stale:
            return
        entry = {"task_id": task_id, "docs": len(stale)}
        if self.refresh == "oracle_mask":
            entry["tokens_masked"] = self._mask_pass(stale, n_cur)
            for d in stale:
                d["n_labels"] = torch.tensor(n_cur)
            self.refresh_log.append(entry)
            log.info(
                "colar_r[oracle_mask]: refreshed %d docs, %d tokens masked",
                len(stale),
                entry["tokens_masked"],
            )
            return
        if self.refresh == "oracle":
            entry["tokens_relabelled"] = self._oracle_pass(stale, n_cur)
        else:
            entry["tokens_relabelled"], entry["candidates"] = self._pseudo_pass(stale, n_cur)
        for d in stale:
            d["n_labels"] = torch.tensor(n_cur)
        self.refresh_log.append(entry)
        log.info(
            "colar_r[%s]: refreshed %d docs, %d tokens relabelled (%s candidates)",
            self.refresh,
            len(stale),
            entry["tokens_relabelled"],
            entry.get("candidates", "n/a"),
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

    def _mask_pass(self, stale: list[dict], n_cur: int) -> int:
        """Idempotent: masked tokens stay -100 and are never counted twice."""
        masked = 0
        for d in stale:
            if "labels_full" not in d:
                continue
            new = mask_refresh(d["labels"], d["labels_full"].long(), n_cur)
            masked += int(((new == -100) & (d["labels"] != -100)).sum())
            d["labels"] = new
        return masked

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

    def _pseudo_pass(self, stale: list[dict], n_cur: int) -> tuple[int, int]:
        """Returns (tokens relabelled, candidate tokens = stale-O tokens predicted post-capture)."""
        probs = self._memory_probs(stale)
        ents, preds = [], []
        for d, p in zip(stale, probs, strict=True):
            cand = pseudo_candidates(d["labels"], p, d["attention_mask"], int(d["n_labels"]))
            ents.append(token_entropy(p)[cand])
            preds.append(p.argmax(-1)[cand])
        ents_all, preds_all = torch.cat(ents), torch.cat(preds)
        n_cand = int(ents_all.numel())
        if n_cand == 0:
            return 0, 0
        thr = pseudo_thresholds(
            ents_all, preds_all, n_cur, self.entropy_floor, self.refresh_quantile
        )
        changed = 0
        for d, p in zip(stale, probs, strict=True):
            new = pseudo_refresh(d["labels"], p, d["attention_mask"], int(d["n_labels"]), thr)
            changed += int((new != d["labels"]).sum())
            d["labels"] = new
        return changed, n_cand
