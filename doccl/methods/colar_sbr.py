"""CoLaR-SBR — Session-Blocked Readout on the whole-document latent bank (AGENT IMPL 2026-11-02).

Diagnosis (CIL-CORD, corrected scorer): every CORD session is a price-like class separated
only by position/context; the shared-softmax head receives contradictory *partial*
supervision — replay docs label post-capture entities O, current docs label old entities O —
and every repair that keeps one shared O (sliced CE, double-sliced, label completion) still
moves the absolute O logit on exactly those surface forms, so new sessions decay/oscillate
and the true-label oracle tops out at 61 (joint 93).

Fix: un-share the readout. One block per session ``b = [O_b, tags_b]`` with its own
background logit. A token labelled O in session-t data is *truly* O_t, so every session's
annotation is complete and correct within its own block; a document banked at session s
supervises block s only (other blocks are masked on it); the trunk is trained by all blocks
and kept stable by CoLaR's latent replay. Task-agnostic inference: per block
``e_b = 1 - p_b(O_b)``; the block with the largest ``e_b`` claims the token if
``e_b > sbr_threshold``, else O.

Closest prior art: ICE (Liu & Huang, ACL Findings 2023 — individual per-session classifiers
on a *frozen* PLM) and SpanKL (AAAI 2023 — per-type sigmoid span heads, no replay). SBR is
the blocked readout with a *trained* trunk and whole-document latent replay on documents,
where each session's data is the full population so every block sees true negatives.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from doccl.eval.metrics import compute_eval_metrics
from doccl.methods.colar import CoLaR
from doccl.types import EvalMetrics, TaskInfo

log = logging.getLogger(__name__)

__all__ = ["CoLaRSBR", "block_targets", "aggregate_blocks"]


def block_targets(labels: torch.Tensor, idx: torch.Tensor, n_head: int) -> torch.Tensor:
    """Global head labels -> block-local targets: O -> 0, in-block tag -> 1 + position,
    anything else (other blocks' tags, -100) -> -100."""
    pos = torch.full((n_head,), -1, dtype=torch.long, device=labels.device)
    pos[idx] = torch.arange(len(idx), device=labels.device)
    tgt = torch.full_like(labels, -100)
    tgt[labels == 0] = 0
    valid = labels > 0
    local = pos[labels.clamp(min=0)]
    inblk = valid & (local >= 0)
    tgt[inblk] = local[inblk] + 1
    return tgt


def aggregate_blocks(
    block_logits: list[torch.Tensor],
    block_idx: list[torch.Tensor],
    threshold: float,
    allowed: set[int] | None,
    n_head: int,
) -> torch.Tensor:
    """Task-agnostic prediction from per-block logits ``[(B, L, 1 + n_b)]`` -> (B, L) head ids.

    The block with the largest entity mass ``1 - p(O_b)`` claims the token when that mass
    exceeds ``threshold``; otherwise O. ``allowed`` (the corrected-scorer restriction) maps
    any out-of-session prediction to O *after* the full-model decision."""
    ref = block_logits[0]
    best_e = torch.zeros(ref.shape[:2], device=ref.device)
    best_tag = torch.zeros(ref.shape[:2], dtype=torch.long, device=ref.device)
    for bl, idx in zip(block_logits, block_idx, strict=True):
        p = bl.float().softmax(-1)
        e = 1.0 - p[..., 0]
        tag = idx[p[..., 1:].argmax(-1)]
        take = e > best_e
        best_e = torch.where(take, e, best_e)
        best_tag = torch.where(take, tag, best_tag)
    preds = torch.where(best_e > threshold, best_tag, torch.zeros_like(best_tag))
    if allowed is not None:
        keep = torch.zeros(n_head, dtype=torch.bool, device=preds.device)
        keep[sorted(i for i in allowed if i < n_head)] = True
        preds = torch.where(keep[preds], preds, torch.zeros_like(preds))
    return preds


class CoLaRSBR(CoLaR):
    """CoLaR with a session-blocked readout (per-session background logit)."""

    name = "colar_sbr"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.threshold = float(config.get("sbr_threshold", 0.5))
        # Attached to the wrapper so EarlyStopper snapshots / trainable_parameters see it.
        self.model.sbr_o = nn.ModuleList().to(self.device)
        self.model.restrict_in_method = True  # train.py sets model._allowed instead of a hook
        self.model._allowed = None
        self.blocks: list[torch.Tensor] = []
        self._cur_block = -1
        self._feat: torch.Tensor | None = None
        self._hook = None

    # ------------------------------------------------------------------ plumbing
    def _hook_classifier(self) -> None:
        """Capture the token features that feed the (re-created on expansion) head."""
        head = self.model.model.classifier
        if getattr(self, "_hooked", None) is head:
            return
        if self._hook is not None:
            self._hook.remove()

        def grab(_mod, inp, _out):
            self._feat = inp[0]

        self._hook = head.register_forward_hook(grab)
        self._hooked = head

    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        super().before_task(task, train_loader)
        self._hook_classifier()
        l2i = self.model.label_to_id
        idx = [l2i[lb] for lb in task.label_set if lb in l2i and lb != "O"]
        self.blocks.append(torch.tensor(idx, dtype=torch.long, device=self.device))
        hidden = getattr(self.model, "hidden_size", None) or self.model.model.config.hidden_size
        self.model.sbr_o.append(nn.Linear(hidden, 1).to(self.device))
        self._cur_block = len(self.blocks) - 1
        log.info("colar_sbr: block %d = %d tags (+ its own O)", self._cur_block, len(idx))

    def _capture_task(self, train_loader: DataLoader) -> None:
        n_before = len(self.store)
        super()._capture_task(train_loader)
        for d in self.store[n_before:]:
            d["block"] = torch.tensor(self._cur_block)

    def _stack_replay(self, docs):
        replay = super()._stack_replay(docs)
        replay["block"] = torch.stack([d["block"] for d in docs])
        return replay

    def _block_logits(self, logits: torch.Tensor, feat: torch.Tensor, b: int) -> torch.Tensor:
        o = self.model.sbr_o[b](feat.to(self.model.sbr_o[b].weight.dtype))
        return torch.cat([o.to(logits.dtype), logits[..., self.blocks[b]]], dim=-1)

    def _block_ce(self, logits, feat, labels, b: int) -> torch.Tensor:
        bl = self._block_logits(logits, feat, b)
        tgt = block_targets(labels, self.blocks[b], logits.shape[-1])
        return F.cross_entropy(
            bl.reshape(-1, bl.shape[-1]).float(), tgt.reshape(-1), ignore_index=-100
        )

    # ------------------------------------------------------------------ losses
    def _current_ce(self, cur_out, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._block_ce(cur_out.logits, self._feat, batch["labels"], self._cur_block)

    def _replay_forward(self, replay: dict[str, torch.Tensor]):
        out = super()._replay_forward(replay)
        logits, feat = out.logits, self._feat
        labels = replay["labels"].to(self.device)
        losses = [
            self._block_ce(logits[i : i + 1], feat[i : i + 1], labels[i : i + 1], int(b))
            for i, b in enumerate(replay["block"].tolist())
        ]
        out.loss = torch.stack(losses).mean()
        return out

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self._hook_classifier()  # expand_classifier re-creates the head between tasks
        out = self.model(**{k: v for k, v in batch.items() if k != "labels"})
        logits, feat = out.logits, self._feat
        if not self.blocks:  # zero-shot reading before the first block exists
            return torch.zeros(logits.shape[:2], dtype=torch.long, device=logits.device)
        bls = [self._block_logits(logits, feat, b) for b in range(len(self.blocks))]
        return aggregate_blocks(
            bls,
            self.blocks,
            self.threshold,
            getattr(self.model, "_allowed", None),
            logits.shape[-1],
        )

    def _score_loader(self, loader: DataLoader) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()
        id_to_label = getattr(self.model, "id_to_label", None) or {
            i: str(i) for i in range(self.model.model.config.num_labels)
        }
        all_preds: list[int] = []
        all_labels: list[int] = []
        try:
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                preds = self._predict(batch)
                labels = batch["labels"]
                mask = labels != -100
                all_preds.extend(preds[mask].cpu().tolist())
                all_labels.extend(labels[mask].cpu().tolist())
        finally:
            if was_training:
                self.model.train()
        m = compute_eval_metrics(
            all_preds, all_labels, id_to_label, getattr(self.model, "task_type", "token")
        )
        m["n"] = len(all_labels)
        return m

    def current_task_val_f1(self, val_loader: DataLoader) -> float:
        m = self._score_loader(val_loader)
        log.info("colar_sbr val p=%.2f r=%.2f f1=%.2f", m["precision"], m["recall"], m["f1"])
        return m["f1"]

    def evaluate(self, eval_loaders: dict[int, DataLoader]) -> dict[int, EvalMetrics]:
        results: dict[int, EvalMetrics] = {}
        for tid, loader in eval_loaders.items():
            m = self._score_loader(loader)
            results[tid] = EvalMetrics(
                task_id=tid,
                f1=m["f1"],
                precision=m["precision"],
                recall=m["recall"],
                n_samples=int(m["n"]),
            )
        return results

    def memory_bytes(self) -> int:
        return super().memory_bytes() + len(self.store)  # 1 byte/doc block id
