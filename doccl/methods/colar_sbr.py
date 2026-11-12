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
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from doccl.eval.metrics import compute_eval_metrics
from doccl.methods.colar import CoLaR
from doccl.types import EvalMetrics, TaskInfo

log = logging.getLogger(__name__)

__all__ = ["CoLaRSBR", "block_targets", "aggregate_blocks", "supervision_plan", "task_domain"]


def block_targets(labels: torch.Tensor, idx: torch.Tensor, n_head: int) -> torch.Tensor:
    """Global head labels -> block-local targets for a document whose session annotates every
    tag of the block: O -> 0, in-block tag -> 1 + position, any other annotated tag -> 0 (a known
    entity of another class is a true negative for this block), -100 stays -100."""
    pos = torch.full((n_head,), -1, dtype=torch.long, device=labels.device)
    pos[idx] = torch.arange(len(idx), device=labels.device)
    tgt = torch.full_like(labels, -100)
    valid = labels >= 0
    tgt[valid] = 0
    local = pos[labels.clamp(min=0)]
    inblk = valid & (local >= 0)
    tgt[inblk] = local[inblk] + 1
    return tgt


_DOMAINS = ("wildreceipt", "funsd", "sroie", "cord", "xfund")


def task_domain(task: TaskInfo) -> str:
    """Annotation-schema family of a task (dataset name); '' when unknown."""
    native = (task.metadata or {}).get("native_dataset")
    if native:
        return str(native)
    name = task.task_name.lower()
    return next((d for d in _DOMAINS if d in name), "")


def supervision_plan(
    session_tags: set[int], session_domain: str, blocks: list[set[int]], block_domains: list[str]
) -> list[tuple[int, str]]:
    """Which blocks a document of this session trains, and how.

    ``full``: every tag of the block is annotated in this session -> complete supervision.
    ``neg``: the block belongs to a different dataset schema -> none of its entities can occur,
    every token is a true negative. Otherwise (same schema, block not annotated) the O labels are
    stale for that block, so it is not trained on this document."""
    plan = []
    for b, (tags, dom) in enumerate(zip(blocks, block_domains, strict=True)):
        if tags <= session_tags:
            plan.append((b, "full"))
        elif dom and session_domain and dom != session_domain:
            plan.append((b, "neg"))
    return plan


def block_scores(
    block_logits: list[torch.Tensor], block_idx: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block entity mass ``1 - p(O_b)`` and best in-block head id, each (B, L, n_blocks)."""
    es, tags = [], []
    for bl, idx in zip(block_logits, block_idx, strict=True):
        p = bl.float().softmax(-1)
        es.append(1.0 - p[..., 0])
        tags.append(idx.to(p.device)[p[..., 1:].argmax(-1)])
    return torch.stack(es, -1), torch.stack(tags, -1)


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
    e, tags = block_scores(block_logits, block_idx)
    best_e, best_b = e.max(-1)  # ties -> lowest block, as the old strict-> loop did
    best_tag = tags.gather(-1, best_b.unsqueeze(-1)).squeeze(-1)
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
        self._dump: dict | None = None  # filled only inside evaluate()
        self.eval_dump: dict[int, dict] = {}  # latest per-task block scores; saved by train.py
        self.neg_weight = float(config.get("sbr_neg_weight", 1.0))
        # AGENT IMPL (2026-11-12, D18): sbr_const_o fixes every block's background logit at 0
        # (ICE-O's constant Other) instead of learning it; supervision, memory and routing are
        # unchanged. The sbr_o heads are still created (state_dict shape) but never used.
        self.const_o = bool(config.get("sbr_const_o", False))
        self.blocks: list[torch.Tensor] = []
        self.block_domains: list[str] = []
        self.sessions: list[tuple[set[int], str]] = []  # (annotated head ids, schema domain)
        self._cur_session = -1
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
        domain = task_domain(task)
        covered = set().union(*[set(b.tolist()) for b in self.blocks]) if self.blocks else set()
        new = [i for i in idx if i not in covered]
        if new:  # a session that only re-annotates known tags (DIL, a revisit) adds no block
            self.blocks.append(torch.tensor(new, dtype=torch.long, device=self.device))
            self.block_domains.append(domain)
            hidden = getattr(self.model, "hidden_size", None) or self.model.model.config.hidden_size
            self.model.sbr_o.append(nn.Linear(hidden, 1).to(self.device))
        self.sessions.append((set(idx), domain))
        self._cur_session = len(self.sessions) - 1
        plan = self._plan(self._cur_session)
        log.info(
            "colar_sbr: session %d (%s, %d tags) -> %d blocks; trains %s",
            self._cur_session,
            domain or "?",
            len(idx),
            len(self.blocks),
            plan,
        )

    def _capture_task(self, train_loader: DataLoader) -> None:
        n_before = len(self.store)
        super()._capture_task(train_loader)
        for d in self.store[n_before:]:
            d["session"] = torch.tensor(self._cur_session)

    def _stack_replay(self, docs):
        replay = super()._stack_replay(docs)
        replay["session"] = torch.stack([d["session"] for d in docs])
        return replay

    def _plan(self, session: int) -> list[tuple[int, str]]:
        tags, dom = self.sessions[session]
        return supervision_plan(
            tags, dom, [set(b.tolist()) for b in self.blocks], self.block_domains
        )

    def _block_logits(self, logits: torch.Tensor, feat: torch.Tensor, b: int) -> torch.Tensor:
        if self.const_o:
            o = torch.zeros_like(logits[..., :1])
        else:
            o = self.model.sbr_o[b](feat.to(self.model.sbr_o[b].weight.dtype))
        return torch.cat([o.to(logits.dtype), logits[..., self.blocks[b]]], dim=-1)

    def _block_ce(self, logits, feat, labels, b: int, mode: str = "full") -> torch.Tensor:
        bl = self._block_logits(logits, feat, b)
        if mode == "neg":
            tgt = torch.where(labels >= 0, torch.zeros_like(labels), labels)
        else:
            tgt = block_targets(labels, self.blocks[b], logits.shape[-1])
        return F.cross_entropy(
            bl.reshape(-1, bl.shape[-1]).float(), tgt.reshape(-1), ignore_index=-100
        )

    def _session_loss(self, logits, feat, labels, session: int) -> torch.Tensor:
        terms = [
            self._block_ce(logits, feat, labels, b, mode)
            * (self.neg_weight if mode == "neg" else 1.0)
            for b, mode in self._plan(session)
        ]
        return torch.stack(terms).sum() if terms else logits.sum() * 0.0

    # ------------------------------------------------------------------ losses
    def _current_ce(self, cur_out, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._session_loss(cur_out.logits, self._feat, batch["labels"], self._cur_session)

    def _replay_forward(self, replay: dict[str, torch.Tensor]):
        out = super()._replay_forward(replay)
        logits, feat = out.logits, self._feat
        labels = replay["labels"].to(self.device)
        losses = [
            self._session_loss(logits[i : i + 1], feat[i : i + 1], labels[i : i + 1], int(sess))
            for i, sess in enumerate(replay["session"].tolist())
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
        if self._dump is not None:
            e, tags = block_scores(bls, self.blocks)
            mask = batch["labels"] != -100
            self._dump["e"].append(e[mask].half().cpu())
            self._dump["tag"].append(tags[mask].short().cpu())
            self._dump["gold"].append(batch["labels"][mask].short().cpu())
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
            self._dump = {"e": [], "tag": [], "gold": []}
            try:
                m = self._score_loader(loader)
                allowed = getattr(self.model, "_allowed", None)
                self.eval_dump[tid] = {k: torch.cat(v) for k, v in self._dump.items() if v} | {
                    "allowed": sorted(allowed) if allowed is not None else None,
                    "n_head": len(getattr(self.model, "label_to_id", {})),
                    "threshold": self.threshold,
                    "id_to_label": dict(getattr(self.model, "id_to_label", None) or {}),
                }
            finally:
                self._dump = None
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
