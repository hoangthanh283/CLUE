"""ICE-O (Liu & Huang, Findings of ACL 2023, arXiv 2305.16559) on a document encoder.

AGENT IMPL (2026-11-12, D18). As published (Sec. 4, App. A.3): the encoder is frozen at its
pre-trained weights in EVERY session; session k gets a new linear classifier W_k trained only
on session k's data with p = softmax([delta; o_k]) where delta is a constant "Other" logit
(0); earlier classifiers are never updated; no memory. Inference: argmax over
[delta; o_1; ...; o_t], i.e. a token gets the tag with the largest logit across all sessions'
heads if it exceeds delta, else O.

Deviations (all forced by our stream / protocol): LayoutLMv3-base instead of BERT-large-cased;
token-level BIO tags (a session head scores its B-/I- tags) instead of span/trigger types;
the clean protocol's 10 epochs with val-F1 early stopping (patience 2) instead of 15 epochs /
patience 3; no gradient accumulation (paper: 8 steps); the frozen encoder runs in eval mode
(no dropout; the paper does not say); heads initialised N(0, 0.02) like every other head here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from doccl.methods.colar_sbr import block_targets
from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo, TrainMetrics

__all__ = ["ICEO", "ice_logits", "ice_o_loss"]


def ice_logits(
    feat: torch.Tensor, heads, blocks: list[torch.Tensor], n_head: int, delta: float
) -> torch.Tensor:
    """Head-space logits: column 0 (O) = delta, each session's tags from its own head, columns
    not yet owned by a head -1e4. argmax of this = the ICE-O inference rule."""
    out = feat.new_full((*feat.shape[:-1], n_head), -1e4)
    out[..., 0] = delta
    for head, idx in zip(heads, blocks, strict=True):
        out[..., idx.to(out.device)] = head(feat).to(out.dtype)
    return out


def ice_o_loss(
    logits_k: torch.Tensor, labels: torch.Tensor, idx: torch.Tensor, n_head: int, delta: float
) -> torch.Tensor:
    """CE over softmax([delta; o_k]); O and any tag outside session k -> class 0 (Other)."""
    z = torch.cat([torch.full_like(logits_k[..., :1], delta), logits_k], -1)
    tgt = block_targets(labels, idx, n_head)
    return F.cross_entropy(z.reshape(-1, z.shape[-1]).float(), tgt.reshape(-1), ignore_index=-100)


class ICEO(NaiveFineTune):
    name = "ice_o"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.delta = float(config.get("other_logit", 0.0))
        self.model.freeze_backbone()  # pre-trained weights, every session
        self.model.ice_heads = nn.ModuleList().to(self.device)  # in state_dict -> early stopper
        self.blocks: list[torch.Tensor] = []
        self._feat: torch.Tensor | None = None
        self._hook = None
        self._hooked = None

    def _hook_classifier(self) -> None:
        """Prepended forward hook on the (re-created on expansion) wrapper head: grab its input
        features and replace its output by the ICE-O logits, before any restriction hook."""
        head = self.model.model.classifier
        if self._hooked is head:
            return
        if self._hook is not None:
            self._hook.remove()

        def swap(_mod, inp, out):
            self._feat = inp[0]
            if not self.blocks:
                return out
            return ice_logits(inp[0], self.model.ice_heads, self.blocks, out.shape[-1], self.delta)

        self._hook = head.register_forward_hook(swap, prepend=True)
        self._hooked = head

    def trainable_parameters(self) -> list[nn.Parameter]:
        heads = self.model.ice_heads
        return list(heads[-1].parameters()) if len(heads) else []

    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        super().before_task(task, train_loader)
        self._hook_classifier()
        l2i = self.model.label_to_id
        covered = {int(i) for b in self.blocks for i in b}
        new = [
            l2i[lb] for lb in task.label_set if lb != "O" and lb in l2i and l2i[lb] not in covered
        ]
        if not new:
            return  # a session with no new tags adds no head (not the case on CIL-CORD)
        for h in self.model.ice_heads:  # earlier classifiers are never updated
            h.requires_grad_(False)
        head = nn.Linear(self.model.hidden_size, len(new)).to(self.device)
        nn.init.normal_(head.weight, std=0.02)
        nn.init.zeros_(head.bias)
        self.model.ice_heads.append(head)
        self.blocks.append(torch.tensor(new, dtype=torch.long, device=self.device))

    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        self._hook_classifier()
        self.model.eval()  # frozen encoder; only the new head trains
        params = self.trainable_parameters()
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.get("lr", 1e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        stopper = self.make_early_stopper(val_loader)
        head, idx = self.model.ice_heads[-1], self.blocks[-1]
        total, n_steps = 0.0, 0
        for epoch in range(int(self.config.get("epochs", 10))):
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                with torch.no_grad():
                    self.model(**{k: v for k, v in batch.items() if k != "labels"})
                loss = ice_o_loss(
                    head(self._feat.detach()), batch["labels"], idx, len(self.model.label_to_id),
                    self.delta,
                )  # fmt: skip
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.config.get("max_grad_norm", 1.0))
                optimizer.step()
                total, n_steps = total + float(loss.item()), n_steps + 1
            if self._early_stop_after_epoch(stopper, val_loader, task, epoch):
                break
        stopper.restore_best(self.model)
        return TrainMetrics(task_id=task.task_id, loss=total / max(n_steps, 1), n_steps=n_steps)

    def evaluate(self, eval_loaders):
        self._hook_classifier()  # the zero-shot reading runs before before_task re-hooks
        return super().evaluate(eval_loaders)
