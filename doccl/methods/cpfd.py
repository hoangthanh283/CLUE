"""CPFD — Confidence-based Pseudo-labeling + Pooled Feature Distillation for CIL NER.

Reference: Zhang et al., "Continual Named Entity Recognition without Catastrophic Forgetting",
EMNLP 2023 Main, ACL Anthology 2023.emnlp-main.509.
Code: github.com/BladeDancer957/CPFD.

CPFD counters the "O-label boundary" problem in class-incremental NER:
old entity tokens are relabeled as 'O' in new-task data, so naive CE destroys
old-type discriminability.  Two mechanisms address this:

**CP (confidence-based pseudo-labeling)**
  The frozen previous-task teacher scores 'O'-labeled tokens.  Per-class entropy
  thresholds are precomputed (one pass of teacher over the new-task training set,
  median entropy per argmax class, clamped to ≥ 0.001 floor).  A token is assigned
  a pseudo old-class label when teacher entropy < per-class threshold; otherwise it
  is masked out (label = -100), so uncertain O-tokens contribute no gradient.

**FD (pooled feature distillation)**
  Two distillation sub-losses:
  a) Logit-KL on O-labeled tokens, student logits sliced to first n_old dims, temp=1.
  b) Three-view attention-map MSE per layer (head-pooled, query-pooled, key-pooled),
     averaged over all transformer layers.  Requires output_attentions=True on both
     student and teacher forwards.

Total objective:
  L = CE(pseudo-relabeled_labels) + distill_loss_coef × distill_weight × (KL + attn_MSE)

where `distill_loss_coef = sqrt((n_old-1) / (n_total-n_old))` adapts naturally:
distillation pressure shrinks as new classes are added (more n_total denominator).

Token-level adaptation notes:
  - O label index = 0 in the CIL-remapped space (same as BIO-NER convention).
  - Attention maps from `output_attentions=True` expose per-layer (B, H, L, L) tensors.
    This adds ~600 MB extra VRAM on LayoutLMv3 (student + teacher, 12 layers); use
    batch_size=1 on the 6 GB local box.
  - `sample_weights` from the reference (old/new token ratio) is simplified to 1.0 in
    this port; `classif_adaptive_factor` (pseudo-label coverage fraction) is kept, since
    it is the paper's key CE-scaling innovation.
  - No exemplar buffer: all anti-forgetting runs on the live new-task data.
"""

from __future__ import annotations

import copy
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo, TrainMetrics

log = logging.getLogger(__name__)

__all__ = ["CPFD"]

_BG_LABEL = 0  # 'O' / background label index in the CIL-remapped space


class CPFD(NaiveFineTune):
    """CPFD — pseudo-labeling + attention distillation CL (see module docstring)."""

    name = "cpfd"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.threshold_floor = float(config.get("threshold", 0.001))
        self.distill_weight = float(config.get("distill_weight", 2.0))
        self.ref_temperature = float(config.get("ref_temperature", 1.0))
        self.adaptive_distill = bool(config.get("adaptive_distill_weight", True))
        self.adaptive_ce = bool(config.get("classif_adaptive_factor", True))
        self.adaptive_ce_min = float(config.get("classif_adaptive_min_factor", 0.0))

        self._teacher: nn.Module | None = None
        self._n_old: int = 0
        self._thresholds: torch.Tensor | None = None  # (n_old,) per-class entropy medians

    @staticmethod
    def _entropy(probs: torch.Tensor) -> torch.Tensor:
        """Per-token Shannon entropy [B, L]."""
        return -(probs * (probs + 1e-8).log()).sum(dim=-1)

    def _forward_with_attentions(
        self, wrapper: nn.Module, batch: dict
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Run inner HF model with output_attentions=True.

        Returns (logits [B,L,C], attentions tuple of (B,H,L,L) per layer).
        """
        inner = wrapper.model  # HF *ForTokenClassification
        safe_keys = {
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "bbox",
            "pixel_values",
            "position_ids",
            "head_mask",
            "inputs_embeds",
        }
        hf_batch = {k: v for k, v in batch.items() if k in safe_keys}
        outputs = inner(**hf_batch, output_attentions=True, return_dict=True)
        attentions = outputs.attentions if outputs.attentions is not None else ()
        return outputs.logits, attentions

    def _attn_mse(
        self,
        s_attns: tuple[torch.Tensor, ...],
        t_attns: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        """Three-view attention MSE averaged over transformer layers."""
        if not s_attns or not t_attns:
            return torch.tensor(0.0, device=self.device)
        total = 0.0
        n_layers = min(len(s_attns), len(t_attns))
        for a_s, a_t in zip(s_attns[:n_layers], t_attns[:n_layers], strict=False):
            # Three pooling views per layer (reference implementation).
            total += F.mse_loss(a_s.mean(1), a_t.mean(1))  # pool heads  [B, L, L]
            total += F.mse_loss(a_s.mean(2), a_t.mean(2))  # pool queries [B, H, L]
            total += F.mse_loss(a_s.mean(3), a_t.mean(3))  # pool keys    [B, H, L]
        return torch.as_tensor(total / n_layers, device=self.device)

    def _adaptive_coef(self, n_old: int, n_total: int) -> float:
        """distill_loss_coefficient = sqrt((n_old-1) / max(1, n_total-n_old))."""
        if not self.adaptive_distill or n_total <= n_old:
            return 1.0
        return math.sqrt(max(0, n_old - 1) / max(1, n_total - n_old))

    @torch.no_grad()
    def _find_median_thresholds(self, train_loader: DataLoader) -> None:
        """One teacher forward pass; per-class entropy medians → self._thresholds."""
        n_old = self._n_old
        buckets: dict[int, list[float]] = {c: [] for c in range(n_old)}
        for batch in train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            labels_2d = batch.get("labels")
            if labels_2d is None:
                continue
            t_logits, _ = self._forward_with_attentions(self._teacher, batch)
            probs = torch.softmax(t_logits, dim=-1)  # (B, L, n_old)
            pseudo = probs.argmax(dim=-1)  # (B, L)
            entr = self._entropy(probs)  # (B, L)
            bg_mask = labels_2d == _BG_LABEL  # O-labeled tokens
            # Collect entropy values grouped by argmax pseudo class.
            flat_entr = entr[bg_mask].cpu().tolist()
            flat_pseudo = pseudo[bg_mask].cpu().tolist()
            for e, c in zip(flat_entr, flat_pseudo, strict=True):
                if c < n_old:
                    buckets[c].append(e)
        # Compute median per class, clamp to threshold_floor.
        thresholds = []
        for c in range(n_old):
            if buckets[c]:
                vals = sorted(buckets[c])
                med = vals[len(vals) // 2]
            else:
                med = self.threshold_floor  # no data → use floor (accept all)
            thresholds.append(max(med, self.threshold_floor))
        self._thresholds = torch.tensor(thresholds, device=self.device)
        log.info("cpfd: per-class entropy thresholds computed for %d old classes", n_old)

    def _apply_pseudo_labels(
        self, labels_2d: torch.Tensor, t_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Relabel O-tokens in-place; return (modified_labels, classif_factor [B,1])."""
        _b, _l = labels_2d.shape  # noqa: N806 (B, L are conventional tensor-dim names)
        labels = labels_2d.clone()
        n_old = self._n_old
        probs = torch.softmax(t_logits, dim=-1)[:, :, :n_old]  # (B, L, n_old)
        pseudo = probs.argmax(dim=-1)  # (B, L)
        entr = self._entropy(probs)  # (B, L)
        bg_mask = labels == _BG_LABEL  # (B, L)
        confident = entr < self._thresholds[pseudo.clamp(0, n_old - 1)]  # (B, L)
        # Confident O-tokens → pseudo old-class label.
        labels[bg_mask & confident] = pseudo[bg_mask & confident]
        # Unconfident O-tokens → masked (excluded from loss).
        labels[bg_mask & ~confident] = -100
        # classif_adaptive_factor: fraction of O-tokens that got pseudo-labels, per sample.
        num = (bg_mask & confident).float().sum(dim=1, keepdim=True)  # (B, 1)
        den = bg_mask.float().sum(dim=1, keepdim=True)
        factor = (num / (den + 1e-6)).clamp(min=self.adaptive_ce_min)  # (B, 1)
        return labels, factor

    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        if task.task_id > 0:
            self._n_old = self.model.classifier.out_features
            # Snapshot teacher (deepcopy before the head is expanded for this task,
            # so it covers exactly the old label set).
            self._teacher = copy.deepcopy(self.model)
            self._teacher.to(self.device).eval()
            for p in self._teacher.parameters():
                p.requires_grad_(False)
            if hasattr(self._teacher, "disable_gradient_checkpointing"):
                self._teacher.disable_gradient_checkpointing()
            # Precompute per-class entropy thresholds.
            self._find_median_thresholds(train_loader)

    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        self.model.train()
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.get("lr", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        stopper = self.make_early_stopper(val_loader)
        is_first_task = task.task_id == 0

        total_loss = 0.0
        n_steps = 0
        for epoch in range(epochs):
            pbar = tqdm(
                train_loader, desc=f"CPFD T{task.task_id} ep{epoch+1}/{epochs}", leave=False
            )
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                labels_2d = batch.get("labels")
                if labels_2d is None:
                    continue

                s_logits, s_attns = self._forward_with_attentions(self.model, batch)
                # (B, L, C_total)

                ce_factor = torch.ones(s_logits.size(0), 1, device=self.device)
                kd_loss = torch.tensor(0.0, device=self.device)
                attn_loss = torch.tensor(0.0, device=self.device)

                if not is_first_task:
                    with torch.no_grad():
                        t_logits, t_attns = self._forward_with_attentions(self._teacher, batch)

                    labels_2d, ce_factor = self._apply_pseudo_labels(labels_2d, t_logits)

                    flat_logits = s_logits.view(-1, s_logits.size(-1))
                    flat_t_logits = t_logits.view(-1, t_logits.size(-1))
                    orig_bg = batch["labels"].view(-1) == _BG_LABEL  # use original O-mask
                    if orig_bg.any():
                        n_old = self._n_old
                        s_log = F.log_softmax(
                            flat_logits[orig_bg, :n_old] / self.ref_temperature, dim=-1
                        )
                        t_soft = F.softmax(flat_t_logits[orig_bg] / self.ref_temperature, dim=-1)
                        kd_loss = F.kl_div(s_log, t_soft, reduction="batchmean")

                    attn_loss = self._attn_mse(s_attns, t_attns)

                # ce_loss_per_token: (B, L) — permute because F.cross_entropy expects (B,C,L)
                ce_per_tok = F.cross_entropy(
                    s_logits.permute(0, 2, 1), labels_2d, ignore_index=-100, reduction="none"
                )  # (B, L)
                valid = labels_2d != -100
                if valid.any():
                    if self.adaptive_ce and not is_first_task:
                        ce_loss = (ce_factor * ce_per_tok)[valid].mean()
                    else:
                        ce_loss = ce_per_tok[valid].mean()
                else:
                    ce_loss = s_logits.sum() * 0.0

                n_total = s_logits.size(-1)
                coef = self._adaptive_coef(self._n_old, n_total) if not is_first_task else 1.0
                distill_loss = coef * self.distill_weight * (kd_loss + attn_loss)
                loss = ce_loss + distill_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], max_grad_norm
                )
                optimizer.step()

                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix({"ce": f"{ce_loss.item():.3f}", "kd": f"{kd_loss.item():.3f}"})

            if self._early_stop_after_epoch(stopper, val_loader, task, epoch):
                break

        stopper.restore_best(self.model)
        return TrainMetrics(
            task_id=task.task_id, loss=total_loss / max(n_steps, 1), n_steps=n_steps
        )

    # after_task: inherited no-op (CPFD has no replay buffer)
