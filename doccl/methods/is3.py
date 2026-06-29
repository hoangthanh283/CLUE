"""IS3 — Incremental Sequence Labeling: A Tale of Two Shifts.

Reference: Chen, Cao, Li, et al., "Incremental Sequence Labeling: A Tale of Two Shifts",
ACL 2024 Findings (arXiv:2402.10447). Code: zzz47zzz/codebase-for-incremental-learning-
with-llm, file models/IS3.py.

IS3 addresses *two* distinct failure modes in class-incremental NER/token-IE:

  E2O ("entity → O"): new-task data relabels old entities as 'O', so naive CE teaches
  the model to predict 'O' on exactly those tokens → catastrophic forgetting.

  O2E ("O → entity"): because CE only rewards new-class logits, the model develops a
  bias toward the latest task; old-entity tokens at inference are over-classified as
  the newest types.

IS3 counters each:
  • E2O  → KD restricted to O-labeled tokens only (the exact tokens old entities were
           relabeled into) over the n_old logit dimensions (temp=1 KL, weight β=2.0).
  • O2E  → (a) Gradient surgery: scale old-entity head rows by grad_weight=0.6 after
           the CE backward while leaving the 'O' row (index 0) and bias untouched;
           (b) Prototype replay: one L2-normalized mean feature vector per seen entity
           class pushed through the live head as a CE term every batch (weight α=0.1).

The 2-stage gradient flow (from the reference implementation):
  1. backward(ce_loss, retain_graph=True) → grad surgery on head.weight
  2. backward(β·kd + α·proto) → KD + prototype grads added on top
  3. optimizer.step()

Key distinctions from LwF: KD is ONLY on O tokens (not all valid tokens); gradient-level
row debiasing on the head; per-class prototype replay — so the three baselines are genuinely
distinct and non-redundant.

Token-level adaptation notes:
  * The reference's classifier_list (one Linear per task) is replaced by our single growing
    head; the grad-surgery loop over classifier_list becomes per-row slices of head.weight
    and head.bias.
  * Prototype storage uses self.model.token_features(batch) → (B,L,D) exactly as the
    reference's obtain_features word-level path.
  * No one-vector-per-sample assumption: IS3 is natively token-level (confirmed in the
    reference code — sentence-level pooling paths are only for generative models).
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo, TrainMetrics

log = logging.getLogger(__name__)

__all__ = ["IS3"]


class IS3(NaiveFineTune):
    """IS3 two-shift CL for token-level sequence labeling (see module docstring)."""

    name = "is3"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.kd_weight = float(config.get("kd_weight", 2.0))  # β
        self.proto_weight = float(config.get("proto_weight", 0.1))  # α
        self.kd_temp_s = float(config.get("kd_temp_student", 1.0))
        self.kd_temp_t = float(config.get("kd_temp_teacher", 1.0))
        self.grad_weight = float(config.get("grad_weight", 0.6))  # old-entity row scale
        self.new_grad_weight = float(
            config.get("new_grad_weight", 1.0)
        )  # new row scale (no-op default)

        self._teacher: nn.Module | None = None
        self._n_old: int = 0  # n seen classes at previous task boundary
        self.prototypes: dict[int, torch.Tensor] = {}  # class_id → (D,) L2-normed mean feature

    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        if task.task_id > 0:
            # Snapshot frozen teacher.
            self._teacher = copy.deepcopy(self.model)
            self._teacher.to(self.device).eval()
            for p in self._teacher.parameters():
                p.requires_grad_(False)
            # Disable gradient checkpointing on the teacher (read-only inference only).
            if hasattr(self._teacher, "disable_gradient_checkpointing"):
                self._teacher.disable_gradient_checkpointing()
            self._n_old = self.model.classifier.out_features

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
        ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        is_first_task = task.task_id == 0

        total_loss = 0.0
        n_steps = 0
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"IS3 T{task.task_id} ep{epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                labels_2d = batch.get("labels")
                if labels_2d is None:
                    continue

                # IS3 uses token_features → classifier rather than the wrapper's full forward,
                # so we can access both features and logits for the KD and prototype losses.
                with self._amp_autocast():
                    feats = self.model.token_features(batch)  # (B, L, D)
                    logits = self.model.classifier(feats)  # (B, L, C_total)

                flat_labels = labels_2d.view(-1)  # (B*L,)
                flat_logits = logits.view(-1, logits.size(-1))  # (B*L, C_total)

                # Token partition (IS3 reference lines 217-218):
                #   ce_mask      → new-entity tokens (not padding, not O)
                #   distill_mask → O-labeled tokens (includes relabeled old entities)
                ce_mask = (flat_labels != -100) & (flat_labels != 0)
                distill_mask = flat_labels == 0

                ce_loss = (
                    ce_loss_fn(flat_logits[ce_mask], flat_labels[ce_mask])
                    if ce_mask.any()
                    else flat_logits.sum() * 0.0
                )
                kd_loss = self._kd_distill(flat_logits, batch, distill_mask)
                pro_loss = self._prototype_loss()
                remain_loss = self.kd_weight * kd_loss + self.proto_weight * pro_loss

                # Stage 1: CE backward → grad surgery on head rows.
                # Stage 2: KD + prototype backward, grads added on top.
                optimizer.zero_grad()
                retain = not is_first_task
                ce_loss.backward(retain_graph=retain)
                if not is_first_task:
                    self._apply_grad_surgery()
                    remain_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], max_grad_norm
                )
                optimizer.step()

                step_loss = float(ce_loss.item()) + float(
                    remain_loss.item() if not is_first_task else 0.0
                )
                total_loss += step_loss
                n_steps += 1
                pbar.set_postfix({"ce": f"{ce_loss.item():.3f}"})

            if self._early_stop_after_epoch(stopper, val_loader, task, epoch):
                break

        stopper.restore_best(self.model)
        return TrainMetrics(
            task_id=task.task_id, loss=total_loss / max(n_steps, 1), n_steps=n_steps
        )

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Build per-class mean L2-normalized prototype features."""
        self.model.eval()
        accum: dict[int, list[torch.Tensor]] = {}
        n_classes = self.model.classifier.out_features
        with torch.no_grad():
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                labels_2d = batch.get("labels")
                if labels_2d is None:
                    continue
                feats = self.model.token_features(batch)  # (B, L, D)
                feats_norm = F.normalize(feats, dim=-1)
                flat_feats = feats_norm.view(-1, feats.size(-1))
                flat_labels = labels_2d.view(-1)
                for cls in range(1, n_classes):  # skip O (=0)
                    mask = flat_labels == cls
                    if mask.any():
                        accum.setdefault(cls, []).append(flat_feats[mask].mean(0).cpu())
        for cls, vecs in accum.items():
            self.prototypes[cls] = torch.stack(vecs).mean(0)  # (D,) on CPU
        log.info(
            "is3: built prototypes for %d classes after task %d", len(self.prototypes), task.task_id
        )
        self.model.train()

    def _kd_distill(
        self,
        flat_logits: torch.Tensor,
        batch: dict,
        distill_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self._teacher is None or not distill_mask.any():
            return flat_logits.sum() * 0.0
        with torch.no_grad():
            t_feats = self._teacher.token_features(batch)
            t_logits = self._teacher.classifier(t_feats)  # (B, L, n_old)
            t_logits_flat = t_logits.view(-1, t_logits.size(-1))

        n_old = t_logits_flat.size(-1)
        s_log = F.log_softmax(flat_logits[distill_mask, :n_old] / self.kd_temp_s, dim=-1)
        t_soft = F.softmax(t_logits_flat[distill_mask] / self.kd_temp_t, dim=-1)
        return F.kl_div(s_log, t_soft, reduction="batchmean")

    def _prototype_loss(self) -> torch.Tensor:
        if not self.prototypes:
            return torch.tensor(0.0, device=self.device, requires_grad=False)
        cls_ids = sorted(self.prototypes)
        pro_feats = torch.stack([self.prototypes[c] for c in cls_ids]).to(self.device)  # (n, D)
        pro_labels = torch.tensor(cls_ids, device=self.device)
        pro_logits = self.model.classifier(pro_feats)  # (n, C_total)
        return F.cross_entropy(pro_logits, pro_labels)

    def _apply_grad_surgery(self) -> None:
        """Scale old-entity head rows by grad_weight (protect O row=0 and new rows)."""
        head = self.model.classifier
        if head.weight.grad is None:
            return
        n_old = self._n_old
        n_total = head.weight.size(0)
        # Rows 1..n_old-1 = old entity types → scale down.
        if n_old > 1:
            head.weight.grad[1:n_old] *= self.grad_weight
            if head.bias is not None and head.bias.grad is not None:
                head.bias.grad[1:n_old] *= self.grad_weight
        # Row 0 = 'O' → left untouched (IS3 reference: i==0 path only scales rows [1:]).
        # Rows n_old..n_total-1 = new entity types → scale by new_grad_weight.
        if n_old < n_total:
            head.weight.grad[n_old:] *= self.new_grad_weight
            if head.bias is not None and head.bias.grad is not None:
                head.bias.grad[n_old:] *= self.new_grad_weight
