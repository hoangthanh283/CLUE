"""HGT — Head-localized Gradient-subspace Transfer (see the design spec
docs/superpowers/specs/2026-06-26-head-gradient-subspace-transfer-design.md).

After each task, store the top-k eigenvectors of that task's head-input (token feature)
second moment. During later tasks, a gradient hook on the head rewrites its gradient to
``orthogonal-to-old-subspaces + transfer_alpha * in-old-subspace`` — alpha=0 protects old
tasks (GPM-style, BWT≈0 control), alpha>0 lets new-task gradients flow into old subspaces
(the positive-transfer hypothesis). Backbone frozen or trainable per ``backbone_trainable``.
Standard-forward method: inference is the inherited NaiveFineTune.evaluate.
"""

from __future__ import annotations

import logging

import torch

from doccl.methods.grad_subspace import feature_moment, steer_gradient, top_eigvecs
from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo

log = logging.getLogger(__name__)

__all__ = ["HGT"]


class HGT(NaiveFineTune):
    name = "hgt"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.backbone_trainable = bool(config.get("backbone_trainable", False))
        self.transfer_alpha = float(config.get("transfer_alpha", 1.0))
        self.subspace_k = int(config.get("subspace_k", 32))
        self.subspace_n_batches = int(config.get("subspace_n_batches", 16))
        self.hidden_dim = model.hidden_size
        if not self.backbone_trainable:
            self.model.freeze_backbone()
            for p in self.model.model.classifier.parameters():
                p.requires_grad = True
        self._task_subspaces: list[torch.Tensor] = []  # each (d, k) orthonormal cols
        self._steer_enabled = False
        # One persistent hook on the head weight; it consults _steer_enabled / subspaces.
        self.model.model.classifier.weight.register_hook(self._steer_head_grad)

    # ─── internal helpers ────────────────────────────────────────────────────
    def _head_device(self) -> torch.device:
        """Return device of the classifier head weight, or CPU if no model (tests)."""
        model = getattr(self, "model", None)
        if model is not None:
            return self.model.model.classifier.weight.device
        return torch.device("cpu")

    # ─── subspace bookkeeping ────────────────────────────────────────────────────
    def _stacked_basis(self) -> torch.Tensor:
        """Concatenate stored per-task subspaces, re-orthonormalise -> (d, m)."""
        if not self._task_subspaces:
            return torch.zeros(self.hidden_dim, 0)
        dev = self._head_device()
        stacked = torch.cat([u.to(dev) for u in self._task_subspaces], dim=1)
        q, _ = torch.linalg.qr(stacked)
        return q

    def _steer_head_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Head-weight grad hook: steer against stored old subspaces when enabled."""
        if not self._steer_enabled or not self._task_subspaces:
            return grad
        basis = self._stacked_basis().to(grad.dtype)
        return steer_gradient(grad, basis, self.transfer_alpha)

    # ─── lifecycle ───────────────────────────────────────────────────────────────
    def before_task(self, task: TaskInfo, train_loader) -> None:
        # Steering only applies from task 1 on (no old subspaces before then).
        self._steer_enabled = task.task_id > 0 and len(self._task_subspaces) > 0

    def after_task(self, task: TaskInfo, train_loader) -> None:
        self._accumulate_subspace(train_loader)
        log.info(
            "hgt: task %d done; stored %d head subspaces (k=%d, alpha=%.2f, bb_trainable=%s)",
            task.task_id,
            len(self._task_subspaces),
            self.subspace_k,
            self.transfer_alpha,
            self.backbone_trainable,
        )

    @torch.no_grad()
    def _accumulate_subspace(self, loader) -> None:
        """Top-k eigenvectors of this task's scored-token feature second moment."""
        self.model.eval()
        moment = torch.zeros(self.hidden_dim, self.hidden_dim, device=self.device)
        for bi, batch in enumerate(loader):
            if bi >= self.subspace_n_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            feats = self.model.token_features(batch)  # (B, L, d)
            labels = batch["labels"][:, : feats.shape[1]]
            f = feats[labels != -100]  # (n_tok, d)
            if f.numel():
                moment += feature_moment(f)
        self._task_subspaces.append(top_eigvecs(moment.cpu(), self.subspace_k))
