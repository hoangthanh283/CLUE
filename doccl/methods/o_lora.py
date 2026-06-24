"""Orthogonal LoRA (O-LoRA).

Reference: Wang et al., "Orthogonal Subspace Learning for Language Model
Continual Learning", EMNLP Findings 2023, arXiv:2310.14152.

Per task t, learn LoRA matrices (A_t, B_t) subject to constraint:
    A_t^T · A_{<t} ≈ 0
i.e., new task's LoRA learns in subspace orthogonal to previous tasks.

This eliminates interference between tasks at the cost of capacity:
rank exhaustion limits the number of tasks supported.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo, TrainMetrics


class OLoRA(NaiveFineTune):
    """O-LoRA — orthogonal subspace LoRA for CL."""

    name = "o_lora"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.rank = config.get("lora_rank", 8)
        self.alpha = config.get("lora_alpha", 16)
        self.dropout = config.get("lora_dropout", 0.1)
        self.lambda_ortho = config.get("lambda_ortho", 0.5)

        # Identify target modules for LoRA injection
        # LayoutLMv3 attention Q/K/V projections
        target_modules = config.get(
            "target_modules",
            ["query", "key", "value"],  # matches layoutlmv3 attention naming
        )

        peft_config = LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=target_modules,
            bias="none",
            task_type="TOKEN_CLS",
        )
        # Wrap the inner HF model
        self.model.model = get_peft_model(self.model.model, peft_config)

        # Storage for past tasks' LoRA A matrices (for orthogonality constraint)
        self.state.custom["past_A_matrices"] = []  # list[dict[layer_name → tensor]]

    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Re-sync PEFT's saved classifier copy after a class-IL head expansion.

        PEFT auto-registers the classifier in ``modules_to_save``, replacing it with a
        ``ModulesToSaveWrapper`` whose ``modules_to_save[active_adapter]`` copy is the
        head the forward actually uses. The wrapper's ``expand_classifier`` widens the
        ``ModulesToSaveWrapper`` reference it sees, but leaves that internal copy (and
        the ``original_module``) at the old, narrower width---so after a CIL boundary a
        new-class label index overflows the stale head and the CUDA cross-entropy kernel
        asserts ``t < n_classes``. We re-point both internal copies at the freshly
        expanded Linear so PEFT's forward uses the correct width. No-op when the
        classifier is a plain Linear (non-PEFT / non-growing head).
        """
        super().before_task(task, train_loader)
        base = getattr(self.model.model, "base_model", None)
        inner = getattr(base, "model", None) if base is not None else None
        wrapper = getattr(inner, "classifier", None) if inner is not None else None
        saved = getattr(wrapper, "modules_to_save", None)
        if saved is None:
            return  # not a ModulesToSaveWrapper → nothing to sync
        # The freshly expanded Linear is what the wrapper now exposes as .classifier.
        new_head = self.model.model.classifier
        if not isinstance(new_head, nn.Linear):
            new_head = getattr(wrapper, "original_module", new_head)
        new_n = new_head.out_features
        for adapter in list(saved.keys()):
            if saved[adapter].out_features != new_n:
                saved[adapter] = new_head
        if getattr(wrapper, "original_module", None) is not None and (
            wrapper.original_module.out_features != new_n
        ):
            wrapper.original_module = new_head
        # The wrapper set num_labels on the OUTER PeftModel, but the inner HF model
        # computes the CE loss with ITS cached ``self.num_labels`` (and config). Sync
        # both on the inner model, or the loss reshape uses the stale (narrower) width
        # and raises ``shape '[-1, old_n]' is invalid``.
        if inner is not None:
            inner.num_labels = new_n
            if getattr(inner, "config", None) is not None:
                inner.config.num_labels = new_n

    def _ortho_loss(self) -> torch.Tensor:
        """O-LoRA subspace-orthogonality penalty: ``sum ||A_t A_past^T||_F^2``.

        Wang et al. (2023) constrain successive LoRA adapters to occupy MUTUALLY
        ORTHOGONAL r-dimensional subspaces. With ``A`` of shape ``(r, in_features)``
        (PEFT's lora_A weight), the row space is the relevant subspace, so the cross-
        Gram is ``A_curr @ A_past.T`` of shape ``(r, r)`` — it is zero exactly when the
        two row subspaces are orthogonal. (The previous code computed ``A_curr.T @
        A_past`` of shape ``(in_features, in_features)``, which measures input-feature
        co-activation, NOT subspace orthogonality, and does not vanish for orthogonal
        adapters — a real bug that disabled the O-LoRA constraint.)
        """
        if not self.state.custom["past_A_matrices"]:
            return torch.zeros((), device=self.device)

        penalty = torch.zeros((), device=self.device)
        # Get current task's LoRA A matrices (active adapter)
        current_A = self._get_current_A_matrices()

        for past_A_dict in self.state.custom["past_A_matrices"]:
            for layer_name, A_curr in current_A.items():
                if layer_name in past_A_dict:
                    A_past = past_A_dict[layer_name].to(self.device)
                    # (r, in) @ (in, r) -> (r, r); zero iff the row subspaces are orthogonal.
                    inner = A_curr @ A_past.T  # (r, r)
                    penalty = penalty + (inner**2).sum()
        return penalty

    def _get_current_A_matrices(self) -> dict[str, torch.Tensor]:
        """Extract current task's LoRA A matrices by name."""
        out = {}
        for name, module in self.model.model.named_modules():
            # PEFT names LoRA matrices as `<target>.lora_A.<adapter_name>`
            if hasattr(module, "lora_A") and hasattr(module.lora_A, "default"):
                # LoRA A is a Linear: weight shape (r, in_features)
                a = module.lora_A.default.weight  # (r, in_features)
                out[name] = a
        return out

    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        self.model.train()
        # Only LoRA parameters are trainable (rest frozen by PEFT)
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.get("lr", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        stopper = self.make_early_stopper(val_loader)

        total_loss = 0.0
        n_steps = 0
        for epoch in range(epochs):
            pbar = tqdm(
                train_loader, desc=f"OLoRA T{task.task_id} ep{epoch+1}/{epochs}", leave=False
            )
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                optimizer.zero_grad()
                outputs = self.model(**batch)
                ce_loss = outputs.loss
                ortho_loss = self.lambda_ortho * self._ortho_loss()
                loss = ce_loss + ortho_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], max_grad_norm
                )
                optimizer.step()
                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix(
                    {"ce": f"{ce_loss.item():.3f}", "ortho": f"{ortho_loss.item():.4f}"}
                )
            if self._early_stop_after_epoch(stopper, val_loader, task, epoch):
                break

        stopper.restore_best(self.model)
        return TrainMetrics(
            task_id=task.task_id,
            loss=total_loss / max(n_steps, 1),
            n_steps=n_steps,
        )

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Snapshot current task's LoRA A matrices for future orthogonality constraint."""
        snapshot = {
            name: A.detach().cpu().clone() for name, A in self._get_current_A_matrices().items()
        }
        self.state.custom["past_A_matrices"].append(snapshot)
