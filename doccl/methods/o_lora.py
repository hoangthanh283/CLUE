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
import torch.nn.functional as F
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

    def _ortho_loss(self) -> torch.Tensor:
        """Penalty: ||A_t^T · A_past||_F^2 summed across layers and past tasks."""
        if not self.state.custom["past_A_matrices"]:
            return torch.zeros((), device=self.device)

        penalty = torch.zeros((), device=self.device)
        # Get current task's LoRA A matrices (active adapter)
        current_A = self._get_current_A_matrices()

        for past_A_dict in self.state.custom["past_A_matrices"]:
            for layer_name, A_curr in current_A.items():
                if layer_name in past_A_dict:
                    A_past = past_A_dict[layer_name].to(self.device)
                    # Penalty: ||A_curr^T · A_past||_F^2
                    inner = A_curr.T @ A_past  # (r, r)
                    penalty = penalty + (inner ** 2).sum()
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
            pbar = tqdm(train_loader, desc=f"OLoRA T{task.task_id} ep{epoch+1}/{epochs}", leave=False)
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
                pbar.set_postfix({"ce": f"{ce_loss.item():.3f}", "ortho": f"{ortho_loss.item():.4f}"})
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
            name: A.detach().cpu().clone()
            for name, A in self._get_current_A_matrices().items()
        }
        self.state.custom["past_A_matrices"].append(snapshot)
