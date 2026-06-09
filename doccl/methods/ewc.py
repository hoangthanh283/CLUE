"""Elastic Weight Consolidation (EWC).

Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks",
PNAS 2017.

Loss: L_total = L_CE(current task) + (λ/2) * Σ_i F_i * (θ_i - θ_i*)^2
where F_i is empirical Fisher information for parameter i,
and θ_i* is the parameter value at the end of the previous task.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.eval.fisher import empirical_fisher_diagonal
from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo, TrainMetrics


class EWC(NaiveFineTune):
    """Online EWC with Fisher accumulation across tasks."""

    name = "ewc"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.state.custom["theta_star"] = {}  # name → snapshot tensor
        self.state.custom["fisher"] = {}  # name → Fisher diagonal tensor
        self.lambda_ = config.get("lambda_", 1000.0)
        self.fisher_n_samples = config.get("fisher_n_samples", 200)

    def _ewc_penalty(self) -> torch.Tensor:
        """Quadratic penalty pulling current params toward θ* weighted by Fisher."""
        if not self.state.custom["fisher"]:
            return torch.zeros((), device=self.device)
        penalty = torch.zeros((), device=self.device)
        params = dict(self.model.named_parameters())
        for name, fisher_val in self.state.custom["fisher"].items():
            if name not in params or name not in self.state.custom["theta_star"]:
                continue
            p = params[name]
            theta_star = self.state.custom["theta_star"][name]
            penalty = penalty + (fisher_val * (p - theta_star) ** 2).sum()
        return penalty

    def train_task(self, task: TaskInfo, train_loader: DataLoader) -> TrainMetrics:
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.trainable_parameters(),
            lr=self.config.get("lr", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)

        total_loss = 0.0
        n_steps = 0
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"EWC T{task.task_id} ep{epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                optimizer.zero_grad()
                outputs = self.model(**batch)
                ce_loss = outputs.loss
                ewc_loss = (self.lambda_ / 2) * self._ewc_penalty()
                loss = ce_loss + ewc_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), max_grad_norm)
                optimizer.step()
                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix({"ce": f"{ce_loss.item():.3f}", "ewc": f"{ewc_loss.item():.3f}"})

        return TrainMetrics(
            task_id=task.task_id,
            loss=total_loss / max(n_steps, 1),
            n_steps=n_steps,
            extra={"ewc_penalty_final": float(ewc_loss.item())},
        )

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Update Fisher and θ* using current task's data."""
        # Snapshot current params as θ*
        self.state.custom["theta_star"] = {
            name: p.detach().clone()
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }

        # Compute Fisher on current task data
        new_fisher = empirical_fisher_diagonal(
            self.model, train_loader, n_samples=self.fisher_n_samples, device=self.device
        )

        # Online EWC: accumulate Fisher across tasks (weighted sum)
        gamma = self.config.get("ewc_gamma", 1.0)  # 1.0 = simple sum
        for name, f in new_fisher.items():
            if name in self.state.custom["fisher"]:
                self.state.custom["fisher"][name] = gamma * self.state.custom["fisher"][name] + f
            else:
                self.state.custom["fisher"][name] = f
