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
            # The class-incremental classifier head grows across tasks, so the current
            # param, the snapshotted theta_star, and the accumulated Fisher can each have
            # a different leading (class) dimension (e.g. 25 vs 13 vs 25). Penalise only
            # the rows present in ALL THREE — the old classes that have a prior — by
            # slicing each to their common minimum shape. New class rows carry no EWC
            # anchor, which is correct. This keeps the penalty well-defined regardless of
            # which snapshot each tensor came from.
            min_shape = tuple(
                min(a, b, c) for a, b, c in zip(p.shape, theta_star.shape, fisher_val.shape)
            )
            idx = tuple(slice(0, s) for s in min_shape)
            penalty = penalty + (fisher_val[idx] * (p[idx] - theta_star[idx]) ** 2).sum()
        return penalty

    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.trainable_parameters(),
            lr=self.config.get("lr", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        stopper = self.make_early_stopper(val_loader)

        total_loss = 0.0
        n_steps = 0
        ewc_loss = torch.zeros((), device=self.device)
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
            if self._early_stop_after_epoch(stopper, val_loader, task, epoch):
                break

        stopper.restore_best(self.model)
        return TrainMetrics(
            task_id=task.task_id,
            loss=total_loss / max(n_steps, 1),
            n_steps=n_steps,
            # Penalty at the last *trained* epoch (a diagnostic only). After
            # restore_best the model weights are the best-val checkpoint, which may
            # differ — this value is not recomputed on the restored weights.
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

        # Online EWC: accumulate Fisher across tasks (weighted sum). The class-incremental
        # classifier head grows between tasks, so the previous Fisher can be smaller than the
        # new one (e.g. [13,768] vs [25,768]). Pad the old accumulator up to the new shape with
        # zeros (new class rows carry no prior importance) before summing, so accumulation never
        # crashes on the shape mismatch.
        gamma = self.config.get("ewc_gamma", 1.0)  # 1.0 = simple sum
        for name, f in new_fisher.items():
            old = self.state.custom["fisher"].get(name)
            if old is None:
                self.state.custom["fisher"][name] = f
            elif old.shape == f.shape:
                self.state.custom["fisher"][name] = gamma * old + f
            else:
                padded = torch.zeros_like(f)
                idx = tuple(slice(0, s) for s in old.shape)
                padded[idx] = old
                self.state.custom["fisher"][name] = gamma * padded + f
