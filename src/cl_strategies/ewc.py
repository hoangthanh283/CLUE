"""Elastic Weight Consolidation (EWC)."""

from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from src.cl_strategies.base import BaseCLStrategy


class EWC(BaseCLStrategy):
    """Diagonal-Fisher EWC.

    L_total = L_task + (lambda/2) * sum_i F_i (theta_i - theta*_i)^2
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lambda_ewc = float(config.get("cl_strategy", {}).get("ewc_lambda", 0.4))
        self.fishers: List[Dict[str, torch.Tensor]] = []
        self.opt_params: List[Dict[str, torch.Tensor]] = []

    @torch.no_grad()
    def _snapshot_params(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def _init_zero_like(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {n: torch.zeros_like(p, device=p.device) for n, p in model.named_parameters() if p.requires_grad}

    def _estimate_fisher(self, model: nn.Module, loader: Iterable) -> Dict[str, torch.Tensor]:
        model.eval()
        fisher = self._init_zero_like(model)
        n_batches = 0
        for batch in loader:
            n_batches += 1
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            for (name, param) in model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    fisher[name] += (param.grad.detach() ** 2)
        if n_batches > 0:
            for name in fisher:
                fisher[name] /= float(n_batches)
        return {k: v.detach().clone() for k, v in fisher.items()}

    def before_task(self, model: nn.Module, task_id: int, train_loader: Optional[Iterable] = None):
        return

    def after_task(self, model: nn.Module, task_id: int, train_loader: Optional[Iterable] = None):
        if train_loader is None:
            raise ValueError("EWC.after_task requires the train_loader to estimate Fisher")
        self.opt_params.append(self._snapshot_params(model))
        self.fishers.append(self._estimate_fisher(model, train_loader))

    def compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]
                     ) -> torch.Tensor:
        loss = outputs["loss"]
        if not self.fishers:
            return loss
        penalty = 0.0
        for task_idx in range(len(self.fishers)):
            fisher = self.fishers[task_idx]
            params_star = self.opt_params[task_idx]
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                diff = param - params_star[name]
                penalty = penalty + (fisher[name] * (diff ** 2)).sum()
        return loss + (self.lambda_ewc / 2.0) * penalty
