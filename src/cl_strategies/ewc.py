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
        # Store EWC buffers (Fisher and optimal params) on CPU to save GPU memory.
        # They are moved to the active device on-the-fly during compute_loss.
        self.store_on_cpu: bool = bool(config.get("cl_strategy", {}).get("ewc_store_on_cpu", True))
        # Max number of batches used for Fisher estimation (configurable)
        self.max_fisher_batches: int = int(config.get("cl_strategy", {}).get("ewc_max_fisher_batches", 500))
        self.fishers: List[Dict[str, torch.Tensor]] = []
        self.opt_params: List[Dict[str, torch.Tensor]] = []

    @torch.no_grad()
    def _snapshot_params(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        params = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        if self.store_on_cpu:
            return {k: v.to("cpu") for k, v in params.items()}
        return params

    def _init_zero_like(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        # Allocate on the same device as parameters for efficient accumulation.
        return {n: torch.zeros_like(p, device=p.device) for n, p in model.named_parameters() if p.requires_grad}

    def _estimate_fisher(self, model: nn.Module, loader: Iterable) -> Dict[str, torch.Tensor]:
        # Switch to eval for a stabler estimate (dropout off); restore mode after
        was_training = model.training
        model.train(False)
        fisher = self._init_zero_like(model)
        n_batches = 0
        
        # Limit the number of batches used for Fisher estimation to save memory
        max_fisher_batches = self.max_fisher_batches
        if hasattr(loader, "__len__"):
            max_fisher_batches = min(self.max_fisher_batches, len(loader))
        for batch in loader:
            if n_batches >= max_fisher_batches:
                break

            n_batches += 1
            device = next(model.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}
            model.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            
            # Accumulate Fisher information and immediately clear gradients to save memory
            for (name, param) in model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    fisher[name] += (param.grad.detach() ** 2)
            
            # Clear gradients immediately to free memory
            model.zero_grad(set_to_none=True)
            
            # Periodic GPU memory cleanup
            if n_batches % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        if n_batches > 0:
            for name in fisher:
                fisher[name] /= float(n_batches)
                
        # Detach and optionally move to CPU for storage.
        fisher_detached = {k: v.detach().clone() for k, v in fisher.items()}
        if self.store_on_cpu:
            fisher_detached = {k: v.to("cpu") for k, v in fisher_detached.items()}
        # Restore original mode
        model.train(was_training)
        return fisher_detached

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

        # Accumulate penalty on active device with autograd enabled.
        penalty = torch.zeros((), device=loss.device, dtype=loss.dtype)

        import sys
        total_bytes = sum(sum(tensor.element_size() * tensor.nelement() for tensor in fisher.values()) for fisher in self.fishers)
        print(f"Size of self.fishers: {len(self.fishers)}, total size: {total_bytes / (1024 ** 2):.2f} MB")

        for task_idx in range(len(self.fishers)):
            fisher = self.fishers[task_idx]
            params_star = self.opt_params[task_idx]
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in params_star or name not in fisher:
                    # Parameter added after earlier tasks or missing fisher; skip.
                    continue

                # Bring stored tensors to the parameter device/dtype (constants, no grads)
                prev_param = params_star[name].to(param.device, dtype=param.dtype)
                F = fisher[name].to(param.device, dtype=param.dtype)

                # Handle parameter size mismatches (e.g., classifier expansion in class-IL)
                if param.shape != prev_param.shape:
                    # Only apply penalty to the overlapping dimensions
                    min_shape = tuple(min(p, pp) for p, pp in zip(param.shape, prev_param.shape))
                    slices = tuple(slice(0, s) for s in min_shape)

                    param_slice = param[slices]
                    prev_param_slice = prev_param[slices]
                    F_slice = F[slices]

                    diff = param_slice - prev_param_slice
                    penalty = penalty + (F_slice * diff.pow(2)).sum()
                else:
                    diff = param - prev_param
                    penalty = penalty + (F * diff.pow(2)).sum()
        return loss + (self.lambda_ewc / 2.0) * penalty
