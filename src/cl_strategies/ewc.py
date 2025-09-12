"""Elastic Weight Consolidation (EWC)."""

from typing import Any, Dict, Iterable, List, Optional
import os
import pickle

import torch
import random
import torch.nn as nn

from src.cl_strategies.base import BaseCLStrategy


class EWC(BaseCLStrategy):
    """Diagonal-Fisher EWC.

    L_total = L_task + (lambda/2) * sum_i F_i (theta_i - theta*_i)^2
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        cl_config = config.get("cl_strategy", {})
        self.lambda_ewc = float(config.get("cl_strategy", {}).get("ewc_lambda", 0.4))
        self.fisher_cache_dir = cl_config.get("fisher_cache_dir", "ewc_cache")
        self.n_fisher_samples = cl_config.get("n_fisher_samples", None)
        self.ewc_chunk_size = int(cl_config.get("ewc_chunk_size", 1_000_000))
        self.store_fishers_on_cpu = cl_config.get("store_fishers_on_cpu", True)
        self.stored_task_ids: List[int] = []

        # Create cache directory.
        os.makedirs(self.fisher_cache_dir, exist_ok=True)

    @torch.no_grad()
    def _snapshot_params(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def _init_zero_like(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {n: torch.zeros_like(p, device=p.device) for n, p in model.named_parameters() if p.requires_grad}

    def _estimate_fisher(self, model: nn.Module, loader: Iterable) -> Dict[str, torch.Tensor]:
        model.eval()
        fisher = self._init_zero_like(model)
        n_batches = 0
        if self.n_fisher_samples is not None:
            # Convert to list if we need to sample.
            loader_list = list(loader)
            if len(loader_list) > self.n_fisher_samples:
                loader = random.sample(loader_list, self.n_fisher_samples)
            else:
                loader = loader_list

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
        # Move Fisher matrices to CPU if configured to save GPU memory.
        if self.store_fishers_on_cpu:
            fisher = {k: v.detach().clone().cpu() for k, v in fisher.items()}
        else:
            fisher = {k: v.detach().clone() for k, v in fisher.items()}
        return fisher

    def before_task(self, model: nn.Module, task_id: int, train_loader: Optional[Iterable] = None):
        return

    def _save_fisher_data(self, task_id: int, fisher: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        """Save Fisher information and parameters to disk."""
        fisher_path = os.path.join(self.fisher_cache_dir, f"fisher_task_{task_id}.pkl")
        params_path = os.path.join(self.fisher_cache_dir, f"params_task_{task_id}.pkl")
        with open(fisher_path, "wb") as fp:
            pickle.dump(fisher, fp)
        with open(params_path, "wb") as fp:
            pickle.dump(params, fp)

    def _load_fisher_data(self, task_id: int):
        """Load Fisher information and parameters from disk."""
        fisher_path = os.path.join(self.fisher_cache_dir, f"fisher_task_{task_id}.pkl")
        params_path = os.path.join(self.fisher_cache_dir, f"params_task_{task_id}.pkl")
        with open(fisher_path, "rb") as fp:
            fisher = pickle.load(fp)
        with open(params_path, "rb") as fp:
            params = pickle.load(fp)
        return fisher, params

    def after_task(self, model: nn.Module, task_id: int, train_loader: Optional[Iterable] = None):
        if train_loader is None:
            raise ValueError("EWC.after_task requires the train_loader to estimate Fisher")

        params = self._snapshot_params(model)
        if self.store_fishers_on_cpu:
            params = {k: v.cpu() for k, v in params.items()}

        # Estimate Fisher information.
        fisher = self._estimate_fisher(model, train_loader)
        self._save_fisher_data(task_id, fisher, params)
        self.stored_task_ids.append(task_id)
        del fisher, params
        torch.cuda.empty_cache()

    def ewc_penalty_chunked(self, param, fisher_param, params_star_param):
        """Compute EWC penalty in chunks to avoid OOM.

        Args:
            param: Current model parameter tensor.
            fisher_param: Corresponding Fisher information tensor.
            params_star_param: Parameter tensor from previous task.

        Returns:
            The computed EWC penalty for the parameter tensor.
        """
        param_flat = param.flatten()
        fisher_flat = fisher_param.flatten()
        params_star_flat = params_star_param.flatten()
        penalty = 0.0

        for idx in range(0, param_flat.numel(), self.ewc_chunk_size):
            end_idx = min(idx + self.ewc_chunk_size, param_flat.numel())
            param_chunk = param_flat[idx:end_idx]
            fisher_chunk = fisher_flat[idx:end_idx].to(param.device, non_blocking=True)
            params_star_chunk = params_star_flat[idx:end_idx].to(param.device, non_blocking=True)
            diff_chunk = param_chunk - params_star_chunk
            
            # Compute penalty for this chunk and detach immediately to avoid gradient accumulation
            chunk_penalty = (fisher_chunk * (diff_chunk ** 2)).sum().detach()
            penalty += chunk_penalty.item()  # Convert to scalar to avoid tensor accumulation
            
            # Clean up intermediate tensors
            del fisher_chunk, params_star_chunk, diff_chunk, chunk_penalty
            
            # Clear cache every few chunks to prevent memory buildup
            if idx % (self.ewc_chunk_size * 10) == 0:
                torch.cuda.empty_cache()
        return torch.tensor(penalty, device=param.device, requires_grad=True)

    def compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]
                     ) -> torch.Tensor:
        """Compute total loss including EWC penalty.

        Args:
            model: The model being trained.
            batch: The current batch of data.
            outputs: The outputs from the model, must include "loss" key.

        Returns:
            The total loss including EWC penalty.
        """
        loss = outputs["loss"]
        if not self.stored_task_ids:
            return loss

        penalty = 0.0
        for task_id in self.stored_task_ids:
            fisher, params_star = self._load_fisher_data(task_id)
            for name, param in model.named_parameters():
                if name not in params_star or name not in fisher or not param.requires_grad:
                    # Parameter didn't exist in previous task, or fisher info not available for this parameter, skip.
                    continue

                fisher_param = fisher[name]
                if param.shape != params_star[name].shape or fisher_param.shape != param.shape:
                    # If parameter shape changed (e.g., classifier grew under class-IL), skip regularization for this
                    # tensor to avoid size mismatch.
                    continue

                params_star_param = params_star[name]
                if fisher_param.numel() > self.ewc_chunk_size:
                    # Use chunked computation to avoid OOM on large tensors.
                    penalty_term = self.ewc_penalty_chunked(param, fisher_param, params_star_param)
                else:
                    diff = param - params_star[name].to(param.device)
                    penalty_term = (fisher_param.to(param.device) * (diff ** 2)).sum()
                penalty += penalty_term

            del fisher, params_star
            torch.cuda.empty_cache()

        final_loss = loss + (self.lambda_ewc / 2.0) * penalty.to(loss.device)
        del penalty
        torch.cuda.empty_cache()
        return final_loss
