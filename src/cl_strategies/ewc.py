"""Elastic Weight Consolidation (EWC)."""

from typing import Any, Dict, Iterable, Optional
import os
import pickle
import gc

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

        # Disk storage for memory efficiency
        self.ewc_cache_dir = config.get("cl_strategy", {}).get("ewc_cache_dir", "ewc_cache")
        os.makedirs(self.ewc_cache_dir, exist_ok=True)
        self.num_tasks = 0

        # Memory optimization settings
        self.param_chunk_size: int = int(config.get("cl_strategy", {}).get("param_chunk_size", 300))
        self.fisher_sparsity: float = float(config.get("cl_strategy", {}).get("fisher_sparsity", 0.9))

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

    def _sparsify_fisher(self, fisher: Dict[str, torch.Tensor], sparsity_ratio: float = 0.9) -> Dict[str, torch.Tensor]:
        """Keep only top-(1-sparsity_ratio) most important Fisher values."""
        sparse_fisher = {}
        for name, F in fisher.items():
            F_flat = F.view(-1)
            k = max(1, int((1 - sparsity_ratio) * F_flat.numel()))

            # Get top-k indices and values
            _, topk_indices = torch.topk(F_flat.abs(), k)

            # Create sparse representation
            sparse_F = torch.zeros_like(F_flat)
            sparse_F[topk_indices] = F_flat[topk_indices]
            sparse_fisher[name] = sparse_F.view(F.shape)

        return sparse_fisher

    def _save_fisher_and_params(self, fisher: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor], task_id: int):
        """Save Fisher matrix and optimal parameters to disk."""
        fisher_path = os.path.join(self.ewc_cache_dir, f"fisher_task_{task_id}.pkl")
        params_path = os.path.join(self.ewc_cache_dir, f"params_task_{task_id}.pkl")

        with open(fisher_path, "wb") as f:
            pickle.dump(fisher, f)
        with open(params_path, "wb") as f:
            pickle.dump(params, f)

    def _load_fisher_and_params(self, task_id: int):
        """Load Fisher matrix and optimal parameters from disk."""
        fisher_path = os.path.join(self.ewc_cache_dir, f"fisher_task_{task_id}.pkl")
        params_path = os.path.join(self.ewc_cache_dir, f"params_task_{task_id}.pkl")

        if not os.path.exists(fisher_path) or not os.path.exists(params_path):
            return None, None

        with open(fisher_path, "rb") as f:
            fisher = pickle.load(f)
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        return fisher, params

    def before_task(self, model: nn.Module, task_id: int, train_loader: Optional[Iterable] = None):
        return

    def after_task(self, model: nn.Module, task_id: int, train_loader: Optional[Iterable] = None):
        if train_loader is None:
            raise ValueError("EWC.after_task requires the train_loader to estimate Fisher")

        # Compute Fisher and snapshot parameters
        optimal_params = self._snapshot_params(model)
        fisher = self._estimate_fisher(model, train_loader)

        # Apply sparsification to reduce memory usage
        fisher = self._sparsify_fisher(fisher, self.fisher_sparsity)

        # Save to disk instead of storing in memory
        self._save_fisher_and_params(fisher, optimal_params, task_id)
        self.num_tasks += 1

        # Force garbage collection
        del fisher, optimal_params
        gc.collect()

    def compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]
                     ) -> torch.Tensor:
        loss = outputs["loss"]
        if self.num_tasks == 0:
            return loss

        # Process tasks one by one to minimize memory usage
        total_penalty = 0.0

        for task_idx in range(self.num_tasks):
            # Load only ONE task's Fisher matrix and parameters
            fisher, params_star = self._load_fisher_and_params(task_idx)
            if fisher is None or params_star is None:
                continue

            # Compute penalty for this task with parameter chunking
            task_penalty = self._compute_task_penalty_chunked(model, fisher, params_star)
            total_penalty += task_penalty

            # CRITICAL: Free memory immediately after processing each task
            del fisher, params_star
            gc.collect()

        penalty = torch.tensor(total_penalty, device=loss.device, dtype=loss.dtype)
        return loss + (self.lambda_ewc / 2.0) * penalty

    def _compute_task_penalty_chunked(self, model: nn.Module, fisher: Dict[str, torch.Tensor],
                                      params_star: Dict[str, torch.Tensor]) -> float:
        """Compute EWC penalty for one task using parameter chunking."""
        penalty = 0.0
        # Get list of parameters for chunking
        param_items = [(name, param) for name, param in model.named_parameters() if param.requires_grad]

        # Process parameters in small chunks to fit in 15GB RAM
        for i in range(0, len(param_items), self.param_chunk_size):
            chunk_params = param_items[i:i + self.param_chunk_size]
            chunk_penalty = self._compute_chunk_penalty(chunk_params, fisher, params_star)
            penalty += chunk_penalty

            # Aggressive garbage collection after each chunk
            if i % (self.param_chunk_size * 2) == 0:
                gc.collect()
        return penalty

    def _compute_chunk_penalty(self, chunk_params, fisher: Dict[str, torch.Tensor],
                               params_star: Dict[str, torch.Tensor]) -> float:
        """Compute penalty for a chunk of parameters on CPU."""
        chunk_penalty = 0.0
        for name, param in chunk_params:
            if name not in params_star or name not in fisher:
                continue

            # Move current parameter to CPU for computation (preserves gradients)
            param_cpu = param.cpu()

            # Fisher and prev_param are already on CPU from disk storage
            prev_param_cpu = params_star[name]
            F_cpu = fisher[name]

            # Handle parameter size mismatches
            if param_cpu.shape != prev_param_cpu.shape:
                min_shape = tuple(min(p, pp) for p, pp in zip(param_cpu.shape, prev_param_cpu.shape))
                slices = tuple(slice(0, s) for s in min_shape)

                diff = param_cpu[slices] - prev_param_cpu[slices]
                chunk_penalty += (F_cpu[slices] * diff.pow(2)).sum().item()
            else:
                diff = param_cpu - prev_param_cpu
                chunk_penalty += (F_cpu * diff.pow(2)).sum().item()
        return chunk_penalty
