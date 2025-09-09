"""Gradient Episodic Memory (GEM)."""

import logging
from typing import Any, Dict

import quadprog
import torch
import torch.nn as nn

from src.cl_strategies.base import BaseCLStrategy
from src.cl_strategies.memory import MemoryBuffer
from src.cl_strategies.utils import get_grad_vector, set_grad_vector

logger = logging.getLogger(__name__)


class GEM(BaseCLStrategy):
    """Gradient Episodic Memory (GEM) implementation.

    GEM uses quadratic programming to project gradients such that they don't
    increase loss on previous tasks. It maintains episodic memory and solves
    a constrained optimization problem to find the closest gradient that
    satisfies the constraints.

    Reference: https://arxiv.org/abs/1706.08840
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        cl_cfg = config.get("cl_strategy", {})
        mem_size = int(cl_cfg.get("memory_size", 2000))
        self.ref_batch_size = int(cl_cfg.get("replay_batch_size", 32))
        # Micro-batch size for computing memory gradients to reduce peak GPU RAM
        self.ref_micro_batch_size = int(cl_cfg.get("replay_microbatch_size", 1))
        self.memory = MemoryBuffer(mem_size)

        # Configuration
        self.constraint_tolerance = float(cl_cfg.get("constraint_tolerance", 0.0))
        self.max_constraints = int(cl_cfg.get("max_constraints", 3))

        # Statistics
        self.qp_solve_count = 0
        self.qp_fail_count = 0
        self.agem_fallback_count = 0

    def update_memory(self, batch: Dict[str, torch.Tensor]):
        self.memory.add_batch(batch)

    def get_solver_stats(self) -> Dict[str, int]:
        """Get statistics about QP solver performance."""
        total_attempts = self.qp_solve_count + self.qp_fail_count
        return {
            "qp_solves": self.qp_solve_count,
            "qp_failures": self.qp_fail_count,
            "agem_fallbacks": self.agem_fallback_count,
            "total_attempts": total_attempts,
            "qp_success_rate": self.qp_solve_count / max(1, total_attempts)
        }

    def on_before_backward(self, model: nn.Module, loss: torch.Tensor):
        """GEM with CPU-based gradients and dual QP to reduce memory usage.

        - Builds memory gradients first and moves them to CPU (avoids GPU spikes)
        - For K==1, performs A-GEM projection without stacking/averaging
        - For K>1, solves the dual QP in KxK space on CPU
        """
        device = next(model.parameters()).device
        if len(self.memory) == 0:
            return

        # 1) Compute current gradient first and free its graph
        model.zero_grad(set_to_none=True)
        loss.backward()
        g = get_grad_vector(model)  # CPU flat grad
        model.zero_grad(set_to_none=True)

        # 2) Build constraints from multiple memory minibatches (gradients on CPU)
        K = min(self.max_constraints, max(1, len(self.memory) // max(1, self.ref_batch_size)))
        G_list = []  # each element is [P] on CPU

        def _iter_slices(batch: Dict[str, torch.Tensor], micro_bs: int):
            # Yield micro-batches sliced along dim 0 for available keys
            keys = [k for k, v in batch.items() if isinstance(v, torch.Tensor) and v.dim() > 0 and v.size(0) > 0]
            if not keys:
                return
            B = int(batch[keys[0]].size(0))
            micro_bs = max(1, min(micro_bs, B))
            for start in range(0, B, micro_bs):
                end = min(B, start + micro_bs)
                yield {k: (v[start:end] if k in batch else None) for k, v in batch.items() if v is not None}

        def _compute_avg_grad_for_batch(mem_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
            # Compute gradient of average loss over mem_batch using token-weighted micro-batches
            model.zero_grad(set_to_none=True)
            # Determine total batch size B from any present tensor
            any_tensor = next((v for v in mem_batch.values() if isinstance(v, torch.Tensor)), None)
            B = int(any_tensor.size(0)) if any_tensor is not None else 1
            # Token count for weighting (aligns with CrossEntropy mean over active tokens)
            attn = mem_batch.get("attention_mask")
            total_tokens = int(attn.sum().item()) if attn is not None else B
            if total_tokens <= 0:
                # Fallback to equal weighting to avoid div-by-zero
                total_tokens = B
            for sub in _iter_slices(mem_batch, self.ref_micro_batch_size):
                outputs = model(**sub)
                sub_loss = outputs["loss"]
                if "attention_mask" in sub and sub["attention_mask"] is not None:
                    sub_tokens = int(sub["attention_mask"].sum().item())
                else:
                    sub_tokens = int(sub["input_ids"].size(0))
                # Weight by token count and normalize by total
                sub_weight = float(sub_tokens) / float(total_tokens)
                sub_loss = sub_loss * sub_weight
                sub_loss.backward()
            g_mem = get_grad_vector(model)  # CPU
            model.zero_grad(set_to_none=True)
            return g_mem

        for _ in range(K):
            mem_batch = self.memory.sample(self.ref_batch_size, device=device)
            if mem_batch is None:
                break
            g_mem = _compute_avg_grad_for_batch(mem_batch)
            G_list.append(g_mem)

        if not G_list:
            # No memory constraints available; use current gradient as-is
            set_grad_vector(model, g)
            return

        # Fallback to A-GEM if only 1 constraint (avoid stacking/mean)
        if len(G_list) == 1:
            logger.debug("Falling back to A-GEM projection (K=1)")
            self.agem_fallback_count += 1
            g_ref = G_list[0]
            dot_product = torch.dot(g, g_ref)
            if dot_product < 0:
                g_ref_norm_sq = g_ref.norm() ** 2 + 1e-12
                proj = g - (dot_product / g_ref_norm_sq) * g_ref
                set_grad_vector(model, proj)
            else:
                # No conflict; keep current gradient
                set_grad_vector(model, g)
            return

        # Solve dual QP on CPU: min_{lambda>=0} 0.5*lambda^T (G^T G) lambda + (G^T g)^T lambda
        # Then v = g + G lambda
        try:
            # Stack constraints to [P, K] on CPU (keep float32 to save RAM)
            G_cpu = torch.stack(G_list, dim=1)  # [P, K], dtype ~ float32
            g_cpu = g  # [P], CPU float

            # Dual matrices (compute in float32 then cast to float64 for solver)
            P_dual = (G_cpu.T @ G_cpu).double().numpy()  # [K, K]
            q_dual = (G_cpu.T @ g_cpu).double().numpy()   # [K]

            # Constraints: lambda >= 0  ->  I^T lambda >= 0
            G_ineq = torch.eye(G_cpu.size(1), dtype=torch.double).numpy()  # [K, K]
            h_ineq = torch.zeros(G_cpu.size(1), dtype=torch.double).numpy()  # [K]

            # Solve for lambda
            # Optional tolerance term: G^T v >= -tol  =>  q += tol
            if self.constraint_tolerance > 0:
                q_dual = q_dual + self.constraint_tolerance

            lam = quadprog.solve_qp(P_dual, q_dual, G_ineq, h_ineq)[0]  # [K]

            # Compose projected gradient
            lam_t = torch.from_numpy(lam).to(dtype=g_cpu.dtype)  # match float32
            v = g_cpu + G_cpu @ lam_t  # [P], float32

            # Validate constraint satisfaction (optional)
            constraints = (G_cpu.T @ v).double().numpy()
            violations = (constraints < -self.constraint_tolerance - 1e-9).sum()
            if violations > 0:
                logger.warning(f"GEM dual QP solution has {violations} constraint violations")

            set_grad_vector(model, v)
            self.qp_solve_count += 1
            logger.debug(f"GEM dual QP solved with K={G_cpu.size(1)} constraints")

        except Exception as e:
            logger.warning(f"GEM QP solver failed: {e}, falling back to A-GEM")
            self.qp_fail_count += 1
            # Use mean of memory gradients (CPU) as reference for A-GEM style projection
            g_ref = torch.stack(G_list, dim=1).mean(dim=1)
            dot_product = torch.dot(g, g_ref)
            if dot_product < 0:
                g_ref_norm_sq = g_ref.norm() ** 2 + 1e-12
                proj = g - (dot_product / g_ref_norm_sq) * g_ref
                set_grad_vector(model, proj)
            else:
                # No conflict; keep current gradient
                set_grad_vector(model, g)
