"""Gradient Episodic Memory (GEM)."""

from typing import Any, Dict

import torch
import quadprog
import torch.nn as nn

from src.cl_strategies.base import BaseCLStrategy
from src.cl_strategies.memory import MemoryBuffer
from src.cl_strategies.utils import get_grad_vector, set_grad_vector


class GEM(BaseCLStrategy):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        cl_cfg = config.get("cl_strategy", {})
        mem_size = int(cl_cfg.get("memory_size", 2000))
        self.ref_batch_size = int(cl_cfg.get("replay_batch_size", 32))
        self.memory = MemoryBuffer(mem_size)

    def update_memory(self, batch: Dict[str, torch.Tensor]):
        self.memory.add_batch(batch)

    def on_before_backward(self, model: nn.Module, loss: torch.Tensor):
        device = next(model.parameters()).device
        if len(self.memory) == 0:
            return

        # Current gradient
        model.zero_grad(set_to_none=True)
        loss.backward()
        g = get_grad_vector(model)

        # Build constraints from multiple memory minibatches
        K = min(3, max(1, len(self.memory) // max(1, self.ref_batch_size)))
        G_list = []
        for _ in range(K):
            mem_batch = self.memory.sample(self.ref_batch_size, device=device)
            if mem_batch is None:
                break
            model.zero_grad(set_to_none=True)
            mem_out = model(**mem_batch)
            mem_out["loss"].backward()
            g_mem = get_grad_vector(model).detach()
            G_list.append(g_mem)

        if not G_list:
            return

        G = torch.stack(G_list, dim=1)  # [P, K]
        # Solve QP: min 0.5 ||v - g||^2 s.t. G^T v >= 0
        g_np = g.detach().cpu().double().numpy()
        G_np = G.detach().cpu().double().numpy()
        P = g_np.shape[0]

        Q = torch.eye(P, dtype=torch.double).numpy()
        c = -g_np
        A = G_np
        b = torch.zeros(G_np.shape[1], dtype=torch.double).numpy()

        sol = quadprog.solve_qp(Q, c, A, b)[0]
        v = torch.from_numpy(sol).to(g.device, dtype=g.dtype)
        set_grad_vector(model, v)
