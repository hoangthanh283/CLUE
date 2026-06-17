"""Sharpness-Aware Minimisation (SAM) two-step, with the C-Flat curvature term.

Reference:
    - Foret et al., "Sharpness-Aware Minimization for Efficiently Improving
      Generalization", ICLR 2021, arXiv:2010.01412 (the base two-step).
    - Bian et al., "Make Continual Learning Stronger via C-Flat", NeurIPS 2024,
      arXiv:2404.00986; C-Flat++ arXiv:2508.18860 — adds a zeroth-order curvature
      penalty on top of SAM so the optimiser is pulled toward *flat* minima, which
      empirically reduces forgetting when bolted onto any CL method.

This is a standalone helper (not a method) so any CL strategy can swap its single
``loss.backward(); optimizer.step()`` for the flat-minima two-step. The canonical
use is::

    sam = SAMStep(params, rho=0.05, cflat_lambda=0.1)
    # step 1: gradient at θ
    optimizer.zero_grad(); loss = forward(); loss.backward()
    clip_grad_norm_(params, max_grad_norm)
    eps = sam.first_step()                 # ascend to the worst-case neighbour θ+ε
    # step 2: gradient at θ+ε
    optimizer.zero_grad(); loss2 = forward(); loss2.backward()
    sam.second_step(eps)                   # restore θ, optionally add curvature, then step
    clip_grad_norm_(params, max_grad_norm) # (caller clips before optimizer.step)
    optimizer.step()

The helper only manipulates ``.grad`` / ``.data`` of the tracked params and the
perturbation ``eps``; the caller owns the optimiser, the forward passes, and grad
clipping. This keeps it composable with replay losses, AMP, etc.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Numerical floor for the perturbation-scaling denominator ||g||. Below this the
# gradient is effectively zero and we skip the ascent (ε = 0) rather than divide by ~0.
_GRAD_NORM_EPS = 1e-12


class SAMStep:
    """Stateless-ish SAM two-step over a fixed parameter list.

    Args:
        params: the parameters being optimised (same list given to the optimiser).
        rho: neighbourhood radius for the ascent step (ε = rho · g/‖g‖). 0.05 is
            the SAM default; larger = flatter-but-looser minima.
        cflat_lambda: weight of the C-Flat zeroth-order curvature term. ``0.0``
            recovers plain SAM. The curvature direction is the difference between
            the perturbed-point gradient and the clean gradient (g₊ − g₀), which
            approximates the top-curvature direction without a Hessian.
    """

    def __init__(self, params: list[nn.Parameter], rho: float = 0.05, cflat_lambda: float = 0.0):
        self.params = [p for p in params if p.requires_grad]
        self.rho = float(rho)
        self.cflat_lambda = float(cflat_lambda)
        # Scale used in first_step (rho/‖g‖); kept so second_step can recover the
        # clean gradient g₀ = eps / scale WITHOUT a separate full-size clone (the eps
        # list already holds g₀·scale). This keeps the C-Flat path memory-flat — a
        # naive g₀ clone doubled gradient memory and OOM'd the 6 GB box.
        self._scale: float = 0.0

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        """‖g‖₂ over all tracked params (the SAM scaling denominator)."""
        norms = [p.grad.detach().norm(2) for p in self.params if p.grad is not None]
        if not norms:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(norms), 2)

    @torch.no_grad()
    def first_step(self) -> list[torch.Tensor]:
        """Ascend to the worst-case neighbour θ+ε and return the per-param ε.

        Must be called right after the *clean* backward (grads = g₀). The returned ε
        tensors (= g₀·scale) double as the cached clean gradient: ``second_step``
        recovers g₀ = ε/scale, so no separate g₀ buffer is allocated even when the
        C-Flat curvature term is active.
        """
        grad_norm = self._grad_norm()
        self._scale = float(self.rho / (grad_norm + _GRAD_NORM_EPS))
        eps_list: list[torch.Tensor] = []
        for p in self.params:
            if p.grad is None:
                eps_list.append(torch.zeros_like(p))
                continue
            eps = p.grad.detach() * self._scale
            p.add_(eps)  # θ ← θ + ε
            eps_list.append(eps)
        return eps_list

    @torch.no_grad()
    def second_step(self, eps_list: list[torch.Tensor]) -> None:
        """Restore θ and set ``.grad`` to the descent direction the optimiser uses.

        Called after the *perturbed* backward (grads = g₊). Undoes the ε ascent so
        the optimiser steps from the original θ, and — when ``cflat_lambda > 0`` —
        blends in the C-Flat curvature direction (g₊ − g₀) so the update favours
        flat regions. With ``cflat_lambda == 0`` the resulting grad is exactly g₊
        (plain SAM). The caller clips and calls ``optimizer.step()`` afterwards.
        """
        recover_g0 = self.cflat_lambda > 0 and self._scale > _GRAD_NORM_EPS
        for p, eps in zip(self.params, eps_list, strict=True):
            p.sub_(eps)  # θ ← θ (restore)
            if p.grad is None:
                continue
            if recover_g0:
                # g₀ = ε / scale (no separate clone); g_used = g₊ + λ·(g₊ − g₀).
                g0 = eps / self._scale
                curvature = p.grad.detach() - g0
                p.grad.add_(curvature, alpha=self.cflat_lambda)
