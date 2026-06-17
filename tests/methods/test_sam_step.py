"""Unit tests for the SAM / C-Flat two-step (doccl.methods.sam_step.SAMStep).

Pure-logic tests (no GPU, no data): verify that the ascent perturbs by the right
radius, that the descent restores the original weights, and that the C-Flat
curvature blend behaves as specified (off when lambda=0, additive when >0).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from doccl.methods.sam_step import SAMStep


def _toy() -> nn.Linear:
    torch.manual_seed(0)
    return nn.Linear(4, 3)


def _set_grads(params: list[nn.Parameter], value: float = 1.0) -> None:
    for p in params:
        p.grad = torch.full_like(p, value)


def test_first_step_perturbs_by_rho_scaled_unit_gradient() -> None:
    # Arrange: a single param with a known gradient → ||g|| is computable.
    m = _toy()
    params = list(m.parameters())
    before = [p.detach().clone() for p in params]
    _set_grads(params, value=2.0)
    rho = 0.05

    sam = SAMStep(params, rho=rho, cflat_lambda=0.0)

    # Compute expected ||g|| over all params (matches SAMStep._grad_norm).
    gnorm = torch.norm(torch.stack([p.grad.norm(2) for p in params]), 2)
    scale = rho / (gnorm + 1e-12)

    # Act
    eps = sam.first_step()

    # Assert: each param moved by exactly g * scale.
    for p, b, e in zip(params, before, eps, strict=True):
        assert torch.allclose(p.data, b + b.new_full(b.shape, 2.0) * scale)
        assert torch.allclose(e, e.new_full(e.shape, 2.0) * scale)


def test_second_step_restores_weights_plain_sam() -> None:
    # Arrange
    m = _toy()
    params = list(m.parameters())
    before = [p.detach().clone() for p in params]
    _set_grads(params, value=1.0)
    sam = SAMStep(params, rho=0.05, cflat_lambda=0.0)

    # Act: ascend, then (simulating a 2nd backward) descend.
    eps = sam.first_step()
    _set_grads(params, value=3.0)  # g_perturbed
    sam.second_step(eps)

    # Assert: weights restored to the original θ; grad is exactly g_perturbed (SAM).
    for p, b in zip(params, before, strict=True):
        assert torch.allclose(p.data, b)
        assert torch.allclose(p.grad, p.grad.new_full(p.grad.shape, 3.0))


def test_cflat_blends_curvature_direction() -> None:
    # Arrange: g0 = 1, g_perturbed = 3, lambda = 0.5
    # g_used = g+ + lambda*(g+ - g0) = 3 + 0.5*(3 - 1) = 4.0
    m = _toy()
    params = list(m.parameters())
    _set_grads(params, value=1.0)  # g0
    sam = SAMStep(params, rho=0.05, cflat_lambda=0.5)

    # Act
    eps = sam.first_step()  # caches g0 = 1
    _set_grads(params, value=3.0)  # g_perturbed
    sam.second_step(eps)

    # Assert
    for p in params:
        assert torch.allclose(p.grad, p.grad.new_full(p.grad.shape, 4.0))


def test_zero_gradient_is_safe() -> None:
    # Arrange: all grads zero → ||g|| ~ 0, perturbation must be ~0 (no div-by-zero blowup).
    m = _toy()
    params = list(m.parameters())
    before = [p.detach().clone() for p in params]
    _set_grads(params, value=0.0)
    sam = SAMStep(params, rho=0.05, cflat_lambda=0.1)

    # Act
    eps = sam.first_step()
    sam.second_step(eps)

    # Assert: weights unchanged and no NaNs introduced.
    for p, b in zip(params, before, strict=True):
        assert torch.allclose(p.data, b)
        assert torch.isfinite(p.data).all()
