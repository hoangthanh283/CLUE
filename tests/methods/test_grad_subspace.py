# tests/methods/test_grad_subspace.py
from __future__ import annotations

import torch

from doccl.methods.grad_subspace import feature_moment, project_orth, steer_gradient, top_eigvecs


def test_feature_moment_is_gram():
    f = torch.randn(7, 5)
    assert torch.allclose(feature_moment(f), f.T @ f, atol=1e-5)


def test_top_eigvecs_orthonormal_and_dominant():
    torch.manual_seed(0)
    # rank-structured SPD: one strong direction.
    v = torch.randn(6, 1)
    m = v @ v.T * 10 + torch.eye(6) * 0.1
    u = top_eigvecs(m, k=2)
    assert u.shape == (6, 2)
    # columns orthonormal
    assert torch.allclose(u.T @ u, torch.eye(2), atol=1e-4)
    # leading eigvec aligns with v (up to sign)
    cos = abs(torch.nn.functional.cosine_similarity(u[:, 0], v[:, 0], dim=0).item())
    assert cos > 0.99


def test_project_orth_removes_in_subspace_component():
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(5, 2)).Q  # (5,2) orthonormal
    g = torch.randn(3, 5)
    g_orth = project_orth(g, basis)
    # g_orth has zero projection onto basis: (g_orth @ basis) ≈ 0
    assert torch.allclose(g_orth @ basis, torch.zeros(3, 2), atol=1e-5)


def test_steer_alpha0_equals_orth_alpha1_equals_identity():
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(5, 2)).Q
    g = torch.randn(3, 5)
    assert torch.allclose(steer_gradient(g, basis, 0.0), project_orth(g, basis), atol=1e-5)
    assert torch.allclose(steer_gradient(g, basis, 1.0), g, atol=1e-5)


def test_steer_empty_basis_is_identity():
    g = torch.randn(3, 5)
    empty = torch.zeros(5, 0)
    assert torch.allclose(steer_gradient(g, empty, 0.0), g, atol=1e-6)
