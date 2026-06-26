"""Unit tests for HGT internals on synthetic tensors (fast, GPU-free). The full lifecycle
on a real LayoutLMv3 is covered in tests/methods/test_all_methods_e2e.py once hgt is
registered there."""

from __future__ import annotations

import torch

from doccl.methods.hgt import HGT


def test_stacked_basis_orthonormal_columns():
    """_stacked_basis concatenates per-task subspaces and re-orthonormalises -> Iᵏ."""
    m = HGT.__new__(HGT)
    m.hidden_dim = 6
    torch.manual_seed(0)
    u0 = torch.linalg.qr(torch.randn(6, 2)).Q
    u1 = torch.linalg.qr(torch.randn(6, 2)).Q
    m._task_subspaces = [u0, u1]
    basis = m._stacked_basis()
    assert basis.shape[0] == 6 and basis.shape[1] <= 4
    assert torch.allclose(basis.T @ basis, torch.eye(basis.shape[1]), atol=1e-4)


def test_stacked_basis_empty_when_no_tasks():
    m = HGT.__new__(HGT)
    m.hidden_dim = 6
    m._task_subspaces = []
    assert m._stacked_basis().shape == (6, 0)


def test_steer_hook_alpha0_protects_alpha1_passes():
    """The registered head-grad hook with alpha=0 zeroes the in-subspace gradient; alpha=1
    leaves it unchanged."""
    m = HGT.__new__(HGT)
    m.hidden_dim = 5
    m.transfer_alpha = 0.0
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(5, 2)).Q
    m._task_subspaces = [basis]
    m._steer_enabled = True
    g = torch.randn(3, 5)
    out0 = m._steer_head_grad(g)
    assert torch.allclose(out0 @ basis, torch.zeros(3, 2), atol=1e-5)  # protected
    m.transfer_alpha = 1.0
    assert torch.allclose(m._steer_head_grad(g), g, atol=1e-5)  # full pass


def test_steer_hook_noop_when_disabled():
    m = HGT.__new__(HGT)
    m.hidden_dim = 5
    m.transfer_alpha = 0.0
    m._task_subspaces = [torch.linalg.qr(torch.randn(5, 2)).Q]
    m._steer_enabled = False
    g = torch.randn(3, 5)
    assert torch.allclose(m._steer_head_grad(g), g, atol=1e-6)
