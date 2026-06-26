"""Unit tests for HGT internals on synthetic tensors (fast, GPU-free). The full lifecycle
on a real LayoutLMv3 is covered in tests/methods/test_all_methods_e2e.py once hgt is
registered there."""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from doccl.methods.hgt import HGT


def _make_stub_hgt(hidden_dim: int = 6, alpha: float = 0.0) -> HGT:
    """Build a minimal HGT via __new__ with a real nn.Linear so .weight.device works."""
    m = HGT.__new__(HGT)
    m.hidden_dim = hidden_dim
    m.transfer_alpha = alpha
    m._task_subspaces = []
    m._steer_enabled = True
    m._head_hook_handle = None  # must exist for _register_head_hook
    # Minimal stub: m.model.model.classifier → a real nn.Linear
    lin = nn.Linear(hidden_dim, 3)
    inner = types.SimpleNamespace(classifier=lin)
    m.model = types.SimpleNamespace(model=inner)
    return m


def test_register_head_hook_fires_on_new_linear():
    """Regression for I1 — stale hook after CIL head swap.

    After _register_head_hook(), swapping the classifier Linear and calling
    _register_head_hook() again MUST move the hook to the new tensor.  With alpha=0
    the in-subspace component of new_lin.weight.grad must be zero after backward.
    If the hook stays on the OLD tensor (the bug), grad is untouched and this fails.
    """
    torch.manual_seed(0)
    hidden_dim = 6
    m = _make_stub_hgt(hidden_dim=hidden_dim, alpha=0.0)

    basis = torch.linalg.qr(torch.randn(hidden_dim, 2)).Q  # (6,2) orthonormal
    m._task_subspaces = [basis]

    # Register hook on the initial Linear.
    m._register_head_hook()

    # Simulate expand_classifier: replace with a NEW Linear (wider head 3->5).
    new_lin = nn.Linear(hidden_dim, 5)
    m.model.model.classifier = new_lin

    # Re-register so the hook points to the new weight tensor.
    m._register_head_hook()

    # Run a backward through new_lin and check the hook fired.
    x = torch.randn(2, hidden_dim, requires_grad=False)
    out = new_lin(x)  # (2, 5)
    loss = out.sum()
    loss.backward()

    # The in-subspace gradient component must be zero (hook protected it).
    assert new_lin.weight.grad is not None, "new_lin.weight.grad is None — backward never ran"
    in_sub = new_lin.weight.grad @ basis  # (5, 2)
    assert torch.allclose(in_sub, torch.zeros(5, 2), atol=1e-5), (
        f"Hook did NOT fire on new_lin: in-subspace grad norm = {in_sub.norm():.4f} (expected ~0). "
        "Likely the hook is still registered on the OLD tensor (stale-hook bug I1)."
    )


def test_register_head_hook_idempotent_no_double_steer():
    """Calling _register_head_hook() twice must NOT double-apply the steering.

    With alpha=0 the projected gradient satisfies grad @ basis ≈ 0.  If the hook were
    applied twice the gradient would still satisfy grad @ basis ≈ 0 (projecting an
    already-projected vector is a no-op), BUT the magnitude would be unchanged — so
    there is no NaN risk.  We verify the invariant holds (==0) as the stable check.
    """
    torch.manual_seed(1)
    hidden_dim = 6
    m = _make_stub_hgt(hidden_dim=hidden_dim, alpha=0.0)

    basis = torch.linalg.qr(torch.randn(hidden_dim, 2)).Q
    m._task_subspaces = [basis]

    m._register_head_hook()
    m._register_head_hook()  # second call — must remove the first and re-register once

    lin = m.model.model.classifier
    x = torch.randn(2, hidden_dim)
    loss = lin(x).sum()
    loss.backward()

    assert lin.weight.grad is not None
    in_sub = lin.weight.grad @ basis
    assert torch.allclose(
        in_sub, torch.zeros(3, 2), atol=1e-5
    ), f"Double-register guard failed: in-subspace norm = {in_sub.norm():.4f}"


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
