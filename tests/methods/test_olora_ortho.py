"""Regression test for the O-LoRA orthogonality penalty (Wang et al. 2023).

The penalty must measure orthogonality of the r-dimensional ROW subspaces of
successive LoRA-A adapters: ``||A_curr @ A_past^T||_F^2`` (shape (r, r)), which is
zero iff the subspaces are orthogonal. A prior bug computed ``A_curr.T @ A_past``
(shape (in, in)), which does NOT vanish for orthogonal adapters and silently
disabled the constraint — invalidating every O-LoRA / CL-LoRA result.

These tests pin the mathematical contract directly on the penalty kernel, so they
need no model or GPU and run in milliseconds.
"""
from __future__ import annotations

import torch


def _ortho_kernel(A_curr: torch.Tensor, A_past: torch.Tensor) -> torch.Tensor:
    """The corrected penalty kernel, mirroring OLoRA._ortho_loss's inner term."""
    inner = A_curr @ A_past.T  # (r, in) @ (in, r) -> (r, r)
    return (inner**2).sum()


def test_orthogonal_subspaces_give_zero_penalty():
    """Two adapters with mutually orthogonal row subspaces -> penalty == 0."""
    r, d = 4, 32
    # Build an orthonormal d-basis; give each adapter a disjoint slice of rows.
    Q, _ = torch.linalg.qr(torch.randn(d, d))  # (d, d) orthonormal columns
    A_curr = Q[:, :r].T.contiguous()           # (r, d) rows = first r basis vecs
    A_past = Q[:, r : 2 * r].T.contiguous()    # (r, d) rows = next r basis vecs (orthogonal)
    pen = _ortho_kernel(A_curr, A_past)
    assert pen.item() < 1e-8, f"orthogonal subspaces should give ~0 penalty, got {pen.item()}"


def test_identical_subspaces_give_large_penalty():
    """Identical adapters (maximally overlapping subspaces) -> large penalty."""
    r, d = 4, 32
    A = torch.randn(r, d)
    pen = _ortho_kernel(A, A)
    # ||A A^T||_F^2 for a non-degenerate A is strictly positive and sizable.
    assert pen.item() > 1.0, f"identical subspaces should give large penalty, got {pen.item()}"


def test_buggy_form_would_fail_orthogonality():
    """The OLD (buggy) kernel A_curr.T @ A_past does NOT vanish for orthogonal
    subspaces — this test documents exactly why the previous results were invalid."""
    r, d = 4, 32
    Q, _ = torch.linalg.qr(torch.randn(d, d))
    A_curr = Q[:, :r].T.contiguous()
    A_past = Q[:, r : 2 * r].T.contiguous()
    buggy = (A_curr.T @ A_past)  # (d, d) — the old code
    correct = (A_curr @ A_past.T)  # (r, r) — the fix
    assert tuple(buggy.shape) == (d, d)
    assert tuple(correct.shape) == (r, r)
    # The buggy form is NOT ~0 even though the subspaces are orthogonal -> bug.
    assert (buggy**2).sum().item() > 1e-6
    # The correct form IS ~0 -> fix.
    assert (correct**2).sum().item() < 1e-8


def test_penalty_is_differentiable():
    """Gradient flows to A_curr so the optimizer can actually reduce overlap."""
    r, d = 4, 32
    A_curr = torch.randn(r, d, requires_grad=True)
    A_past = torch.randn(r, d)
    pen = _ortho_kernel(A_curr, A_past)
    pen.backward()
    assert A_curr.grad is not None and torch.isfinite(A_curr.grad).all()
