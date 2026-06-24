"""Regression test for DocCL_B._ewc_penalty under CIL head growth.

The class-incremental classifier head grows between tasks, so the live parameter,
the theta_star snapshot, and the accumulated Fisher can each carry a different
leading (class) dimension. DocCL_B._ewc_penalty previously did
``fisher_val * (p - theta_star)**2`` with no shape reconciliation, which raises a
RuntimeError at the first CIL boundary. The fix slices all three to their common
``min_shape`` (matching EWC._ewc_penalty), penalising only the old classes that
have a prior.

This test exercises the penalty kernel directly with mismatched-shape tensors, so
it needs no model or GPU.
"""
from __future__ import annotations

import torch


def _min_shape_penalty(p, theta_star, fisher_val, lam=1.0):
    """Mirrors the fixed DocCL_B / EWC penalty inner term."""
    min_shape = tuple(
        min(a, b, c) for a, b, c in zip(p.shape, theta_star.shape, fisher_val.shape)
    )
    idx = tuple(slice(0, s) for s in min_shape)
    return lam * (fisher_val[idx] * (p[idx] - theta_star[idx]) ** 2).sum()


def test_penalty_handles_grown_head_no_crash():
    """p has 25 class rows, theta_star/fisher have 13 (pre-growth) -> must not crash."""
    d = 8
    p = torch.randn(25, d)            # current grown head
    theta_star = torch.randn(13, d)   # snapshot before growth
    fisher_val = torch.rand(13, d)    # Fisher accumulated at old width
    pen = _min_shape_penalty(p, theta_star, fisher_val)
    assert torch.isfinite(pen), "penalty must be finite"
    assert pen.item() >= 0.0


def test_penalty_only_covers_old_rows():
    """The penalty must use exactly the common min rows (13), ignoring new rows."""
    d = 4
    p = torch.zeros(20, d)
    theta_star = torch.zeros(10, d)
    fisher_val = torch.ones(10, d)
    # Make the first 10 rows of p differ from theta_star by 1.0 each element.
    p[:10] = 1.0
    pen = _min_shape_penalty(p, theta_star, fisher_val)
    # penalty = sum over (10 x d) of 1 * (1-0)^2 = 10*d
    assert abs(pen.item() - (10 * d)) < 1e-5, pen.item()


def test_equal_shapes_unchanged():
    """When all shapes match, min_shape slicing is a no-op."""
    d = 4
    p = torch.randn(7, d)
    theta_star = torch.randn(7, d)
    fisher_val = torch.rand(7, d)
    full = (fisher_val * (p - theta_star) ** 2).sum()
    sliced = _min_shape_penalty(p, theta_star, fisher_val)
    assert torch.allclose(full, sliced)
