"""SimpleCIL / RanPAC baselines on synthetic features (offline, no downloads)."""

from __future__ import annotations

import torch

from doccl.methods.frozen_ptm import RanPAC, SimpleCIL, ridge_solve


class _Dummy(torch.nn.Module):
    hidden_size = 8

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Linear(1, 1)


def _separable(n_cls=4, per=30, d=8):
    centres = torch.eye(d)[:n_cls] * 5
    x = torch.cat([centres[c] + 0.3 * torch.randn(per, d) for c in range(n_cls)])
    y = torch.arange(n_cls).repeat_interleave(per)
    return x, y


def test_ridge_recovers_linear_map():
    torch.manual_seed(0)
    x = torch.randn(200, 5)
    b = torch.randn(5, 3)
    beta = ridge_solve(x.T @ x, x.T @ (x @ b), 1e-6)
    assert torch.allclose(beta, b, atol=1e-3)


def test_ranpac_and_simplecil_separate_toy_classes():
    torch.manual_seed(0)
    x, y = _separable()
    rp = RanPAC(_Dummy(), {"rp_dim": 256})
    rp.fit(x, y)
    assert (rp._predict(x).argmax(-1) == y).float().mean() == 1.0
    sc = SimpleCIL(_Dummy(), {})
    sc._absorb(x, y)
    assert (sc._predict(x).argmax(-1) == y).float().mean() == 1.0


def test_weighted_fit_runs_and_unfitted_guard():
    torch.manual_seed(0)
    x, y = _separable()
    rp = RanPAC(_Dummy(), {"rp_dim": 64})
    assert not rp._fitted()
    rp.fit(x, y, torch.rand(len(y)) + 0.5)
    assert rp._fitted() and rp.lam is not None
