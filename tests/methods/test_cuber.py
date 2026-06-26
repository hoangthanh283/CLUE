"""Unit tests for CUBER internals on synthetic tensors (fast, GPU-free)."""

from __future__ import annotations

import torch

from doccl.methods.cuber import CUBER


def test_layer_basis_steers_per_layer():
    """_steer_for_layer steers a layer's grad against ITS stored subspace only."""
    m = CUBER.__new__(CUBER)
    m.transfer_alpha = 0.0
    m._steer_enabled = True
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(5, 2)).Q
    m._layer_subspaces = {"layerA": [basis]}
    g = torch.randn(4, 5)
    out = m._steer_for_layer("layerA", g)
    assert torch.allclose(out @ basis, torch.zeros(4, 2), atol=1e-5)  # protected
    # a layer with no stored subspace is unchanged
    assert torch.allclose(m._steer_for_layer("unknown", g), g, atol=1e-6)


def test_steer_disabled_is_identity():
    m = CUBER.__new__(CUBER)
    m.transfer_alpha = 0.0
    m._steer_enabled = False
    m._layer_subspaces = {"layerA": [torch.linalg.qr(torch.randn(5, 2)).Q]}
    g = torch.randn(4, 5)
    assert torch.allclose(m._steer_for_layer("layerA", g), g, atol=1e-6)
