# doccl/methods/grad_subspace.py
"""Gradient-subspace numerics for head-localized gradient transfer (HGT) and CUBER.

A task's "subspace" is the top-k eigenvectors of its uncentered feature second moment
M = Σ f fᵀ (f = the features feeding a linear layer over scored tokens). A weight W's
gradient G (out×in) acts on a stored input-subspace U (in×k) through G U. Removing that
component (``project_orth``) protects the stored task's outputs (GPM/Adam-NSCL); keeping a
fraction α of it (``steer_gradient``) lets new-task updates flow into the stored subspace
— the transfer knob whose α=0 vs α>0 contrast is the method's headline ablation.
"""

from __future__ import annotations

import torch


def feature_moment(feats: torch.Tensor) -> torch.Tensor:
    """Uncentered second moment featsᵀfeats. feats (N, d) -> (d, d)."""
    return feats.T @ feats


def top_eigvecs(moment: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k eigenvectors (largest eigenvalue first) of a symmetric matrix. (d, d) -> (d, k)."""
    # eigh returns ascending eigenvalues; take the last k, reverse to descending.
    evals, evecs = torch.linalg.eigh(moment.double())
    k = min(k, evecs.shape[1])
    idx = torch.arange(evecs.shape[1] - 1, evecs.shape[1] - 1 - k, -1, device=evecs.device)
    return evecs.index_select(1, idx).to(moment.dtype)


def project_orth(grad: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """grad (C, d) minus its component acting on basis (d, m) orthonormal cols -> (C, d)."""
    if basis.numel() == 0 or basis.shape[1] == 0:
        return grad
    return grad - (grad @ basis) @ basis.T


def steer_gradient(grad: torch.Tensor, basis: torch.Tensor, alpha: float) -> torch.Tensor:
    """project_orth + alpha * in-subspace component. alpha=0 -> protection; 1 -> identity."""
    if basis.numel() == 0 or basis.shape[1] == 0:
        return grad
    in_sub = (grad @ basis) @ basis.T
    return (grad - in_sub) + alpha * in_sub
