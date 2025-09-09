"""
Gradient vector utilities for GEM/A-GEM.
"""

import torch
import torch.nn as nn


def get_grad_vector(model: nn.Module) -> torch.Tensor:
    """Flatten model gradients into a single CPU tensor.

    Collects per-parameter gradients and moves them to CPU as they are read
    to avoid duplicating a large contiguous buffer on GPU. For parameters with
    no gradient, inserts a CPU zero vector of matching shape to preserve layout.
    """
    grads = []
    for pp in model.parameters():
        if pp.grad is not None:
            grads.append(pp.grad.detach().view(-1).cpu())
        else:
            grads.append(torch.zeros(pp.numel(), dtype=pp.dtype, device="cpu"))
    return torch.cat(grads)


def set_grad_vector(model: nn.Module, new_grads: torch.Tensor):
    """Load a flattened gradient vector into model parameters.

    Accepts a 1-D tensor on CPU or GPU. Slices are moved to each parameter's
    device to avoid device mismatch issues.
    """
    if new_grads.device.type != "cpu" and new_grads.is_sparse:
        # Ensure dense for slicing below
        new_grads = new_grads.to_dense()

    pointer = 0
    for pp in model.parameters():
        numel = pp.numel()
        if pp.requires_grad:
            slice_view = new_grads[pointer: pointer + numel].view_as(pp)
            # Ensure grad is on the same device as the parameter
            grad_tensor = slice_view.to(pp.device)
            # Assign a fresh tensor to avoid inadvertent views
            pp.grad = grad_tensor.clone()
        pointer += numel
