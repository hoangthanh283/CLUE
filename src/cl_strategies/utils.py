"""
Gradient vector utilities for GEM/A-GEM.
"""

import torch
import torch.nn as nn


def get_grad_vector(model: nn.Module) -> torch.Tensor:
    grads = []
    for pp in model.parameters():
        if pp.grad is not None:
            grads.append(pp.grad.view(-1))
        else:
            grads.append(torch.zeros_like(pp).view(-1))
    return torch.cat(grads)


def set_grad_vector(model: nn.Module, new_grads: torch.Tensor):
    pointer = 0
    for pp in model.parameters():
        numel = pp.numel()
        if pp.requires_grad:
            pp.grad = new_grads[pointer: pointer + numel].view_as(pp).clone()
        pointer += numel
