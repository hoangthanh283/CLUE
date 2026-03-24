"""Tests for cl_strategies/utils.py – gradient vector helpers."""

import torch
import torch.nn as nn
import pytest

from src.cl_strategies.utils import (
    get_grad_vector,
    get_grad_vector_exclude_classifier,
    set_grad_vector,
    set_grad_vector_exclude_classifier,
)
from tests.conftest import TinyModel


def _run_backward(model: nn.Module):
    """Do a forward+backward to populate .grad tensors."""
    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    labels = torch.randint(0, 3, (batch_size, seq_len))
    outputs = model(input_ids=input_ids, labels=labels)
    outputs["loss"].backward()


# ---------------------------------------------------------------------------
# get_grad_vector
# ---------------------------------------------------------------------------


def test_get_grad_vector_length(tiny_model):
    _run_backward(tiny_model)
    g = get_grad_vector(tiny_model)
    total_params = sum(p.numel() for p in tiny_model.parameters())
    assert g.shape == (total_params,)


def test_get_grad_vector_zeros_when_no_grad():
    """Parameters without .grad should contribute zeros."""
    model = TinyModel()
    model.zero_grad()
    # Don't do backward → all grads are None
    g = get_grad_vector(model)
    assert torch.all(g == 0)


# ---------------------------------------------------------------------------
# set_grad_vector
# ---------------------------------------------------------------------------


def test_set_grad_vector_round_trip(tiny_model):
    _run_backward(tiny_model)
    original_g = get_grad_vector(tiny_model).clone()
    new_g = torch.ones_like(original_g) * 99.0
    set_grad_vector(tiny_model, new_g)
    recovered = get_grad_vector(tiny_model)
    assert torch.allclose(recovered, new_g)


def test_set_then_get_preserves_values(tiny_model):
    total = sum(p.numel() for p in tiny_model.parameters())
    g = torch.arange(total, dtype=torch.float32)
    # Need at least one backward to allocate .grad buffers
    _run_backward(tiny_model)
    set_grad_vector(tiny_model, g)
    result = get_grad_vector(tiny_model)
    assert torch.allclose(result, g)


# ---------------------------------------------------------------------------
# get_grad_vector_exclude_classifier
# ---------------------------------------------------------------------------


def test_exclude_classifier_shorter_than_full(tiny_model):
    _run_backward(tiny_model)
    g_full = get_grad_vector(tiny_model)
    g_excl = get_grad_vector_exclude_classifier(tiny_model)
    # Excluding classifier means fewer params
    assert g_excl.numel() < g_full.numel()


def test_exclude_classifier_length(tiny_model):
    _run_backward(tiny_model)
    non_cls_params = sum(
        p.numel()
        for name, p in tiny_model.named_parameters()
        if "classifier" not in name
    )
    g = get_grad_vector_exclude_classifier(tiny_model)
    assert g.shape == (non_cls_params,)


# ---------------------------------------------------------------------------
# set_grad_vector_exclude_classifier
# ---------------------------------------------------------------------------


def test_set_grad_vector_exclude_classifier(tiny_model):
    _run_backward(tiny_model)
    non_cls_params = sum(
        p.numel()
        for name, p in tiny_model.named_parameters()
        if "classifier" not in name
    )
    new_g = torch.zeros(non_cls_params)
    set_grad_vector_exclude_classifier(tiny_model, new_g)
    # Verify backbone params have zero grad
    for name, p in tiny_model.named_parameters():
        if "classifier" not in name and p.requires_grad and p.grad is not None:
            assert torch.all(p.grad == 0), f"Expected zero grad for {name}"


# ---------------------------------------------------------------------------
# get_grad_vector_exclude_classifier: zeros_like for no-grad backbone (line 38)
# ---------------------------------------------------------------------------


def test_get_grad_vector_exclude_classifier_zeros_when_no_grad():
    """When backbone params have no .grad (None), they contribute zeros (line 38)."""
    model = TinyModel()
    model.zero_grad()
    # No backward → all grads are None
    g = get_grad_vector_exclude_classifier(model)
    assert torch.all(g == 0)
