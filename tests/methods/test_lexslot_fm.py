"""Unit tests for LexSlot-FM internals (synthetic tensors, fast, GPU-free)."""

from __future__ import annotations

import torch

from doccl.methods.lexslot_fm import FunctionalHeadSlots


def test_fm_fullrank_logits_all():
    d, C = 8, 5
    fm = FunctionalHeadSlots(d, C, max_tasks=3)
    # Two owned tasks
    fm.claim(0)
    fm.claim(1)
    with torch.no_grad():
        fm.weight[0].copy_(torch.randn(d, C) * 0.1)
        fm.bias[0].copy_(torch.randn(C) * 0.1)
        fm.weight[1].copy_(torch.randn(d, C) * 0.1)
        fm.bias[1].copy_(torch.randn(C) * 0.1)
    feats = torch.randn(2, 4, d)
    out = fm.logits_all(feats)
    assert out.shape == (2, 4, 2, C), f"expected (2,4,2,{C}) got {out.shape}"
    # Manual check task 0
    expected_0 = feats @ fm.weight[0] + fm.bias[0]
    assert torch.allclose(out[:, :, 0, :], expected_0, atol=1e-6)


def test_fm_fullrank_logits_one():
    d, C = 8, 5
    fm = FunctionalHeadSlots(d, C, max_tasks=3)
    idx = fm.claim(0)
    with torch.no_grad():
        fm.weight[idx].copy_(torch.randn(d, C) * 0.1)
        fm.bias[idx].copy_(torch.randn(C) * 0.1)
    feats = torch.randn(2, 4, d)
    out = fm.logits_one(feats, idx)
    expected = feats @ fm.weight[idx] + fm.bias[idx]
    assert torch.allclose(out, expected, atol=1e-6)


def test_fm_fullrank_freeze_is_immutable():
    d, C = 8, 5
    fm = FunctionalHeadSlots(d, C, max_tasks=3)
    fm.claim(0)
    fm.claim(1)
    fm.freeze_owned()
    for i in [0, 1]:
        assert fm.weight[i].requires_grad is False
        assert fm.bias[i].requires_grad is False
        assert fm.frozen_tasks[i] is True


def test_fm_fullrank_expand_labels_preserves_owned():
    d, C_old, C_new = 8, 5, 10
    fm = FunctionalHeadSlots(d, C_old, max_tasks=3)
    idx = fm.claim(0)
    with torch.no_grad():
        fm.weight[idx].copy_(torch.randn(d, C_old) * 0.1)
        fm.bias[idx].copy_(torch.randn(C_old) * 0.1)
    old_w = fm.weight[idx].data.clone()
    old_b = fm.bias[idx].data.clone()
    fm.expand_labels(C_new)
    assert fm.n_labels == C_new
    assert fm.weight[idx].shape == (d, C_new)
    assert fm.bias[idx].shape == (C_new,)
    assert torch.allclose(fm.weight[idx].data[:, :C_old], old_w, atol=1e-6)
    assert torch.allclose(fm.bias[idx].data[:C_old], old_b, atol=1e-6)
    assert torch.all(fm.weight[idx].data[:, C_old:] == 0)
    assert torch.all(fm.bias[idx].data[C_old:] == 0)


def test_fm_fullrank_claim_zero_init():
    d, C = 8, 5
    fm = FunctionalHeadSlots(d, C, max_tasks=3)
    idx = fm.claim(0)
    assert torch.all(fm.weight[idx] == 0)
    assert torch.all(fm.bias[idx] == 0)
    # logits_one should return all zeros
    feats = torch.randn(2, 4, d)
    out = fm.logits_one(feats, idx)
    assert torch.all(out == 0)


def test_fm_fullrank_no_owned_returns_zero():
    d, C = 8, 5
    fm = FunctionalHeadSlots(d, C, max_tasks=3)
    feats = torch.randn(2, 4, d)
    out = fm.logits_all(feats)
    assert out.shape == (2, 4, 0, C)


def test_fm_fullrank_untrained_claim_produces_no_effect():
    d, C = 8, 5
    fm = FunctionalHeadSlots(d, C, max_tasks=3)
    idx = fm.claim(0)
    assert fm.owner[idx] == 0
    assert fm.frozen_tasks[idx] is False
    feats = torch.randn(2, 4, d)
    out = fm.logits_all(feats)
    assert out.shape == (2, 4, 1, C)
    assert torch.all(out == 0)
