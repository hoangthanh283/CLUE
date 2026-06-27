from __future__ import annotations

import torch

from doccl.methods.lexslot_memory import LogitSlots, ReprSlots


def test_logit_slots_delta_shape_and_zero_init():
    ls = LogitSlots(n_slots=4, hidden_dim=8, n_labels=5)
    feats = torch.randn(2, 6, 8)
    d = ls.logits_delta(feats)
    assert d.shape == (2, 6, 5)
    # zero-init proj -> the delta is an EXACT no-op at init (isolated slots inject no noise).
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-7)


def test_repr_slots_zero_init_is_noop():
    rs = ReprSlots(n_slots=4, hidden_dim=8, rank=2)
    h = torch.randn(2, 6, 8)
    d = rs.repr_delta(h)
    # zero-init up -> the shift is an EXACT no-op at init.
    assert torch.allclose(d, torch.zeros_like(d), atol=1e-7)


def test_logit_slots_expand_labels_preserves():
    ls = LogitSlots(n_slots=3, hidden_dim=8, n_labels=5)
    with torch.no_grad():
        ls.proj.copy_(torch.arange(3 * 5, dtype=torch.float).reshape(3, 5))
    ls.expand_labels(7)
    assert ls.proj.shape == (3, 7)
    assert torch.allclose(ls.proj[:, :5], torch.arange(3 * 5, dtype=torch.float).reshape(3, 5))


def test_logit_slots_grad_mask_zeros_isolated_slot():
    ls = LogitSlots(n_slots=3, hidden_dim=4, n_labels=2)
    # proj is zero-init (no-op); set it nonzero so the bilinear form has gradient flow,
    # making the grad-mask assertions meaningful (the mask, not the init, is under test).
    with torch.no_grad():
        ls.proj.copy_(torch.randn(3, 2) * 0.1)
    ls.set_grad_mask(torch.tensor([1.0, 0.0, 1.0]))  # slot 1 frozen
    feats = torch.randn(1, 3, 4)
    loss = ls.logits_delta(feats).pow(2).sum()
    loss.backward()
    assert torch.allclose(ls.values.grad[1], torch.zeros(4), atol=1e-6)  # masked
    assert ls.values.grad[0].abs().sum() > 0  # unmasked trains
    # proj hook: masked slot row must be zeroed, unmasked slot row must be non-zero
    assert torch.allclose(ls.proj.grad[1], torch.zeros(2), atol=1e-6)  # masked
    assert ls.proj.grad[0].abs().sum() > 0  # unmasked trains


def test_repr_slots_delta_shape():
    rs = ReprSlots(n_slots=4, hidden_dim=8, rank=2)
    h = torch.randn(2, 6, 8)
    d = rs.repr_delta(h)
    assert d.shape == (2, 6, 8)


def test_repr_slots_grad_mask_zeros_isolated_slot():
    rs = ReprSlots(n_slots=3, hidden_dim=4, rank=2)
    # up is zero-init (no-op); set it nonzero so repr_delta depends on BOTH down and up,
    # making the grad-mask assertions meaningful (the mask, not the init, is under test).
    with torch.no_grad():
        rs.up.copy_(torch.randn(3, 2, 4) * 0.1)
    rs.set_grad_mask(torch.tensor([0.0, 1.0, 1.0]))  # slot 0 frozen
    h = torch.randn(1, 3, 4)
    loss = rs.repr_delta(h).pow(2).sum()
    loss.backward()
    assert torch.allclose(rs.down.grad[0], torch.zeros(4, 2), atol=1e-6)
    assert rs.down.grad[1].abs().sum() > 0
    # up hook: masked slot tensor must be zeroed, unmasked slot tensor must be non-zero
    assert torch.allclose(rs.up.grad[0], torch.zeros(2, 4), atol=1e-6)  # masked
    assert rs.up.grad[1].abs().sum() > 0  # unmasked trains


def test_set_owner():
    ls = LogitSlots(n_slots=4, hidden_dim=4, n_labels=2)
    ls.set_owner([0, 1], task_id=3)
    assert ls.slot_owner[0] == 3 and ls.slot_owner[1] == 3 and ls.slot_owner[2] == -1
