"""Unit tests for FisherMaskFreeze internals (synthetic tensors, fast, GPU-free).

Covers: exact per-tensor top-p mask fraction (incl. the all-zero-Fisher tie case),
gradient masking, byte-exact frozen entries across an AdamW step with nonzero
weight decay (the decoupled decay would otherwise shrink zero-grad entries), the
CIL head-growth slice path, and Fisher accumulation with shape padding.
Full lifecycle on real LayoutLMv3 is in tests/methods/test_all_methods_e2e.py.
"""

from __future__ import annotations

import torch
from torch import nn

from doccl.methods.fisher_mask import FisherMaskFreeze


class _Model(nn.Module):
    def __init__(self, d=10, nl=4):
        super().__init__()
        self.enc = nn.Linear(d, d)
        self.classifier = nn.Linear(d, nl)


def _method(p=0.5, model=None):
    torch.manual_seed(0)
    return FisherMaskFreeze(model or _Model(), {"mask_top_p": p, "fisher_n_samples": 2})


def _fake_fisher(m, magnitude=None):
    return {
        name: (torch.rand_like(prm) if magnitude is None else torch.full_like(prm, magnitude))
        for name, prm in m.model.named_parameters()
    }


def test_mask_fraction_exact_per_tensor():
    m = _method(p=0.5)
    m.fisher_acc = {k: v.cpu() for k, v in _fake_fisher(m).items()}
    m._rebuild_masks()
    for name, mask in m.masks.items():
        frac = mask.float().mean().item()
        k = round(0.5 * mask.numel())
        assert abs(frac - k / mask.numel()) < 1e-6, name


def test_all_zero_fisher_does_not_overfreeze():
    m = _method(p=0.5)
    m.fisher_acc = {k: torch.zeros_like(v).cpu() for k, v in _fake_fisher(m).items()}
    m._rebuild_masks()
    for mask in m.masks.values():
        k = round(0.5 * mask.numel())
        assert int(mask.sum()) == k  # exactly k, never "everything ties at zero"


def test_frozen_entries_byte_exact_through_adamw_step():
    m = _method(p=0.5)
    m.fisher_acc = {k: v.cpu() for k, v in _fake_fisher(m).items()}
    m._rebuild_masks()
    model = m.model
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    x = torch.randn(8, 10)
    loss = model.classifier(model.enc(x)).pow(2).mean()
    loss.backward()
    m._mask_grads()
    saved = m._snapshot_frozen()
    opt.step()
    m._restore_frozen(saved)
    for name, p in model.named_parameters():
        mask = m.masks[name].cpu()
        assert torch.equal(p.data[mask], before[name][mask]), f"{name}: frozen entries moved"
        assert not torch.allclose(p.data[~mask], before[name][~mask]), f"{name}: nothing trained"


def test_head_growth_masks_only_old_block():
    m = _method(p=0.99)
    m.fisher_acc = {
        k: torch.ones_like(v).cpu() + torch.rand_like(v).cpu() for k, v in _fake_fisher(m).items()
    }
    m._rebuild_masks()
    # Simulate CIL head growth: 4 -> 6 labels.
    old = m.model.classifier
    m.model.classifier = nn.Linear(10, 6)
    with torch.no_grad():
        m.model.classifier.weight[:4] = old.weight
        m.model.classifier.bias[:4] = old.bias
    x = torch.randn(4, 10)
    m.model.classifier(m.model.enc(x)).pow(2).mean().backward()
    grads_before = {n: p.grad.clone() for n, p in m.model.named_parameters()}
    m._mask_grads()
    w_grad = m.model.classifier.weight.grad
    assert torch.all(w_grad[4:] == grads_before["classifier.weight"][4:])  # new rows untouched
    old_mask = m.masks["classifier.weight"]
    assert torch.all(w_grad[:4][old_mask] == 0)  # old block masked


def test_after_task_accumulates_and_pads_grown_head(monkeypatch):
    m = _method(p=0.5)
    fake = {"canned": {"classifier.weight": torch.ones(4, 10)}}
    monkeypatch.setattr(
        "doccl.methods.fisher_mask.empirical_fisher_diagonal",
        lambda *a, **k: {n: t.clone() for n, t in fake["canned"].items()},
    )
    task = type("T", (), {"task_id": 0})()
    m.after_task(task, train_loader=None)
    assert torch.all(m.fisher_acc["classifier.weight"] == 1)
    # Head grows 4 -> 6 between tasks; the old accumulator must zero-pad.
    fake["canned"] = {"classifier.weight": torch.ones(6, 10)}
    m.after_task(task, train_loader=None)
    acc = m.fisher_acc["classifier.weight"]
    assert acc.shape == (6, 10)
    assert torch.all(acc[:4] == 2) and torch.all(acc[4:] == 1)
    assert m.masks["classifier.weight"].shape == (6, 10)
