"""Unit tests for the SD-LoRA port on synthetic tensors (fast, GPU-free).

Covers the load-bearing mechanism — the decoupled direction/magnitude forward in
:class:`SDLoraLinear` (no-op start, current-term train-vs-eval normalization asymmetry,
previous-term always normalized, freeze semantics) — plus the method's Q/V injection and
backbone-freeze wiring. The full lifecycle on a real backbone is in
``tests/methods/test_all_methods_e2e.py`` once ``sd_lora`` is registered there.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from doccl.methods.sd_lora import SDLoRA, SDLoraLinear


# ─── SDLoraLinear: the decoupled forward ─────────────────────────────────────────
def test_no_task_is_identity():
    orig = nn.Linear(4, 4)
    layer = SDLoraLinear(orig, rank=2)
    x = torch.randn(3, 4)
    assert torch.allclose(layer(x), orig(x))


def test_fresh_task_starts_as_noop():
    # B initialized to zero → a freshly added task contributes nothing (standard LoRA init).
    orig = nn.Linear(4, 4)
    layer = SDLoraLinear(orig, rank=2)
    layer.add_task()
    x = torch.randn(3, 4)
    layer.train()
    assert torch.allclose(layer(x), orig(x), atol=1e-6)
    layer.eval()
    assert torch.allclose(layer(x), orig(x), atol=1e-6)


def test_current_task_unnormalized_in_train_normalized_in_eval():
    orig = nn.Linear(4, 4)
    layer = SDLoraLinear(orig, rank=2)
    layer.add_task()
    with torch.no_grad():
        layer.B_banks[-1].copy_(torch.randn(4, 2))  # make the direction non-trivial
        layer.alphas[-1].fill_(1.0)
    x = torch.randn(3, 4)
    a, b = layer.A_banks[-1], layer.B_banks[-1]
    delta = (x @ a.t()) @ b.t()
    norm = b.norm() * a.norm()

    layer.train()
    assert torch.allclose(layer(x), orig(x) + delta, atol=1e-5)  # un-normalized in train
    layer.eval()
    assert torch.allclose(layer(x), orig(x) + delta / norm, atol=1e-5)  # normalized in eval


def test_previous_task_always_normalized():
    # Task 0 (frozen, non-trivial) + task 1 (current, no-op B=0). The current no-op means
    # train and eval agree, and both equal orig + the *normalized* task-0 contribution.
    orig = nn.Linear(4, 4)
    layer = SDLoraLinear(orig, rank=2)
    layer.add_task()
    with torch.no_grad():
        layer.B_banks[0].copy_(torch.randn(4, 2))
        layer.alphas[0].fill_(0.7)
    layer.freeze_current_direction()
    layer.add_task()  # task 1, B=0 → current no-op
    x = torch.randn(3, 4)
    a0, b0 = layer.A_banks[0], layer.B_banks[0]
    expected = orig(x) + 0.7 * ((x @ a0.t()) @ b0.t()) / (b0.norm() * a0.norm())

    layer.train()
    assert torch.allclose(layer(x), expected, atol=1e-5)
    layer.eval()
    assert torch.allclose(layer(x), expected, atol=1e-5)


def test_freeze_current_direction_keeps_scalar_trainable():
    layer = SDLoraLinear(nn.Linear(4, 4), rank=2)
    layer.add_task()
    assert layer.A_banks[-1].requires_grad and layer.B_banks[-1].requires_grad
    layer.freeze_current_direction()
    assert not layer.A_banks[-1].requires_grad and not layer.B_banks[-1].requires_grad
    assert layer.alphas[-1].requires_grad  # magnitudes stay trainable every task


def test_add_task_grows_banks():
    layer = SDLoraLinear(nn.Linear(4, 4), rank=2)
    assert layer.n_tasks == 0
    layer.add_task()
    layer.add_task()
    assert layer.n_tasks == 2 == len(layer.A_banks) == len(layer.B_banks) == len(layer.alphas)


# ─── SDLoRA method: injection + freeze wiring ────────────────────────────────────
class _Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(4, 4)
        self.key = nn.Linear(4, 4)  # must NOT be wrapped (SD-LoRA targets Q,V only)
        self.value = nn.Linear(4, 4)


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = _Attn()
        self.classifier = nn.Linear(4, 3)


class _FakeWrapper:
    def __init__(self) -> None:
        self.model = _Inner()


def _make_method() -> SDLoRA:
    m = SDLoRA.__new__(SDLoRA)
    m.rank = 2
    m.alpha_init = 0.8
    m.target_modules = {"query", "value"}
    m.sd_layers = []
    m.model = _FakeWrapper()
    m.device = torch.device("cpu")
    return m


def test_injection_wraps_only_q_and_v():
    m = _make_method()
    m._inject()
    inner = m.model.model
    assert isinstance(inner.attention.query, SDLoraLinear)
    assert isinstance(inner.attention.value, SDLoraLinear)
    assert isinstance(inner.attention.key, nn.Linear)  # key untouched
    assert len(m.sd_layers) == 2


def test_freeze_keeps_only_head_trainable():
    m = _make_method()
    m._inject()
    m._freeze_backbone_keep_head()
    trainable = {n for n, p in m.model.model.named_parameters() if p.requires_grad}
    assert all("classifier" in n for n in trainable)
    assert any("classifier" in n for n in trainable)  # head IS trainable


def test_before_task_adds_one_direction_per_layer():
    m = _make_method()
    m._inject()
    m._freeze_backbone_keep_head()
    m.before_task(SimpleNamespace(task_id=0), None)
    for layer in m.sd_layers:
        assert layer.n_tasks == 1
        assert layer.A_banks[-1].requires_grad  # current direction trainable
    m.before_task(SimpleNamespace(task_id=1), None)
    for layer in m.sd_layers:
        assert layer.n_tasks == 2
        assert not layer.A_banks[0].requires_grad  # prior direction now frozen
        assert layer.A_banks[1].requires_grad
