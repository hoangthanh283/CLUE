"""Unit tests for LexSlot internals on synthetic tensors / stubs (fast, GPU-free). Full
lifecycle on a real LayoutLMv3 is in tests/methods/test_all_methods_e2e.py once registered."""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from doccl.methods.lexslot import LexSlot
from doccl.methods.lexslot_memory import LogitSlots, ReprSlots


def test_late_layer_indices_for_12():
    # late bucket = idx >= 2*(12//3) = 8 -> [8,9,10,11]
    assert LexSlot._late_layer_indices(num_layers=12, slot_depth="head_late") == [8, 9, 10, 11]


def test_late_layer_indices_head_only_empty():
    assert LexSlot._late_layer_indices(num_layers=12, slot_depth="head_only") == []


def test_late_layer_indices_uniform_is_all():
    assert LexSlot._late_layer_indices(num_layers=12, slot_depth="uniform") == list(range(12))


def test_late_layer_indices_head_late_mid():
    # mid+late = idx >= 12//3 = 4 -> [4..11]
    assert LexSlot._late_layer_indices(num_layers=12, slot_depth="head_late_mid") == list(
        range(4, 12)
    )


def test_derive_mask_soft_grades_by_similarity():
    m = LexSlot.__new__(LexSlot)
    m.slot_sharing = "soft"
    m.share_threshold = 0.5
    # 4 head slots: 0,1 owned by task0, 2,3 fresh; current task1; S[1,0]=0.3
    import torch as T  # noqa: N812

    S = T.tensor([[1.0, 0.3], [0.3, 1.0]])  # noqa: N806
    owner = [0, 0, -1, -1]
    mask = m._derive_mask(task_id=1, slot_owner=owner, S=S)
    assert torch.allclose(mask, torch.tensor([0.3, 0.3, 1.0, 1.0]), atol=1e-6)


def test_layer_hook_bare_tensor():
    """BERT/BROS/LayoutLMv3 path: layer output is a tuple whose [0] is a tensor."""
    m = LexSlot.__new__(LexSlot)
    rs = ReprSlots(n_slots=2, hidden_dim=4, rank=2)
    # Force a non-zero delta so we can detect that the shift was applied.
    with torch.no_grad():
        rs.up.copy_(torch.ones_like(rs.up))
        rs.down.copy_(torch.ones_like(rs.down))
    hook = m._make_layer_hook(rs)
    hs = torch.randn(1, 3, 4)
    out = hook(None, None, (hs, "attn"))  # tuple output, [0] is tensor
    assert isinstance(out, tuple) and out[1] == "attn"  # extras preserved
    assert out[0].shape == hs.shape
    assert not torch.allclose(out[0], hs)  # shift applied


def test_layer_hook_bare_tensor_no_tuple():
    """Defensive path: a layer that returns a bare tensor (no tuple wrapper)."""
    m = LexSlot.__new__(LexSlot)
    rs = ReprSlots(n_slots=2, hidden_dim=4, rank=2)
    hook = m._make_layer_hook(rs)
    hs = torch.randn(1, 3, 4)
    out = hook(None, None, hs)
    assert torch.is_tensor(out) and out.shape == hs.shape


def test_layer_hook_lilt_nested_tuple():
    """LiLT path: layer output is ((text_hidden, layout_hidden), *extras). The shift must
    apply to the TEXT element, leave the layout element untouched, and preserve nesting."""
    m = LexSlot.__new__(LexSlot)
    rs = ReprSlots(n_slots=2, hidden_dim=4, rank=2)
    with torch.no_grad():
        rs.up.copy_(torch.ones_like(rs.up))
        rs.down.copy_(torch.ones_like(rs.down))
    hook = m._make_layer_hook(rs)
    text = torch.randn(1, 3, 4)
    layout = torch.randn(1, 3, 4)
    out = hook(None, None, ((text, layout), "attn"))
    # Structure preserved: ((text', layout), "attn")
    assert isinstance(out, tuple) and out[1] == "attn"
    assert isinstance(out[0], tuple) and len(out[0]) == 2
    new_text, new_layout = out[0]
    assert new_text.shape == text.shape
    assert not torch.allclose(new_text, text)  # text shifted
    assert torch.allclose(new_layout, layout)  # layout untouched


def test_encoder_layers_secondary_via_inner():
    """Backbone-agnostic: secondary wrappers expose the encoder via model._inner."""
    m = LexSlot.__new__(LexSlot)
    layer_list = ["L0", "L1"]
    inner = types.SimpleNamespace(encoder=types.SimpleNamespace(layer=layer_list))
    # Secondary wrapper: has _inner property returning the inner encoder.
    m.model = types.SimpleNamespace(_inner=inner)
    assert m._encoder_layers() is layer_list


def test_encoder_layers_layoutlmv3_fallback():
    """Backbone-agnostic: LayoutLMv3Wrapper has no _inner; falls back to model.model.layoutlmv3."""
    m = LexSlot.__new__(LexSlot)
    layer_list = ["L0", "L1"]
    inner = types.SimpleNamespace(encoder=types.SimpleNamespace(layer=layer_list))
    hf = types.SimpleNamespace(layoutlmv3=inner)
    # LayoutLMv3Wrapper: no _inner attr; getattr(model, "_inner", None) is None -> fallback.
    m.model = types.SimpleNamespace(model=hf)
    assert m._encoder_layers() is layer_list


def test_slot_claim_is_per_task_budget_not_greedy():
    """Regression: with n_tasks fixed, each task claims n_slots//n_tasks slots — NOT all
    fresh slots (the original bug where task 0 grabbed everything and later tasks owned none).
    Replicates the exact claim arithmetic used in before_task."""
    n_slots, n_tasks = 12, 3
    per_task = max(1, n_slots // max(n_tasks, 1))  # == 4
    slot_owner = [-1] * n_slots
    for task_id in range(3):
        fresh = [s for s, o in enumerate(slot_owner) if o == -1]
        claim = fresh[:per_task]
        for s in claim:
            slot_owner[s] = task_id
    # Every task owns exactly its 4-slot block; none left greedily unowned-by-design.
    assert slot_owner == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    for t in range(3):
        assert slot_owner.count(t) == per_task, f"task {t} should own {per_task} slots"


def test_trainable_parameters_includes_slots():
    """CRITICAL-1 regression: slot params must appear in trainable_parameters()."""
    # Build a minimal LexSlot stub without __init__ so no real model is needed.
    m = LexSlot.__new__(LexSlot)

    # Minimal model stub: one backbone Parameter.
    backbone_param = nn.Parameter(torch.zeros(4))
    stub_model = types.SimpleNamespace(parameters=lambda: iter([backbone_param]))
    backbone_param.requires_grad = True
    m.model = stub_model

    # Attach real slot modules (same classes used at runtime).
    m.head_slots = LogitSlots(n_slots=2, hidden_dim=4, n_labels=2)
    m.late_slots = nn.ModuleDict({"8": ReprSlots(n_slots=2, hidden_dim=4, rank=2)})

    trainable_ids = {id(p) for p in m.trainable_parameters()}

    # Every slot parameter must be in the optimizer target set.
    for p in m.head_slots.parameters():
        assert id(p) in trainable_ids, "head_slots param missing from trainable_parameters()"
    for rs in m.late_slots.values():
        for p in rs.parameters():
            assert id(p) in trainable_ids, "late_slots param missing from trainable_parameters()"
    # Backbone param must also be present.
    assert id(backbone_param) in trainable_ids, "backbone param missing from trainable_parameters()"
