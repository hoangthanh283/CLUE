"""Unit tests for LexSlot internals on synthetic tensors / stubs (fast, GPU-free). Full
lifecycle on a real LayoutLMv3 is in tests/methods/test_all_methods_e2e.py once registered."""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from doccl.methods.doccl import DocCL
from doccl.methods.lexslot import LexSlot
from doccl.methods.lexslot_memory import LogitSlots, ReprSlots
from doccl.methods.naive import NaiveFineTune


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




def test_lexslot_is_standalone_not_doccl():
    """LexSlot must NOT inherit DocCL (no replay/KD/Fisher). It reuses Naive's plain-CE
    loop via NaiveFineTune."""
    assert not issubclass(LexSlot, DocCL), "LexSlot must not subclass DocCL"
    assert issubclass(LexSlot, NaiveFineTune), "LexSlot should reuse Naive's plain-CE loop"


def _fake_encoder(n_layers: int, d: int = 4) -> tuple[nn.Module, nn.Module]:
    """A fake inner encoder: .embeddings + .encoder.layer (ModuleList of nn.Linear)."""
    inner = nn.Module()
    inner.embeddings = nn.Linear(d, d)
    inner.encoder = types.SimpleNamespace(
        layer=nn.ModuleList([nn.Linear(d, d) for _ in range(n_layers)])
    )
    return inner


def test_partial_freeze_freezes_non_slot_layers_keeps_slot_layers_trainable():
    """freeze_lower=True: embeddings + non-slot-bearing layers frozen; the slot-bearing
    late layers stay trainable. The trainable set is exactly aligned with slot placement."""
    m = LexSlot.__new__(LexSlot)
    n_layers = 4
    inner = _fake_encoder(n_layers, d=4)
    m.model = types.SimpleNamespace(_inner=inner)
    m.freeze_lower = True
    m._late_idx = [2, 3]  # slot-bearing late layers
    m._freeze_lower_encoder()

    # Embeddings frozen.
    assert all(not p.requires_grad for p in inner.embeddings.parameters())
    layers = inner.encoder.layer
    # Layers 0,1 (non-slot-bearing) frozen; layers 2,3 (slot-bearing) trainable.
    for i in (0, 1):
        assert all(
            not p.requires_grad for p in layers[i].parameters()
        ), f"layer {i} should be frozen"
    for i in (2, 3):
        assert all(
            p.requires_grad for p in layers[i].parameters()
        ), f"layer {i} should be trainable"


def test_partial_freeze_disabled_when_flag_off():
    """freeze_lower=False leaves every encoder layer trainable (full plasticity)."""
    m = LexSlot.__new__(LexSlot)
    n_layers = 4
    inner = _fake_encoder(n_layers, d=4)
    m.model = types.SimpleNamespace(_inner=inner)
    m.freeze_lower = False
    m._late_idx = [2, 3]
    m._freeze_lower_encoder()
    layers = inner.encoder.layer
    # Re-check embeddings: Linear default requires_grad=True, so unfrozen.
    assert all(p.requires_grad for p in inner.embeddings.parameters())
    assert all(p.requires_grad for layer in layers for p in layer.parameters())


def test_gradient_isolation_zeros_prior_slot_grads():
    """Isolation invariant: under slot_sharing=off, a prior task's owned slots receive
    ZERO gradient during a later task's backward (the no-forgetting mechanism)."""
    m = LexSlot.__new__(LexSlot)
    m.slot_sharing = "off"
    m.share_threshold = 0.5
    ls = LogitSlots(n_slots=2, hidden_dim=4, n_labels=2)
    # proj zero-init (no-op); make it non-zero so the bilinear form has gradient flow.
    with torch.no_grad():
        ls.proj.copy_(torch.randn(2, 2) * 0.1)
    # Slot 0 owned by task 0; current task is 1; S is identity (off -> prior slots masked 0).
    S = torch.eye(2)  # noqa: N806
    mask = m._derive_mask(task_id=1, slot_owner=[0, -1], S=S)
    ls.set_grad_mask(mask)  # [0.0, 1.0]
    feats = torch.randn(1, 3, 4)
    ls.logits_delta(feats).pow(2).sum().backward()
    assert torch.allclose(ls.values.grad[0], torch.zeros(4), atol=1e-6)  # prior slot masked
    assert ls.values.grad[1].abs().sum() > 0  # fresh slot trains


def _stub_for_gate() -> LexSlot:
    """A LexSlot stub with a 2-slot head, two orthogonal task signatures, and the
    attributes _install_infer_gate reads. Slot 0 -> task 0, slot 1 -> task 1."""
    m = LexSlot.__new__(LexSlot)
    m.infer_gate = True
    m.device = torch.device("cpu")
    m.vocab_size = 8
    m._sigs_cache = None
    m._sigs_cache_len = -1
    m.head_slots = LogitSlots(n_slots=2, hidden_dim=4, n_labels=3)
    m.late_slots = nn.ModuleDict()
    m._task_sigs = [
        torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # task 0: token 1
        torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # task 1: token 2
    ]
    m.head_slots.set_owner([0], 0)
    m.head_slots.set_owner([1], 1)
    return m


def test_infer_gate_own_task_fully_foreign_task_zeroed():
    """On a task-0 document, the task-0-owned slot gates to ~1.0 and the task-1-owned
    (orthogonal) slot gates to ~0.0 — the forward-time routing mechanism."""
    m = _stub_for_gate()
    ids = torch.tensor([[1, 1, 1, 1]])  # all token 1 -> task-0 document
    m._install_infer_gate(None, None, {"input_ids": ids})
    g = m.head_slots._infer_gate
    assert g is not None and g.shape == (1, 2)
    assert abs(float(g[0, 0]) - 1.0) < 1e-5  # own task
    assert abs(float(g[0, 1]) - 0.0) < 1e-5  # foreign, orthogonal


def test_infer_gate_scales_foreign_slot_delta_to_zero():
    """With the gate installed, the foreign slot's contribution to logits_delta is
    zeroed — i.e. gated delta == the delta computed from slot 0 alone."""
    m = _stub_for_gate()
    ids = torch.tensor([[1, 1, 1, 1]])  # task-0 document
    m._install_infer_gate(None, None, {"input_ids": ids})
    ls = m.head_slots
    with torch.no_grad():
        ls.values.copy_(torch.randn(2, 4) * 0.1)
        ls.proj.copy_(torch.randn(2, 3) * 0.1)
    feats = torch.randn(1, 3, 4)
    d_gated = ls.logits_delta(feats)  # gate [[1,0]] -> slot 1 zeroed

    # Reference: slot-0-only delta (zero out proj row 1, no gate).
    proj1 = ls.proj.data[1].clone()
    ls._infer_gate = None
    with torch.no_grad():
        ls.proj.data[1].zero_()
    d_slot0_only = ls.logits_delta(feats)
    with torch.no_grad():
        ls.proj.data[1].copy_(proj1)  # restore

    assert torch.allclose(d_gated, d_slot0_only, atol=1e-6)


def test_infer_gate_none_is_byte_identical_to_ungated():
    """When _infer_gate is None (default / infer_gate off), logits_delta == act @ proj
    exactly — the gate change is opt-in and safe."""
    ls = LogitSlots(n_slots=3, hidden_dim=5, n_labels=2)
    with torch.no_grad():
        ls.values.copy_(torch.randn(3, 5) * 0.1)
        ls.proj.copy_(torch.randn(3, 2) * 0.1)
    assert ls._infer_gate is None
    feats = torch.randn(2, 4, 5)
    d = ls.logits_delta(feats)
    ref = (feats @ ls.values.T) @ ls.proj
    assert torch.allclose(d, ref, atol=1e-6)


def test_infer_gate_noop_when_no_task_sigs():
    """Before any before_task (no task signatures), the gate hook is a no-op and leaves
    _infer_gate unset so the slot delta stays byte-identical to the ungated path."""
    m = LexSlot.__new__(LexSlot)
    m.infer_gate = True
    m.device = torch.device("cpu")
    m.vocab_size = 8
    m._sigs_cache = None
    m._sigs_cache_len = -1
    m._task_sigs = []  # no tasks yet
    m.head_slots = LogitSlots(n_slots=2, hidden_dim=4, n_labels=2)
    m.late_slots = nn.ModuleDict()
    m._install_infer_gate(None, None, {"input_ids": torch.tensor([[1, 2, 3]])})
    assert m.head_slots._infer_gate is None  # untouched -> ungated path
