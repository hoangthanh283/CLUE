"""Unit tests for the MagMax port — the merge *lifecycle* on synthetic tensors (fast,
GPU-free). The ``max_abs`` merge math itself is covered in ``tests/methods/test_lca.py``
(``test_merge_state_dicts_max_abs``); here we check MagMax's wiring: backbone-only
snapshots, max-magnitude consolidation loaded back into the live model, and the growing
classifier head left untouched. The full lifecycle on a real LayoutLMv3 lives in
``tests/methods/test_all_methods_e2e.py`` once ``magmax`` is registered there.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from doccl.methods.magmax import MagMax


class _FakeInner(nn.Module):
    """Minimal stand-in for the inner HF model: one backbone param + one classifier (head)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Parameter(torch.zeros(4))  # "backbone" → merged
        self.classifier = nn.Parameter(torch.zeros(3))  # "head" → excluded from merge


class _FakeWrapper:
    def __init__(self) -> None:
        self.model = _FakeInner()


def _make_magmax(merge_method: str = "max_abs") -> MagMax:
    """Build a MagMax without running __init__ (no real backbone needed), à la test_lca."""
    m = MagMax.__new__(MagMax)
    m.model = _FakeWrapper()
    m.merge_method = merge_method
    m.merge_coef = 1.0
    m.merge_topk = 100
    m._base_backbone = None
    m._task_backbones = []
    return m


def _task(tid: int) -> SimpleNamespace:
    return SimpleNamespace(task_id=tid)


def test_backbone_state_excludes_classifier_head():
    m = _make_magmax()
    m.model.model.encoder.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    state = m._backbone_state()
    assert "encoder" in state
    assert "classifier" not in state  # the growing head must never enter the merge
    assert torch.allclose(state["encoder"], torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_before_task_captures_base_once():
    m = _make_magmax()
    m.before_task(_task(0), None)
    base0 = m._base_backbone["encoder"].clone()
    # Mutate the model and call before_task again — base must NOT be recaptured.
    m.model.model.encoder.data = torch.tensor([9.0, 9.0, 9.0, 9.0])
    m.before_task(_task(1), None)
    assert torch.allclose(m._base_backbone["encoder"], base0)  # still the original θ_0


def test_max_magnitude_consolidation_picks_largest_abs_per_coord():
    m = _make_magmax(merge_method="max_abs")
    # θ_0 = 0.
    m.before_task(_task(0), None)
    # Task 0 fine-tunes the encoder to τ0 = [3, -1, 0, 0]; after_task snapshots + merges.
    m.model.model.encoder.data = torch.tensor([3.0, -1.0, 0.0, 0.0])
    m.after_task(_task(0), None)
    assert torch.allclose(m.model.model.encoder.data, torch.tensor([3.0, -1.0, 0.0, 0.0]))
    # Task 1 → τ1 = [-2, 5, 0, 0]; max-|.| per coord vs τ0 → [3, 5, 0, 0].
    m.model.model.encoder.data = torch.tensor([-2.0, 5.0, 0.0, 0.0])
    m.after_task(_task(1), None)
    assert torch.allclose(m.model.model.encoder.data, torch.tensor([3.0, 5.0, 0.0, 0.0]))
    # The classifier head was never written by the merge.
    assert torch.allclose(m.model.model.classifier.data, torch.zeros(3))


def test_merge_coef_scales_task_vector():
    m = _make_magmax(merge_method="max_abs")
    m.merge_coef = 0.5
    m.before_task(_task(0), None)
    m.model.model.encoder.data = torch.tensor([4.0, -2.0, 0.0, 0.0])
    m.after_task(_task(0), None)
    # θ* = θ_0 + 0.5 · τ0 = 0 + 0.5·[4,-2,0,0] = [2,-1,0,0].
    assert torch.allclose(m.model.model.encoder.data, torch.tensor([2.0, -1.0, 0.0, 0.0]))
