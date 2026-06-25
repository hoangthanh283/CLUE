"""Tests for the secondary-backbone (LiLT/BROS) pilot conditions.

These guard the wiring that lets the forgetting localizer (CKA/Fisher/displacement)
run on LiLT and BROS as well as LayoutLMv3+BERT — i.e. that the head/depth-dominant
forgetting claim is testable across multiple multimodal backbones, not just one.

Network/GPU-free: construction + invariants only (real training is the grid's job).
"""

from __future__ import annotations

import pytest

from doccl.pilot.run_pilot import (
    _MASKLESS_CONDITIONS,
    _SECONDARY_BACKBONES,
    ALL_CONDITIONS,
)


def test_secondary_backbones_registered_in_all_conditions():
    for cond in ("cl_lilt", "cr_bros"):
        assert cond in ALL_CONDITIONS
        assert cond in _SECONDARY_BACKBONES


def test_secondary_backbones_are_maskless():
    """LiLT/BROS are vision-free text+layout — no modality mask is threaded."""
    assert "cl_lilt" in _MASKLESS_CONDITIONS
    assert "cr_bros" in _MASKLESS_CONDITIONS
    # The LayoutLMv3 family stays masked.
    assert "c4_full" not in _MASKLESS_CONDITIONS


@pytest.mark.parametrize(
    "condition,inner_attr",
    [("cl_lilt", "lilt"), ("cr_bros", "bros")],
)
def test_build_condition_model_constructs_secondary_backbone(condition, inner_attr):
    """The condition builds its wrapper with a null mask and the depth-bucket groups
    the displacement localizer needs."""
    pytest.importorskip("transformers")
    from doccl.pilot.run_pilot import _build_condition_model

    class _Task:
        label_set = ["O", "B-HEADER", "I-HEADER", "B-QUESTION", "I-QUESTION"]

    class _Scenario:
        tasks = [_Task()]

    try:
        model, mask = _build_condition_model(condition, _Scenario())
    except OSError:
        pytest.skip("backbone weights not available offline")
    assert mask is None  # maskless secondary backbone
    assert model._inner_attr == inner_attr
    # The forgetting localizer reads these; they must be populated (not all in misc).
    by_depth = model.param_groups_by_depth
    assert any(len(v) for k, v in by_depth.items() if k != "misc")
    assert "head" in by_depth
