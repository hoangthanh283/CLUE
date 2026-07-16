"""Plumbing test for rca_baselines.eval_under_mask — each mask reaches the model, and
the result dict carries f1/per_class/confusion. Uses a stub model (no real LayoutLMv3);
the actual masking math is LayoutLMv3Wrapper._apply_mask's job, covered by pilot tests.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

from doccl.types import ModalityMask

_spec = importlib.util.spec_from_file_location(
    "rca_baselines", Path(__file__).resolve().parents[2] / "scripts" / "rca_baselines.py"
)
rca_baselines = importlib.util.module_from_spec(_spec)
sys.modules["rca_baselines"] = rca_baselines
_spec.loader.exec_module(rca_baselines)

ID_TO_LABEL = {0: "O", 1: "B-A"}


class _Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_masks: list[ModalityMask] = []

    def forward(self, input_ids, attention_mask=None, modality_mask=ModalityMask.FULL):
        self.seen_masks.append(modality_mask)
        b, seq = input_ids.shape
        logits = torch.zeros(b, seq, len(ID_TO_LABEL))
        logits[..., 1] = 1.0  # always predict B-A
        return type("Out", (), {"logits": logits})()


def _loader():
    return [
        {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
            "labels": torch.tensor([[1, 1, -100, 0], [1, 0, 0, -100]]),
        }
    ]


def test_each_mask_reaches_the_model_and_result_schema():
    model = _Stub()
    for mask in rca_baselines.EVAL_MASKS:
        out = rca_baselines.eval_under_mask(model, _loader(), mask, ID_TO_LABEL, "cpu")
        assert set(out) == {"f1", "per_class", "confusion"}
        assert len(out["confusion"]) == len(ID_TO_LABEL)
    assert model.seen_masks == list(rca_baselines.EVAL_MASKS)


def test_confusion_and_f1_consistent_with_stub_predictions():
    out = rca_baselines.eval_under_mask(_Stub(), _loader(), ModalityMask.FULL, ID_TO_LABEL, "cpu")
    cm = out["confusion"]  # stub predicts class 1 everywhere; 6 supervised tokens
    assert cm[0][1] == 3 and cm[1][1] == 3 and cm[0][0] == 0
    assert 0 < out["f1"] <= 100
