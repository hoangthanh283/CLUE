"""Image-backbone paths of the diagnostics tooling (H1 pilot conditions)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from doccl.eval.cka import _select_vectors
from doccl.pilot import run_pilot


def test_select_vectors_uses_cls_when_no_attention_mask():
    feat = torch.randn(4, 197, 8)
    out = _select_vectors(feat, attention_mask=None, labels=None, token_level=True)
    assert out.shape == (4, 8)
    assert torch.equal(out, feat[:, 0, :])


class _Tiny(nn.Module):
    task_type = "image"

    def __init__(self):
        super().__init__()
        self.enc = nn.Linear(12, 8)
        self.classifier = nn.Linear(8, 3)
        self.id_to_label = {i: f"c{i}" for i in range(3)}

    def forward(self, pixel_values, labels=None, **kw):
        logits = self.classifier(self.enc(pixel_values.flatten(1)))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return type("O", (), {"logits": logits, "loss": loss})()


def _loader():
    items = [
        {"pixel_values": torch.randn(3, 2, 2), "labels": torch.tensor(i % 3)} for i in range(6)
    ]
    return DataLoader(items, batch_size=3)


def test_pilot_slow_recipe_trains_head_faster_than_trunk():
    m = _Tiny()
    enc0, head0 = m.enc.weight.clone(), m.classifier.weight.clone()
    run_pilot._train_naive(m, _loader(), None, 1, torch.device("cpu"), True, recipe="slow")
    assert (m.classifier.weight - head0).abs().mean() > (m.enc.weight - enc0).abs().mean()
    res = run_pilot._evaluate_all(m, {0: _loader()}, None, torch.device("cpu"), True)
    assert 0.0 <= res[0]["f1"] <= 100.0


def test_vision_conditions_registered():
    assert {"cv_vit_fast", "cv_vit_slow"} <= set(run_pilot.ALL_CONDITIONS)
    assert {"cv_vit_fast", "cv_vit_slow"} <= run_pilot._MASKLESS_CONDITIONS
