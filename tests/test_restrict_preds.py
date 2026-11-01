"""AGENT FIX 2026-11-01: eval-time prediction restriction for token CIL (train.py)."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

spec = importlib.util.spec_from_file_location("train_mod", Path("scripts/train.py"))
train_mod = importlib.util.module_from_spec(spec)
sys.modules["train_mod"] = train_mod
spec.loader.exec_module(train_mod)


def test_restrict_preds_masks_disallowed_classes_in_eval_only():
    head = torch.nn.Linear(4, 5)
    model = SimpleNamespace(
        model=SimpleNamespace(classifier=head, config=SimpleNamespace(num_labels=5))
    )
    x = torch.zeros(2, 4)
    with torch.no_grad():
        head.bias.zero_()
        head.bias[3] = 5.0  # disallowed class wins -> must become O (index 0), not class 2
        head.bias[2] = 1.0
    with train_mod.restrict_preds(model, {0, 2}):
        head.eval()
        out = head(x)
        assert (out[:, [1, 3, 4]] <= -1e4).all()
        assert (out.argmax(-1) == 0).all()
        with torch.no_grad():
            head.bias[3] = -5.0  # allowed class 2 wins on its own -> kept
        assert (head(x).argmax(-1) == 2).all()
        head.train()
        assert (head(x) > -1e3).all()  # training untouched
    head.eval()
    assert (head(x) > -1e3).all()  # hook removed
