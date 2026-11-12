"""AGENT IMPL 2026-11-12: opt-in full-label prediction dump (train.py, training.full_label_dump)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from omegaconf import OmegaConf

spec = importlib.util.spec_from_file_location("train_mod_fld", Path("scripts/train.py"))
train_mod = importlib.util.module_from_spec(spec)
sys.modules["train_mod_fld"] = train_mod
spec.loader.exec_module(train_mod)


class _Shared:
    """Shared-softmax stand-in: evaluate() runs the head once per batch, like NaiveFineTune."""

    def __init__(self):
        torch.manual_seed(0)
        self.model = SimpleNamespace(
            model=SimpleNamespace(classifier=nn.Linear(4, 3).eval()),
            id_to_label={0: "O", 1: "B-a", 2: "I-a"},
        )

    def evaluate(self, loaders):
        for _tid, loader in loaders.items():
            for batch in loader:
                self.model.model.classifier(batch["x"])
        return {tid: SimpleNamespace(f1=50.0) for tid in loaders}


def _data():
    labels = [[1, 2, -100, 0], [0, -100, -100, 1], [2, 0, 1, -100]]
    return [{"x": torch.randn(4, 4), "labels": torch.tensor(lb)} for lb in labels]


def _cfg(dump: bool):
    return OmegaConf.create(
        {
            "training": {
                "batch_size": 2,
                "num_workers": 0,
                "full_label_eval": True,
                "full_label_dump": dump,
            }
        }
    )


def test_dump_is_off_by_default_and_saves_masked_tokens_when_on(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"AA": 1.0}))
    ds = _data()
    scen = SimpleNamespace(tasks=[0, 1], full_eval_dataset=ds)
    assert train_mod.add_full_label_f1(tmp_path, _cfg(False), scen, _Shared()) == 50.0
    assert not (tmp_path / "full_label_dump.pt").exists()

    method = _Shared()
    assert train_mod.add_full_label_f1(tmp_path, _cfg(True), scen, method) == 50.0
    d = torch.load(tmp_path / "full_label_dump.pt", weights_only=False)
    assert d["gold"].tolist() == [1, 2, 0, 0, 1, 2, 0, 1]
    assert d["doc"].tolist() == [0, 0, 0, 1, 1, 2, 2, 2]
    assert d["logits"].shape == (8, 3) and d["full_label_f1"] == 50.0
    x = torch.cat([ds[i]["x"][ds[i]["labels"] != -100] for i in range(3)])
    ref = method.model.model.classifier(x)
    assert torch.allclose(d["logits"].float(), ref, atol=1e-2)
    assert json.loads((tmp_path / "metrics.json").read_text())["full_label_f1"] == 50.0


def test_blocked_readout_dump_uses_eval_dump(tmp_path):
    class _Blocked(_Shared):
        eval_dump: dict = {}  # CoLaRSBR defines it in __init__

        def evaluate(self, loaders):
            n = sum(int((b["labels"] != -100).sum()) for b in next(iter(loaders.values())))
            self.eval_dump = {1: {"e": torch.rand(n, 2), "tag": torch.ones(n, 2), "threshold": 0.5}}
            return {1: SimpleNamespace(f1=70.0)}

    (tmp_path / "metrics.json").write_text("{}")
    scen = SimpleNamespace(tasks=[0, 1], full_eval_dataset=_data())
    assert train_mod.add_full_label_f1(tmp_path, _cfg(True), scen, _Blocked()) == 70.0
    d = torch.load(tmp_path / "full_label_dump.pt", weights_only=False)
    assert d["e"].shape == (8, 2) and d["threshold"] == 0.5 and "logits" not in d


def test_failed_dump_still_records_full_label_f1(tmp_path):
    class _NoHead:  # no .model.model.classifier -> the dump fails, the F1 must survive
        def evaluate(self, loaders):
            return {tid: SimpleNamespace(f1=42.0) for tid in loaders}

    (tmp_path / "metrics.json").write_text("{}")
    scen = SimpleNamespace(tasks=[0], full_eval_dataset=_data())
    assert train_mod.add_full_label_f1(tmp_path, _cfg(True), scen, _NoHead()) == 42.0
    assert not (tmp_path / "full_label_dump.pt").exists()
