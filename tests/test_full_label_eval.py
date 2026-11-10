"""AGENT IMPL 2026-11-11: opt-in full-label final evaluation hook (train.py)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

spec = importlib.util.spec_from_file_location("train_mod_fl", Path("scripts/train.py"))
train_mod = importlib.util.module_from_spec(spec)
sys.modules["train_mod_fl"] = train_mod
spec.loader.exec_module(train_mod)


class _Method:
    def evaluate(self, loaders):
        return {tid: SimpleNamespace(f1=91.5) for tid in loaders}


def _cfg(on: bool):
    return OmegaConf.create(
        {"training": {"batch_size": 2, "num_workers": 0, "full_label_eval": on}}
    )


def test_full_label_f1_is_appended_only_when_enabled_and_available(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"AA": 1.0}))
    scen = SimpleNamespace(tasks=[0, 1, 2], full_eval_dataset=[{"x": 1}, {"x": 2}])
    assert train_mod.add_full_label_f1(tmp_path, _cfg(False), scen, _Method()) is None
    assert train_mod.add_full_label_f1(tmp_path, _cfg(True), scen, _Method()) == 91.5
    assert json.loads((tmp_path / "metrics.json").read_text()) == {
        "AA": 1.0,
        "full_label_f1": 91.5,
    }
    none = SimpleNamespace(tasks=[0], full_eval_dataset=None)
    assert train_mod.add_full_label_f1(tmp_path, _cfg(True), none, _Method()) is None


def test_full_label_eval_never_raises(tmp_path):
    class Boom:
        def evaluate(self, loaders):
            raise RuntimeError("boom")

    (tmp_path / "metrics.json").write_text("{}")
    scen = SimpleNamespace(tasks=[0], full_eval_dataset=[1])
    assert train_mod.add_full_label_f1(tmp_path, _cfg(True), scen, Boom()) is None
