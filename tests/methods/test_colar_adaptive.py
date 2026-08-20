from types import SimpleNamespace
from unittest.mock import patch

from doccl.methods.colar import CoLaR
from doccl.methods.colar_adaptive import CoLaRAdaptive
from doccl.methods.latent_replay import LatentReplay


def _make(config):
    def fake_parent_init(method, _model, _config):
        method.docs_per_task = int(_config.get("docs_per_task", 5))
        method.rank_r = int(_config.get("rank_r", 64))

    with patch.object(CoLaR, "__init__", fake_parent_init):
        return CoLaRAdaptive(None, config)


def test_empty_schedules_leave_colar_defaults_untouched():
    method = _make({"docs_per_task": 5, "rank_r": 64})

    assert method.docs_schedule == [] and method.rank_schedule == []
    assert method.docs_per_task == 5 and method.rank_r == 64


def test_schedules_apply_per_task_and_fall_back_beyond_length():
    method = _make(
        {"docs_per_task": 5, "rank_r": 64, "docs_schedule": [20, 5, 0], "rank_schedule": [64, 128]}
    )
    banked = []

    def fake_after_task(self, task, train_loader):
        banked.append((self.docs_per_task, self.rank_r))

    with patch.object(CoLaR, "after_task", fake_after_task):
        for t in range(4):
            method.after_task(SimpleNamespace(task_id=t), None)

    # task 2 banks nothing (docs=0, rank falls back to the sticky prior value);
    # task 3 is beyond both schedules and keeps the last applied values.
    assert banked == [(20, 64), (5, 128), (0, 128), (0, 128)]


def test_zero_docs_skips_banking_in_parent():
    # The docs_per_task > 0 guard in LatentReplay.after_task is what makes a
    # 0-entry in docs_schedule an actual no-bank instruction.
    import inspect

    src = inspect.getsource(LatentReplay.after_task)
    assert "docs_per_task > 0" in src
