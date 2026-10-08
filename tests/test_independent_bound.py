"""Independent per-task (multi-model) upper bound: scenario slicing + result aggregation."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from torch.utils.data import Dataset

from doccl.data.scenarios import CLScenario, _build_vision_cil, single_task_scenario
from doccl.types import ScenarioType, TaskInfo


class _Toy(Dataset):
    def __init__(self, n_per_class: int, num_classes: int):
        self.targets = [c for c in range(num_classes) for _ in range(n_per_class)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return Image.new("RGB", (8, 8)), self.targets[i]


def _dil_like(n_tasks: int = 3) -> CLScenario:
    tasks = [
        TaskInfo(
            task_id=i,
            task_name=f"t{i}",
            label_set=["O", "A"],
            is_first=i == 0,
            is_last=i == n_tasks - 1,
            metadata={"native_dataset": f"d{i}"},
        )
        for i in range(n_tasks)
    ]
    return CLScenario(
        "dil",
        ScenarioType.DIL,
        tasks,
        [f"tr{i}" for i in range(n_tasks)],
        [f"ev{i}" for i in range(n_tasks)],
    )


def test_dil_slice_keeps_task_k_only():
    one = single_task_scenario(_dil_like(), 1)
    assert one.name == "dil_indep1"
    assert len(one.tasks) == 1 and one.tasks[0].is_first and one.tasks[0].is_last
    assert one.tasks[0].task_id == 0 and one.tasks[0].task_name == "t1"
    assert one.train_datasets == ["tr1"] and one.eval_datasets == ["ev1"]
    assert one.tasks[0].metadata["native_dataset"] == "d1"


def test_dil_slice_rejects_growing_head_and_bad_index():
    cil = _dil_like()
    cil.scenario_type = ScenarioType.CIL
    with pytest.raises(ValueError, match="fixed label space"):
        single_task_scenario(cil, 0)
    with pytest.raises(ValueError, match="out of range"):
        single_task_scenario(_dil_like(), 3)


def test_vision_only_task_uses_local_head_index():
    bases = (_Toy(2, 8), _Toy(1, 8), None, 8)
    full = _build_vision_cil("toy", bases, num_sessions=4, class_order_seed=0)
    one = _build_vision_cil("toy", bases, num_sessions=4, class_order_seed=0, only_task=2)
    assert one.name == "toy_indep2" and len(one.tasks) == 1
    assert one.tasks[0].label_set == full.tasks[2].label_set  # same classes as session 2
    labels = sorted(
        {int(one.train_datasets[0][i]["labels"]) for i in range(len(one.train_datasets[0]))}
    )
    assert labels == [0, 1]  # local head: 0..m-1, not the cumulative CIL indices
    assert len(one.train_datasets[0]) == 4 and len(one.eval_datasets[0]) == 2


def test_add_independent_rows_averages_over_tasks():
    from scripts.analyze_results import add_independent_rows

    base = {
        "state": "finished",
        "method": "naive",
        "model_family": "layoutlmv3",
        "seed": 42,
        "BWT": 0.0,
        "AF": 0.0,
        "FWT": np.nan,
        "mean_time_per_task_s": 10.0,
        "peak_gpu_mem_mb": 1.0,
        "matrix": None,
    }
    df = pd.DataFrame(
        [
            {**base, "name": "dil_indep0_naive_seed42", "scenario": "dil_indep0", "AA": 80.0},
            {**base, "name": "dil_indep1_naive_seed42", "scenario": "dil_indep1", "AA": 90.0},
            {**base, "name": "dil_naive_seed42", "scenario": "dil", "AA": 40.0},
        ]
    )
    out = add_independent_rows(df)
    ind = out[out.method == "independent"]
    assert len(ind) == 1 and ind.iloc[0].scenario == "dil" and ind.iloc[0].AA == 85.0
    assert ind.iloc[0].AF == 2  # task coverage
    assert len(out) == 4  # original rows untouched
