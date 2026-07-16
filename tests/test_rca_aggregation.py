"""Unit tests for the RCA Tier-A aggregation logic — synthetic, hand-computable inputs.

The one place a silent bug hides is index arithmetic (task off-by-one, wrong matrix
triangle) — these tests pin the numbers by hand.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # so rca_matrix_dynamics can import rca_per_class
    spec.loader.exec_module(mod)
    return mod


rca_per_class = _load("rca_per_class")
rca_matrix_dynamics = _load("rca_matrix_dynamics")


def _ledger():
    rows = []
    for method, aa, key_f1 in (("er", 86.8, 87.0), ("naive", 40.0, 0.0), ("joint", 89.7, 90.0)):
        for task_idx, classes in enumerate(
            [
                {"KEY": (key_f1, 100), "VALUE": (aa, 200)},
                {"KEY": (key_f1, 50), "VALUE": (aa, 400)},
                {"VALUE": (aa, 300)},  # CORD: no KEY class
            ]
        ):
            for cls, (f1, support) in classes.items():
                rows.append(
                    {
                        "run": f"dil_{method}_seed42",
                        "method": method,
                        "family": rca_per_class.FAMILY.get(method, "other"),
                        "scenario": "dil",
                        "seed": 42,
                        "AA": aa,
                        "BWT": 0.0,
                        "task_idx": task_idx,
                        "class": cls,
                        "is_avg": False,
                        "precision": f1,
                        "recall": f1,
                        "f1": f1,
                        "support": support,
                    }
                )
    return pd.DataFrame(rows)


def test_cross_task_class_f1_is_support_weighted():
    cross = rca_per_class.cross_task_class_f1(_ledger())
    er_key = cross[(cross.method == "er") & (cross["class"] == "KEY")].iloc[0]
    assert er_key.support == 150  # 100 + 50, CORD contributes nothing
    assert abs(er_key.class_f1 - 87.0) < 1e-9  # constant per-task F1 -> same weighted


def test_m5q6_flags_naive_not_er():
    verdicts = rca_per_class.m5q6_verdicts(rca_per_class.cross_task_class_f1(_ledger()))
    v = verdicts.set_index("method")
    assert not v.loc["er", "degeneracy_flag"]  # KEY tracks AA -> not degenerate
    assert v.loc["naive", "degeneracy_flag"]  # KEY dead, VALUE carries AA -> degenerate


def test_class_survival_gap_to_joint():
    df = _ledger()
    surv = rca_per_class.class_survival(df, rca_per_class.joint_reference(df))
    cell = surv[(surv.method == "naive") & (surv["class"] == "KEY") & (surv.task_idx == 0)]
    assert abs(float(cell.gap_to_joint.iloc[0]) - (0.0 - 90.0)) < 1e-9


def _runs(matrix):
    return pd.DataFrame(
        [
            {
                "run": "r",
                "method": "naive",
                "family": "none",
                "scenario": "dil",
                "seed": 42,
                "AA_reported": float(np.nanmean(matrix[-1])),
                "matrix": matrix,
                "T": matrix.shape[0],
            }
        ]
    )


def test_immediate_vs_gradual_decomposition():
    # task 0: peak 90 -> 30 after one boundary (immediate -60) -> 40 final (gradual +10)
    m = np.array([[90.0, np.nan, np.nan], [30.0, 85.0, np.nan], [40.0, 60.0, 95.0]])
    drops = rca_matrix_dynamics.immediate_vs_gradual(_runs(m))
    t0 = drops[drops.task == 0].iloc[0]
    assert t0.immediate_drop == -60.0 and t0.gradual_drop == 10.0 and t0.total_drop == -50.0
    t1 = drops[drops.task == 1].iloc[0]  # last boundary: immediate == total, gradual 0
    assert t1.immediate_drop == -25.0 and t1.gradual_drop == 0.0


def test_trajectory_skips_upper_triangle_and_aa_crosscheck_passes():
    m = np.array([[90.0, np.nan], [40.0, 80.0]])
    runs = _runs(m)
    traj = rca_matrix_dynamics.per_task_trajectory(runs)
    assert len(traj) == 3  # (0,0), (1,0), (1,1) — NaN future cell dropped
    cc = rca_matrix_dynamics.aa_crosscheck(runs)
    assert not cc.mismatch.any()
