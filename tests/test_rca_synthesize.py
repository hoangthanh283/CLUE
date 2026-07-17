"""Unit tests for the RCA Tier-C synthesis — synthetic, hand-computable Tier B JSON.

Pins the same silent-bug spots as test_rca_aggregation: index arithmetic (str task keys,
boundary offsets), share denominators, and the a2 join not dropping rows.
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
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rca_synthesize = _load("rca_synthesize")

LABELS = ["O", "B-KEY", "I-KEY", "B-VALUE", "I-VALUE"]


def _conf(rows):
    return np.asarray(rows, dtype=float).tolist()


def _tier_b_json():
    """2-task run, JSON-parsed shape (string task keys), hand-computable numbers."""

    def ev(f1_by_mask, confusion):
        return {
            m: {"f1": f1, "per_class": {}, "confusion": confusion} for m, f1 in f1_by_mask.items()
        }

    # Boundary 0: task 0 at-learning. full 80, text_only 70, text_layout 75, image_layout 10
    # (image_layout fails the >=20 validity guard).
    identity = _conf(np.eye(5) * 10)
    # Boundary 1: task 0 forgotten. Gold KEY rows (1,2): 2 retained in KEY block,
    # 8 to B-VALUE, 4 to O, 2 to I-VALUE -> total 16, off 14.
    forgot = _conf(
        [
            [10, 0, 0, 0, 0],  # O
            [4, 1, 0, 6, 1],  # B-KEY
            [0, 0, 1, 2, 1],  # I-KEY
            [0, 0, 0, 9, 0],  # B-VALUE
            [0, 0, 0, 0, 9],  # I-VALUE
        ]
    )
    return {
        "method": "naive",
        "seed": 42,
        "scenario": "dil",
        "labels": LABELS,
        "masks": rca_synthesize.MASKS,
        "caveats": [],
        "boundaries": [
            {
                "after_task": 0,
                "task_name": "funsd",
                "eval": {
                    "0": ev(
                        {
                            "full": 80.0,
                            "text_only": 70.0,
                            "text_layout": 75.0,
                            "image_layout": 10.0,
                        },
                        identity,
                    )
                },
            },
            {
                "after_task": 1,
                "task_name": "sroie",
                "displacement_by_group": {
                    "text_word_embed": 3.0,
                    "layout_2d_pos_embed": 1.0,
                    "image_patch_embed": 0.5,
                    "classifier": 9.0,
                },
                "displacement_by_depth": {
                    "input": 1.0,
                    "early": 1.0,
                    "mid": 2.0,
                    "late": 4.0,
                    "head": 9.0,
                },
                "eval": {
                    "0": ev(
                        {"full": 30.0, "text_only": 40.0, "text_layout": 35.0, "image_layout": 5.0},
                        forgot,
                    ),
                    "1": ev(
                        {
                            "full": 90.0,
                            "text_only": 80.0,
                            "text_layout": 85.0,
                            "image_layout": 15.0,
                        },
                        identity,
                    ),
                },
            },
        ],
    }


def test_modality_deltas_drop_and_validity_guard():
    d = rca_synthesize.modality_deltas(_tier_b_json())
    t0 = d[(d.task_idx == 0)].set_index("mask")
    assert t0.loc["full", "drop"] == 50.0  # 80 -> 30
    assert t0.loc["text_only", "drop"] == 30.0  # 70 -> 40
    assert bool(t0.loc["image_layout", "mask_valid"]) is False  # at-learning 10 < 20
    assert bool(t0.loc["full", "is_old_task"]) is True
    assert bool(d[(d.task_idx == 1) & (d["mask"] == "full")].is_old_task.iloc[0]) is False


def test_confusion_flow_shares_hand_computed():
    f = rca_synthesize.confusion_flow(_tier_b_json())
    key = f[(f.entity == "KEY") & (f.task_idx == 0)].iloc[0]
    assert key.gold_tokens == 16.0
    assert key.retained_share == 2.0 / 16.0  # KEY block: [[1,0],[0,1]]
    assert key.off_diag_mass == 14.0
    assert key.to_O_share == 4.0 / 14.0
    assert key.to_VALUE_share == 10.0 / 14.0  # 6+2 B-VALUE + 1+1 I-VALUE
    assert key.top1_col == "B-VALUE" and key.top1_share == 8.0 / 14.0
    # boundary 0 emits nothing; only old tasks at boundary 1
    assert set(f.boundary) == {1} and set(f.task_idx) == {0}


def test_displacement_profile_and_join_drops_no_rows(tmp_path):
    prof = rca_synthesize.displacement_profile(_tier_b_json())
    assert len(prof) == 1  # boundary 0 has no displacement
    row = prof.iloc[0]
    assert row.late_share == 4.0 / 8.0 and row.input_share == 1.0 / 8.0
    assert row.head_displacement == 9.0 and row.text_word_embed_disp == 3.0
    assert bool(row.trunk_frozen) is False

    a2 = tmp_path / "a2_family_signature.csv"
    pd.DataFrame(
        [{"method": "naive", "family": "none", "immediate_share": 1.12, "total_drop": -67.7}]
    ).to_csv(a2, index=False)
    joined = rca_synthesize.join_signature(prof, a2)
    assert len(joined) == len(prof)  # join must not drop rows
    assert joined.immediate_share.iloc[0] == 1.12

    # unknown method still keeps its rows (left join), signature NaN
    prof2 = prof.assign(method="mystery")
    joined2 = rca_synthesize.join_signature(prof2, a2)
    assert len(joined2) == 1 and joined2.immediate_share.isna().all()


def test_marginal_snap_hand_computed():
    s = rca_synthesize.marginal_snap(_tier_b_json())
    assert len(s) == 1  # boundary 1, old task 0 only
    row = s.iloc[0]
    # forgot matrix: O-row [10,0,0,0,0] -> O fully retained
    assert row.o_row_acc == 1.0
    # pred column sums [14,1,1,17,11]/44 -> top1 = B-VALUE
    assert row.pred_top1 == "B-VALUE"
    # trained (task 1) gold = identity*10 -> uniform marginal; cos = 1/(||pred||*sqrt(5))
    assert abs(row.cos_pred_vs_trained_gold - 0.798) < 0.001
    assert 0 < row.cos_pred_vs_own_gold <= 1


def test_trunk_frozen_flag():
    data = _tier_b_json()
    data["boundaries"][1]["displacement_by_depth"] = {"head": 9.0}  # colar-style frozen trunk
    prof = rca_synthesize.displacement_profile(data)
    assert bool(prof.iloc[0].trunk_frozen) is True
    assert np.isnan(prof.iloc[0].late_share)
