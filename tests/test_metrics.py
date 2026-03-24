"""Tests for src/training/cl_metrics.py."""

import os
from unittest.mock import patch

import pytest

from src.training.cl_metrics import compute_cl_metrics, save_aaa_curve_plot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mat(data):
    """Convert nested list to float matrix."""
    return [[float(x) for x in row] for row in data]


# ---------------------------------------------------------------------------
# Single-task edge case
# ---------------------------------------------------------------------------


def test_single_task():
    R = _mat([[0.8]])
    R0 = [0.0]
    m = compute_cl_metrics(R, R0, ["task0"])
    assert m["ACC"] == pytest.approx(0.8)
    assert m["BWT"] == pytest.approx(0.0)
    assert m["FWT"] == pytest.approx(0.0)
    assert m["AAA"] == pytest.approx(0.8)
    assert m["AAA_curve"] == pytest.approx([0.8])
    assert m["Forgetting"] == pytest.approx(0.0)
    assert m["Forgetting_per_task"] == {"task0": pytest.approx(0.0)}


# ---------------------------------------------------------------------------
# Two-task case – manual verification
# ---------------------------------------------------------------------------


def test_two_tasks_basic():
    # After task0: acc on task0 = 0.9
    # After task1: acc on task0 = 0.7, acc on task1 = 0.8
    R = _mat([
        [0.9, 0.0],   # after task0 (task1 not yet trained → 0)
        [0.7, 0.8],   # after task1
    ])
    R0 = [0.0, 0.1]
    m = compute_cl_metrics(R, R0, ["A", "B"])

    # ACC = mean of final row
    assert m["ACC"] == pytest.approx((0.7 + 0.8) / 2)

    # BWT = (R[1,0] - R[0,0]) / (T-1) = (0.7 - 0.9) / 1 = -0.2
    assert m["BWT"] == pytest.approx(-0.2)

    # FWT = (R[0,1] - R0[1]) / (T-1) = (0.0 - 0.1) / 1 = -0.1
    assert m["FWT"] == pytest.approx(-0.1)

    # AAA_curve: after task0 → avg(0.9)=0.9, after task1 → avg(0.7,0.8)=0.75
    assert m["AAA_curve"] == pytest.approx([0.9, 0.75])
    assert m["AAA"] == pytest.approx((0.9 + 0.75) / 2)

    # Forgetting on A: best across rows ≥0 is R[0,0]=0.9, final=0.7 → 0.2
    assert m["Forgetting_per_task"]["A"] == pytest.approx(0.2)
    # Forgetting on B: best=0.8, final=0.8 → 0.0
    assert m["Forgetting_per_task"]["B"] == pytest.approx(0.0)
    assert m["Forgetting"] == pytest.approx(0.1)  # (0.2+0.0)/2


# ---------------------------------------------------------------------------
# Three-task case
# ---------------------------------------------------------------------------


def test_three_tasks():
    R = _mat([
        [0.8, 0.0, 0.0],
        [0.7, 0.9, 0.0],
        [0.6, 0.8, 0.85],
    ])
    R0 = [0.0, 0.0, 0.0]
    m = compute_cl_metrics(R, R0, ["t0", "t1", "t2"])

    assert m["ACC"] == pytest.approx((0.6 + 0.8 + 0.85) / 3)

    # BWT = mean(R[2,0]-R[0,0], R[2,1]-R[1,1]) / (3-1)
    bwt = ((0.6 - 0.8) + (0.8 - 0.9)) / 2
    assert m["BWT"] == pytest.approx(bwt)

    assert len(m["AAA_curve"]) == 3
    assert m["AAA_curve"][0] == pytest.approx(0.8)
    assert m["AAA_curve"][1] == pytest.approx((0.7 + 0.9) / 2)
    assert m["AAA_curve"][2] == pytest.approx((0.6 + 0.8 + 0.85) / 3)


# ---------------------------------------------------------------------------
# Perfect retention – no forgetting, no BWT
# ---------------------------------------------------------------------------


def test_perfect_retention():
    R = _mat([
        [0.9, 0.0, 0.0],
        [0.9, 0.85, 0.0],
        [0.9, 0.85, 0.95],
    ])
    R0 = [0.0, 0.0, 0.0]
    m = compute_cl_metrics(R, R0, ["t0", "t1", "t2"])

    assert m["BWT"] == pytest.approx(0.0)
    assert m["Forgetting"] == pytest.approx(0.0)
    for v in m["Forgetting_per_task"].values():
        assert v == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Positive FWT – model already knows next task
# ---------------------------------------------------------------------------


def test_positive_fwt():
    R = _mat([
        [0.9, 0.6],
        [0.85, 0.95],
    ])
    R0 = [0.0, 0.4]  # R0 low → FWT positive
    m = compute_cl_metrics(R, R0, ["t0", "t1"])
    # FWT = (R[0,1] - R0[1]) / 1 = (0.6 - 0.4) = 0.2
    assert m["FWT"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_assertion_wrong_shape():
    with pytest.raises(AssertionError):
        compute_cl_metrics([[0.5, 0.4]], [0.0, 0.0], ["t0", "t1"])  # R is 1×2 not 2×2


# ---------------------------------------------------------------------------
# save_aaa_curve_plot creates a file
# ---------------------------------------------------------------------------


def test_save_aaa_curve_plot_creates_file(tmp_path):
    out = str(tmp_path / "aaa.png")
    save_aaa_curve_plot([0.5, 0.6, 0.7], ["t0", "t1", "t2"], out)
    assert os.path.isfile(out)
    assert os.path.getsize(out) > 0


def test_save_aaa_curve_plot_sns_exception(tmp_path):
    """When sns.set_context raises, the exception is silently caught (lines 94-95)."""
    out = str(tmp_path / "aaa_sns.png")
    with patch("seaborn.set_context", side_effect=Exception("sns error")):
        save_aaa_curve_plot([0.5, 0.6], ["t0", "t1"], out)
    assert os.path.isfile(out)
