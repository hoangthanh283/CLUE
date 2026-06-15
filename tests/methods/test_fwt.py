"""Tests for true Forward Transfer (FWT) computation.

FWT = mean_{i>0} (R[i-1, i] - b_i), where R[i-1, i] is the model's zero-shot F1 on
task i *before* training it, and b_i is the single-task from-scratch baseline. These
tests cover the per-run path (CLMetricsTracker.forward_transfer) and the aggregation
path (analyze_results.compute_fwt_per_run), and assert the two agree.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from doccl.eval.metrics import CLMetricsTracker

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from analyze_results import compute_fwt_per_run  # noqa: E402


def _matrix_with_zeroshot() -> list:
    # dil: tasks [funsd, sroie, cord]. Upper triangle = zero-shot future-task F1.
    return [
        [88.0, 3.85, np.nan],   # R[0,1] = zero-shot sroie after funsd
        [np.nan, 82.0, 0.71],   # R[1,2] = zero-shot cord after sroie
        [np.nan, np.nan, 90.0],
    ]


def test_per_run_and_aggregation_agree() -> None:
    R = _matrix_with_zeroshot()
    base = {"funsd": {"mean": 85.0}, "sroie": {"mean": 81.0}, "cord": {"mean": 90.0}}

    fwt_agg = compute_fwt_per_run(R, "dil", base)
    tracker = CLMetricsTracker(num_tasks=3, baseline_perf=[85.0, 81.0, 90.0])
    tracker.matrix = np.array(R)
    fwt_run = tracker.forward_transfer()

    expected = float(np.mean([3.85 - 81.0, 0.71 - 90.0]))
    assert math.isclose(fwt_run, expected, abs_tol=1e-6)
    assert math.isclose(fwt_agg, fwt_run, abs_tol=1e-6)


def test_fwt_nan_when_no_baseline_seeded() -> None:
    # Per-run: tracker without baseline_perf must NOT report 0.0 (0 is a real value).
    tracker = CLMetricsTracker(num_tasks=3)
    tracker.matrix = np.array(_matrix_with_zeroshot())
    assert math.isnan(tracker.forward_transfer())


def test_fwt_nan_for_lower_triangular_legacy_matrix() -> None:
    # Legacy runs have no zero-shot term → FWT must be NaN, never fabricated.
    legacy = [
        [88.0, np.nan, np.nan],
        [70.0, 82.0, np.nan],
        [20.0, 30.0, 90.0],
    ]
    base = {"funsd": {"mean": 85.0}, "sroie": {"mean": 81.0}, "cord": {"mean": 90.0}}
    assert math.isnan(compute_fwt_per_run(legacy, "dil", base))


def test_fwt_nan_for_joint_style_matrix() -> None:
    # Joint fills only the last row (no sequential zero-shot) → FWT undefined → NaN.
    tracker = CLMetricsTracker(num_tasks=3, baseline_perf=[85.0, 81.0, 90.0])
    tracker.matrix = np.array(
        [[np.nan] * 3, [np.nan] * 3, [88.0, 82.0, 90.0]]
    )
    assert math.isnan(tracker.forward_transfer())


def test_fwt_nan_when_dataset_baseline_missing() -> None:
    R = _matrix_with_zeroshot()
    assert math.isnan(compute_fwt_per_run(R, "dil", {}))
