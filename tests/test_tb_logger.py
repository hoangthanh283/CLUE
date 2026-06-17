"""Offline tests for the TensorBoard forgetting-diagnostic logger.

Exercises the matplotlib heatmap rendering and the graceful no-op path (no crash
when TensorBoard is disabled or the package is unavailable). Actual event-file
writes are validated on the grid machine where ``tensorboard`` is installed.
"""
from __future__ import annotations

import numpy as np

from doccl.utils.tb_logger import TBLogger, render_forgetting_heatmap


def _sample_matrix() -> np.ndarray:
    m = np.full((3, 3), np.nan)
    m[0, 0] = 90.0
    m[1, 0] = 70.0
    m[1, 1] = 85.0
    m[2, 0] = 60.0
    m[2, 1] = 80.0
    m[2, 2] = 88.0
    return m


def test_render_forgetting_heatmap_shape():
    img = render_forgetting_heatmap(_sample_matrix(), ["t0", "t1", "t2"])
    assert img is not None
    assert img.ndim == 3 and img.shape[2] == 3  # HWC RGB
    assert img.dtype == np.uint8


def test_disabled_logger_is_noop():
    tb = TBLogger("/tmp/doccl_tb_disabled", enabled=False)
    assert not tb.enabled
    m = _sample_matrix()
    # None of these should raise when disabled.
    tb.log_scalars({"final/AA": 80.0, "final/FWT": float("nan")}, step=2)
    tb.log_retention(m, task_idx=2)
    tb.log_forgetting_matrix(m, step=2, task_names=["t0", "t1", "t2"])
    tb.flush()
    tb.close()


def test_enabled_logger_does_not_crash_without_tensorboard(tmp_path):
    # When tensorboard is installed it writes events; when absent it degrades to a
    # no-op. Either way the calls must not raise.
    tb = TBLogger(tmp_path / "tb", enabled=True)
    m = _sample_matrix()
    tb.log_scalars({"metrics/running_AA": 75.0}, step=2)
    tb.log_retention(m, task_idx=2)
    tb.close()
