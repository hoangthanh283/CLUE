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


def test_diagnostic_logging_does_not_crash(tmp_path):
    import torch.nn as nn

    tb = TBLogger(tmp_path / "tb", enabled=True)
    lin = nn.Linear(4, 3)
    groups = {"classifier": [lin.weight, lin.bias]}
    # weight histograms + per-group L2 scalars
    tb.log_param_histograms(groups, "weights", step=0, which="weight")
    # grad path with no grads present → must not raise (skips empty groups)
    tb.log_param_histograms(groups, "grads", step=0, which="grad")
    # per-group scalar series (displacement / fisher / cka shape)
    tb.log_group_scalars({"head": 1.2, "late": 0.4, "bad": float("nan")}, "displacement/0_to_1", 1)
    tb.close()


def test_diagnostic_logging_noop_when_disabled():
    import torch.nn as nn

    tb = TBLogger("/tmp/doccl_tb_diag_disabled", enabled=False)
    lin = nn.Linear(2, 2)
    tb.log_param_histograms({"g": [lin.weight]}, "weights", step=0)
    tb.log_group_scalars({"a": 1.0}, "displacement/0_to_1", step=0)
