"""Unit test for scripts/rca_tb_mine.scan_tb_dir against a synthetic event file."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

_spec = importlib.util.spec_from_file_location(
    "rca_tb_mine", Path(__file__).resolve().parents[1] / "scripts" / "rca_tb_mine.py"
)
rca_tb_mine = importlib.util.module_from_spec(_spec)
sys.modules["rca_tb_mine"] = rca_tb_mine
_spec.loader.exec_module(rca_tb_mine)


def _write_events(tb_dir: Path, tags: dict[str, float]) -> None:
    w = SummaryWriter(log_dir=str(tb_dir))
    for tag, value in tags.items():
        w.add_scalar(tag, value, global_step=0)
    w.close()


def test_scan_extracts_final_scalars_and_no_displacement(tmp_path):
    tb = tmp_path / "run_a" / "tb"
    _write_events(tb, {"final/AA": 87.6, "final/BWT": -1.7, "retention/task_0": 89.2})
    row = rca_tb_mine.scan_tb_dir(tb)
    assert row["run"] == "run_a"
    assert not row["has_displacement"]
    assert abs(row["AA"] - 87.6) < 1e-4
    assert abs(row["BWT"] - (-1.7)) < 1e-4


def test_scan_detects_displacement_tags(tmp_path):
    tb = tmp_path / "run_b" / "tb"
    _write_events(tb, {"displacement_depth/head": 0.5})
    assert rca_tb_mine.scan_tb_dir(tb)["has_displacement"]
