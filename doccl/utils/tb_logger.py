"""TensorBoard logging for CL experiments, focused on the forgetting diagnostic.

Runs alongside W&B. The centerpiece is the T×T retention matrix
``R[i][j] = entity-F1 on task j after training task i``, exposed three ways so the
forgetting story is visible *while* a run trains:

    diagnostic/forgetting_matrix  — annotated heatmap image, re-logged after every
                                    task (scrub the step slider to watch forgetting
                                    fill the lower triangle as tasks are learned)
    retention/task_<j>            — task j's F1 vs. training step → its forgetting
                                    curve (it should decay as later tasks are learned)
    metrics/running_AA, final/*   — running + final AA / BWT / AF / FWT scalars

Gracefully degrades to a no-op when the ``tensorboard`` package is unavailable, so a
run never fails over logging.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def render_forgetting_heatmap(
    matrix: np.ndarray, task_names: list[str] | None = None
) -> np.ndarray | None:
    """Render ``R[i][j]`` as an annotated heatmap → HWC uint8 RGB array (or None).

    NaN cells (not-yet-measured) are masked. Returns None if matplotlib is missing.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib is a hard dep, but be safe
        return None

    T = matrix.shape[0]
    size = max(4.0, T * 0.9)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap="viridis", vmin=0, vmax=100, aspect="equal")
    ax.set_xlabel("evaluated on task j")
    ax.set_ylabel("after training task i")
    labels = task_names or [str(i) for i in range(T)]
    ax.set_xticks(range(T))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(T))
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(T):
        for j in range(T):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.0f}",
                    ha="center",
                    va="center",
                    color="white" if v < 55 else "black",
                    fontsize=7,
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="entity F1")
    fig.tight_layout()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return img


class TBLogger:
    """Thin ``SummaryWriter`` wrapper; a no-op when TensorBoard isn't installed/enabled."""

    def __init__(self, log_dir: str | Path, enabled: bool = True):
        self.writer = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(log_dir))
            log.info("TensorBoard logging to %s", log_dir)
        except Exception as e:  # tensorboard not installed → degrade gracefully
            log.warning("TensorBoard disabled (%s)", e)

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        if not self.enabled:
            return
        for k, v in scalars.items():
            if v is not None and np.isfinite(v):
                self.writer.add_scalar(k, float(v), step)

    def log_retention(self, matrix: np.ndarray, task_idx: int) -> None:
        """Per-seen-task F1 after training ``task_idx`` → the live forgetting curves."""
        if not self.enabled:
            return
        for j in range(task_idx + 1):
            v = matrix[task_idx, j]
            if not np.isnan(v):
                self.writer.add_scalar(f"retention/task_{j}", float(v), task_idx)
        row = matrix[task_idx, : task_idx + 1]
        if np.any(~np.isnan(row)):
            self.writer.add_scalar("metrics/running_AA", float(np.nanmean(row)), task_idx)

    def log_forgetting_matrix(
        self, matrix: np.ndarray, step: int, task_names: list[str] | None = None
    ) -> None:
        if not self.enabled:
            return
        img = render_forgetting_heatmap(matrix, task_names)
        if img is not None:
            self.writer.add_image("diagnostic/forgetting_matrix", img, step, dataformats="HWC")
        self.writer.flush()

    def flush(self) -> None:
        if self.enabled:
            self.writer.flush()

    def close(self) -> None:
        if self.enabled:
            self.writer.flush()
            self.writer.close()
