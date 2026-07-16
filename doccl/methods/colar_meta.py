"""CoLaR-Meta — memory inside the weights: Benna–Fusi metaplastic consolidation + CoLaR.

M1 of the read-side-memory direction, the weight-DYNAMICS substrate: instead of storing
content, each plastic parameter carries a chain of m hidden variables with geometrically
slower timescales (Benna & Fusi, Nature Neuroscience 2016 — the beaker cascade). After
every optimizer step, adjacent levels diffuse toward each other; the deep levels act as a
slow anchor that damps fast drift of the weight without a second network copy or a
quadratic penalty. Composed WITH CoLaR replay (not a substitute): replay supplies real
past-task gradients, the cascade supplies inertia.

FALSIFICATION-RISK TRACK (flagged in the plan): EWC — the closest
protect-old-weights-with-per-weight-state precedent in this repo — already failed here.
This differs mechanically (dynamics, not penalty; anchors adapt rather than pin), but the
prior is against it. The ``meta_m=1`` control is byte-identical CoLaR and MUST reproduce
its numbers before any m>=2 result is read.

The consolidator is built lazily at the first task with task_id >= 1 — after the freeze
map is applied, so its state covers exactly the plastic bucket (layers >= k + head), and
task 0 (plain fine-tune, nothing to protect yet) runs untouched. Known simplification:
EarlyStopper's best-weight restore can desync weights from the chain at a task boundary;
the chain re-equilibrates within a few steps of the next task.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

from doccl.methods.colar import CoLaR
from doccl.types import TaskInfo, TrainMetrics

log = logging.getLogger(__name__)

__all__ = ["CoLaRMeta", "MetaplasticConsolidator"]


class MetaplasticConsolidator:
    """Per-parameter Benna–Fusi chain: level 0 IS the weight, levels 1..m-1 are buffers.

    Discretization: adjacent levels exchange at rate ``eta / g_base**i`` (pair i,i+1),
    and the deeper member of each pair moves ``g_base``x slower (wider beaker). m=1
    means no buffers and ``step()`` is a true no-op — the ablation control.
    """

    def __init__(
        self,
        params: list[nn.Parameter],
        m: int = 3,
        g_base: float = 2.0,
        eta: float = 0.01,  # ponytail: calibration knob — diffusion strength per step
    ) -> None:
        if m < 1:
            raise ValueError(f"meta_m must be >= 1, got {m}")
        self.params = params
        self.m = m
        self.g_base = float(g_base)
        self.eta = float(eta)
        # levels equilibrated at the current weights — a zero init would drag toward 0
        self.u: list[list[torch.Tensor]] = [[p.data.clone() for _ in range(m - 1)] for p in params]

    @torch.no_grad()
    def step(self) -> None:
        """Diffuse the chain after an optimizer step (which moved level 0)."""
        for p, chain in zip(self.params, self.u, strict=True):
            if not chain:
                continue
            levels = [p.data, *chain]
            flows = [levels[i] - levels[i + 1] for i in range(len(levels) - 1)]
            for i, flow in enumerate(flows):
                eta_i = self.eta / (self.g_base**i)
                levels[i].sub_(flow, alpha=eta_i)
                levels[i + 1].add_(flow, alpha=eta_i / self.g_base)

    def state_bytes(self) -> int:
        return sum(u.numel() * u.element_size() for chain in self.u for u in chain)


class CoLaRMeta(CoLaR):
    """CoLaR replay + multi-timescale metaplastic consolidation on the plastic bucket."""

    name = "colar_meta"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.meta_m = int(config.get("meta_m", 3))
        self.meta_g_base = float(config.get("meta_g_base", 2.0))
        self.meta_eta = float(config.get("meta_eta", 0.01))
        self._consolidator: MetaplasticConsolidator | None = None

    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        if task.task_id >= 1 and self._consolidator is None:
            self._consolidator = MetaplasticConsolidator(
                self.trainable_parameters(),
                m=self.meta_m,
                g_base=self.meta_g_base,
                eta=self.meta_eta,
            )
            log.info(
                "colar_meta: consolidator on %d plastic tensors, m=%d, state=%.0f MB",
                len(self._consolidator.params),
                self.meta_m,
                self._consolidator.state_bytes() / 1e6,
            )
        return super().train_task(task, train_loader, val_loader)

    def _post_optimizer_step(self) -> None:
        if self._consolidator is not None:
            self._consolidator.step()

    def consolidator_state_bytes(self) -> int:
        """Optimizer-state bytes — reported separately from memory_bytes() (the replay
        Pareto axis analyze_results.py parses); different budget, don't conflate."""
        return self._consolidator.state_bytes() if self._consolidator else 0
