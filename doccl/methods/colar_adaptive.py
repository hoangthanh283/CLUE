"""CA-CoLaR — coverage-adaptive CoLaR: per-task (docs, rank) bank allocation.

T1 measurement (docs/CACOLAR_T1_T2_2026-08-20.md): the replay dials are separable and
task-heterogeneous — FUNSD-like tasks are coverage-hungry (+37.97 F1 from d5→d50 at r64)
while SROIE-like tasks are fidelity-hungry (+14.49 from r64→r128 at d50), and the final
task's bank is never replayed within the stream. CA-CoLaR therefore spends a fixed
doc·rank byte budget non-uniformly: ``docs_schedule[t]`` / ``rank_schedule[t]`` set the
bank shape per task. Everything else — training path, capture, replay — is inherited
unchanged from CoLaR; with empty schedules this class is byte-identical to ``colar``.
"""

from __future__ import annotations

import logging

from torch.utils.data import DataLoader

from doccl.methods.base import TaskInfo
from doccl.methods.colar import CoLaR

log = logging.getLogger(__name__)

__all__ = ["CoLaRAdaptive"]


class CoLaRAdaptive(CoLaR):
    """CoLaR with preregistered per-task (docs_per_task, rank_r) schedules."""

    name = "colar_adaptive"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.docs_schedule = [int(x) for x in (config.get("docs_schedule") or [])]
        self.rank_schedule = [int(x) for x in (config.get("rank_schedule") or [])]

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        t = task.task_id
        if t < len(self.docs_schedule):
            self.docs_per_task = self.docs_schedule[t]
        if t < len(self.rank_schedule):
            self.rank_r = self.rank_schedule[t]
        log.info(
            "colar_adaptive: task %d banks docs=%d rank=%d", t, self.docs_per_task, self.rank_r
        )
        super().after_task(task, train_loader)
