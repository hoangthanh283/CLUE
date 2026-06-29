"""MagMax — model-merging continual learning via max-magnitude task-vector selection.

Reference: Marczak, Twardowski, Trzciński, Cygert, "MagMax: Leveraging Model Merging for
Seamless Continual Learning", ECCV 2024 (arXiv:2407.06322). MagMax sequentially fine-tunes
the backbone across tasks, forms each task vector ``τ_t = θ_t − θ_0`` relative to the
pre-trained init, and combines them with **maximum-magnitude selection**: for every
parameter coordinate, keep the single task-vector entry with the largest absolute value,
then set ``θ* = θ_0 + λ · MaxMag({τ_1, …, τ_t})``. No exemplars, no rehearsal, no
regularizer during training — the entire CL mechanism lives in the offline merge.

This is the weight-merging baseline the proposed methods (DocMERGE / LCA / DocCL) must beat.
It is deliberately the *merge-only* control: unlike ``lca`` it performs **no** classifier
realignment after merging, which isolates two things at once — (i) the value of the merge
operator (max-magnitude vs LCA's TIES) and (ii) the value of LCA's post-merge head
realignment (MagMax omits it, so the gap to ``lca`` is exactly that realignment's worth).

Token-level adaptation (faithful in spirit):
  * MagMax targets image-classification ViTs with a fixed head; here the head **grows**
    (CIL). Following the repo's ``lca`` convention, the max-magnitude merge is applied to
    the **backbone only** (``"classifier" not in name``); the growing token-classification
    head is carried forward as trained (old logit rows preserved by ``expand_classifier``).
    MagMax has no head-correction step by design, so this is the faithful merge-only variant.
  * Merge runs at every task boundary (not just the end) so the CL retention matrix has a
    consolidated model to evaluate after each task. The merged backbone is loaded back, so
    the next task continues from the merged anchor (matching ``lca``'s lifecycle).

``merge_method`` defaults to ``max_abs`` (the MagMax operator); ``max`` / ``min`` / ``ties``
are accepted for ablation (e.g. ``ties`` = a task-arithmetic / TIES-merge control without
LCA's realignment), all served by the shared :func:`merge_state_dicts`.
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from doccl.methods.naive import NaiveFineTune
from doccl.methods.ties_merge import merge_state_dicts
from doccl.types import TaskInfo

log = logging.getLogger(__name__)

__all__ = ["MagMax"]


class MagMax(NaiveFineTune):
    """MagMax max-magnitude model-merging CL (see module docstring).

    Inherits :class:`NaiveFineTune`'s plain fine-tuning ``train_task`` (MagMax fine-tunes
    each task with no extra loss term) and its standard-forward ``evaluate`` (single growing
    head). Only the merge lifecycle (``before_task`` / ``after_task``) is added.
    """

    name = "magmax"

    def __init__(self, model, config):
        super().__init__(model, config)
        # Merge operator: ``max_abs`` is MagMax's max-magnitude selection (default).
        self.merge_method = config.get("merge_method", "max_abs")
        # λ scaling on the merged task vector (MagMax sweeps this; 1.0 is the headline value).
        self.merge_coef = float(config.get("merge_coef", 1.0))
        # TIES keep-% — only used when merge_method == "ties" (100 = no trim).
        self.merge_topk = int(config.get("merge_topk", 100))

        self._base_backbone: dict[str, torch.Tensor] | None = None  # θ_0 (pre-task anchor)
        self._task_backbones: list[dict[str, torch.Tensor]] = []  # per-task θ_t (backbone)

    def _backbone_state(self) -> dict[str, torch.Tensor]:
        """Snapshot backbone params (everything but the classifier head) on CPU."""
        return {
            n: p.detach().clone().cpu()
            for n, p in self.model.model.named_parameters()
            if "classifier" not in n
        }

    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        # θ_0 captured once, before any task vector is formed. The CL loop has already
        # expanded the head for this task; the backbone is still the pre-training init.
        if self._base_backbone is None:
            self._base_backbone = self._backbone_state()

    # train_task: inherited from NaiveFineTune (plain AdamW FT, early stopping) — MagMax
    # adds no training-time term; the CL mechanism is the after_task merge.

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Snapshot this task's backbone, then max-magnitude merge all task vectors onto θ_0."""
        self._task_backbones.append(self._backbone_state())
        self._merge_backbones()
        log.info(
            "magmax: task %d done; merged %d task vectors via %r (lamb=%.3g)",
            task.task_id,
            len(self._task_backbones),
            self.merge_method,
            self.merge_coef,
        )

    @torch.no_grad()
    def _merge_backbones(self) -> None:
        """θ* = θ_0 + λ · MaxMag({τ_1..τ_t}); load the merged backbone into the live model."""
        if self._base_backbone is None or not self._task_backbones:
            return
        merged = merge_state_dicts(
            self._base_backbone,
            self._task_backbones,
            method=self.merge_method,
            lamb=self.merge_coef,
            topk=self.merge_topk,
        )
        own = dict(self.model.model.named_parameters())
        for name, val in merged.items():
            if name in own:
                own[name].data.copy_(val.to(own[name].device))
