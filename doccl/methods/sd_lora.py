"""SD-LoRA — Scalable Decoupled LoRA for class-incremental learning.

Reference: Wu et al., "SD-LoRA: Scalable Decoupled Low-Rank Adaptation for Class
Incremental Learning", ICLR 2025 (Oral), arXiv:2501.13198; code WuYichen-97/SD-Lora-CL.

The idea is to **decouple** each task's LoRA update into a frozen *direction* and a
*learnable scalar magnitude*. The backbone is frozen; per task t we add one LoRA pair
``(A_t, B_t)`` to the Q and V projections. Once a task is done its direction is frozen and
L2-normalized so its scalar owns all the scale; the per-task scalars ``α_1…α_t`` are **all
re-trainable every task**, letting the model re-weight old directions as new ones arrive.
The adapted projection weight is

    W = W0 + Σ_{i=1..t}  α_i · (B_i A_i) / (‖B_i‖_F · ‖A_i‖_F)

(implemented functionally on the activation, never materializing ΔW). Rehearsal-free, and
**task-id-free at inference**: all task directions are summed into one forward, no router,
no per-task gating — a clean fit for CIL token IE.

Faithful-port notes (vs the image-CIL original):
  * Targets **Q and V only** (not K) — matches the reference's qkv injection.
  * Loss is **plain token cross-entropy** (the released SD-LoRA's orthogonality term is
    commented out; stability comes from the architecture, not the loss). We deliberately do
    NOT copy the reference's ``logits[:, known:]`` class-slice CE masking — the CL loop's
    ``CIL_LabelRemapper`` already aligns targets to the growing head, so we use full-head CE
    via the wrapper's ``outputs.loss``.
  * The reference normalizes previous directions but leaves the *current* task's term
    un-normalized **during training** (α_t co-adapts with raw ‖B_t A_t‖), then normalizes it
    at eval once it has become a stored direction. We reproduce this exactly by gating on
    ``self.training`` inside :class:`SDLoraLinear` — so the test-time weight is the clean
    normalized sum over all i ≤ t, while training keeps the reference's co-adaptation.
  * No NCM/prototype head: SD-LoRA's head is a plain growing ``Linear`` (same as ours); the
    reference's disk-persistence / head-averaging plumbing is process-restart cruft we skip.
  * peft cannot express a *learnable per-adapter scalar* with ``(BA)/(‖B‖‖A‖)`` normalization
    summed over all adapters, so we inject a custom module instead of using ``get_peft_model``.

Efficiency variants SD-LoRA-RR (rank taper) and SD-LoRA-KD (magnitude-threshold direction
drop) are for 10–20-task ImageNet runs; our scenarios have ~3–6 tasks, so base SD-LoRA at
r=10 is ported and the variants are deferred.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo, TrainMetrics

log = logging.getLogger(__name__)

__all__ = ["SDLoRA", "SDLoraLinear"]

_NORM_EPS = 1e-8


class SDLoraLinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with SD-LoRA's decoupled per-task direction/magnitude.

    Holds one ``(A_i, B_i, α_i)`` triplet per task in parallel ``ParameterList``s. Previous
    directions are frozen and L2-normalized; the current direction is trainable and (during
    training only) un-normalized. All scalars ``α_i`` stay trainable every task.
    """

    def __init__(self, orig: nn.Linear, rank: int = 10, alpha_init: float = 0.8) -> None:
        super().__init__()
        self.orig = orig  # frozen base projection W0 (its params are frozen by the method)
        self.rank = int(rank)
        self.alpha_init = float(alpha_init)
        self.in_features = orig.in_features
        self.out_features = orig.out_features
        # Per-task banks; task i is "current" iff i == len(self.alphas) - 1.
        self.A_banks = nn.ParameterList()  # each (r, in)
        self.B_banks = nn.ParameterList()  # each (out, r)
        self.alphas = nn.ParameterList()  # each scalar (1,)

    @property
    def n_tasks(self) -> int:
        return len(self.alphas)

    def add_task(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        """Append a fresh trainable direction + scalar; callers freeze prior directions first."""
        dev = device if device is not None else self.orig.weight.device
        dt = dtype if dtype is not None else self.orig.weight.dtype
        a = nn.Parameter(torch.empty(self.rank, self.in_features, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))  # standard LoRA A init
        b = nn.Parameter(
            torch.zeros(self.out_features, self.rank, device=dev, dtype=dt)
        )  # B=0 → no-op start
        alpha = nn.Parameter(torch.full((1,), self.alpha_init, device=dev, dtype=dt))
        self.A_banks.append(a)
        self.B_banks.append(b)
        self.alphas.append(alpha)

    def freeze_current_direction(self) -> None:
        """Freeze the most-recent ``(A, B)`` direction (scalars stay trainable)."""
        if self.n_tasks == 0:
            return
        self.A_banks[-1].requires_grad_(False)
        self.B_banks[-1].requires_grad_(False)

    def _delta(self, x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # B (A x): (..., in) @ (in, r) -> (..., r) @ (r, out) -> (..., out)
        return (x @ a.transpose(0, 1)) @ b.transpose(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.orig(x)
        if self.n_tasks == 0:
            return y
        # Previous tasks: frozen, L2-normalized directions, scaled by their learnable α.
        for i in range(self.n_tasks - 1):
            a, b, alpha = self.A_banks[i], self.B_banks[i], self.alphas[i]
            norm = (b.norm() * a.norm()).clamp_min(_NORM_EPS)
            y = y + alpha * self._delta(x, a, b) / norm
        # Current task: un-normalized while training (α co-adapts with raw ‖BA‖); normalized
        # at eval, where it has effectively become a stored direction (reference behavior).
        a, b, alpha = self.A_banks[-1], self.B_banks[-1], self.alphas[-1]
        cur = self._delta(x, a, b)
        if self.training:
            y = y + alpha * cur
        else:
            norm = (b.norm() * a.norm()).clamp_min(_NORM_EPS)
            y = y + alpha * cur / norm
        return y


class SDLoRA(NaiveFineTune):
    """SD-LoRA — decoupled-magnitude LoRA CL (see module docstring)."""

    name = "sd_lora"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.rank = int(config.get("lora_rank", 10))
        self.alpha_init = float(config.get("alpha_init", 0.8))
        self.target_modules = set(config.get("target_modules", ["query", "value"]))

        # Inject SD-LoRA into Q/V projections, then freeze the backbone (head stays trainable).
        self.sd_layers: list[SDLoraLinear] = []
        self._inject()
        self._freeze_backbone_keep_head()

    # ─── injection / freezing ────────────────────────────────────────────────────
    def _inject(self) -> None:
        root = self.model.model
        targets = [
            (name, mod)
            for name, mod in root.named_modules()
            if isinstance(mod, nn.Linear) and name.split(".")[-1] in self.target_modules
        ]
        for name, mod in targets:
            wrapped = SDLoraLinear(mod, rank=self.rank, alpha_init=self.alpha_init)
            self._set_submodule(root, name, wrapped)
            self.sd_layers.append(wrapped)
        log.info(
            "sd_lora: injected into %d projections (%s)",
            len(self.sd_layers),
            sorted(self.target_modules),
        )

    @staticmethod
    def _set_submodule(root: nn.Module, dotted: str, new: nn.Module) -> None:
        parts = dotted.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new)

    def _freeze_backbone_keep_head(self) -> None:
        # Freeze everything (incl. the wrapped orig projections, which have no banks yet);
        # keep the growing classifier head trainable. Banks are added later (requires_grad=True).
        for n, p in self.model.model.named_parameters():
            p.requires_grad_("classifier" in n)

    # ─── lifecycle ───────────────────────────────────────────────────────────────
    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        # The CL loop has already expanded the head for this task. Freeze every prior
        # direction, then add this task's fresh (A, B, α). All scalars remain trainable.
        for layer in self.sd_layers:
            layer.freeze_current_direction()
            layer.add_task(device=self.device)

    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        self.model.train()
        # Trainable: current (A_t, B_t) + all scalars α + the classifier head.
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.get("lr", 1e-3),
            weight_decay=self.config.get("weight_decay", 0.0),
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        stopper = self.make_early_stopper(val_loader)

        total_loss = 0.0
        n_steps = 0
        for epoch in range(epochs):
            pbar = tqdm(
                train_loader, desc=f"SDLoRA T{task.task_id} ep{epoch+1}/{epochs}", leave=False
            )
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss  # plain token CE (no ortho term in released SD-LoRA)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], max_grad_norm
                )
                optimizer.step()
                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix({"ce": f"{loss.item():.3f}"})
            if self._early_stop_after_epoch(stopper, val_loader, task, epoch):
                break

        stopper.restore_best(self.model)
        return TrainMetrics(
            task_id=task.task_id, loss=total_loss / max(n_steps, 1), n_steps=n_steps
        )

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        # Freeze this task's direction; its scalar stays trainable for future re-weighting.
        for layer in self.sd_layers:
            layer.freeze_current_direction()
        log.info(
            "sd_lora: task %d done; %d directions held",
            task.task_id,
            len(self.sd_layers[0].alphas) if self.sd_layers else 0,
        )

    # evaluate: inherited from NaiveFineTune (standard forward; SDLoraLinear sums all task
    # directions, and in eval mode the current term is normalized → faithful test-time W).
