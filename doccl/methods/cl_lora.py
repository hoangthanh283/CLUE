"""CL-LoRA — dual-adapter continual LoRA (``cl_lora``).

The 2025 "currency" member of the LoRA/PEFT family; the successor to O-LoRA.

Reference: He, Duan, Zhu, "CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free
Class-Incremental Learning", CVPR 2025 (arXiv:2505.24816).

CL-LoRA pairs a **task-shared** adapter (knowledge carried across tasks, regularised
so prior knowledge is retained) with a **task-specific** adapter (orthogonality-
constrained per-task subspace, exactly O-LoRA's mechanism). The original targets
CIL image classification with a prototype/NCM head; here it is adapted to
LayoutLMv3 **token-classification**:

    - task-specific subspace + orthogonality  → inherited verbatim from ``OLoRA``
      (per-task LoRA on Q/K/V, ``||A_t^T A_{<t}||_F^2`` penalty).
    - task-shared knowledge retention          → a LwF-style temperature-scaled KD
      term toward a frozen teacher (the model after the previous task), distilled
      over the *previously known* logits only (the ``n_old`` slice). This is the
      token-classification realisation of CL-LoRA's shared-adapter knowledge
      distillation, reusing the validated ``LwF`` teacher machinery.
    - gradient reassignment                     → ``shared_grad_scale`` down-weights
      the KD gradient (the shared-knowledge pull) so the task-specific subspace can
      still fit the new task; ``1.0`` = full KD, ``0.0`` = pure O-LoRA.

With ``kd_alpha=0`` it degrades exactly to O-LoRA, so the added mechanism is a clean
superset of the prior baseline.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.o_lora import OLoRA
from doccl.types import TaskInfo, TrainMetrics


class CLLoRA(OLoRA):
    """Dual-adapter continual LoRA: O-LoRA subspaces + shared-knowledge KD."""

    name = "cl_lora"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.kd_alpha = config.get("kd_alpha", 1.0)
        self.temperature = config.get("temperature", 2.0)
        # Gradient reassignment: scales the shared-knowledge (KD) pull. 1.0 = full,
        # 0.0 = pure O-LoRA. Lets the task-specific subspace dominate new-task fitting.
        self.shared_grad_scale = config.get("shared_grad_scale", 1.0)
        self.state.custom["teacher"] = None  # frozen previous-task model (set in after_task)

    def _kd_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Temperature-scaled KD over previously-known logits only (CIL-safe).

        Mirrors ``LwF._kd_loss`` / ``DocCL._kd_loss``: distil just the first
        ``n_old`` logit dims (the teacher predates the classifier-head growth), with
        ``-100`` positions masked.
        """
        T = self.temperature
        n_old = teacher_logits.shape[-1]
        student_old = student_logits[..., :n_old]
        valid = mask.unsqueeze(-1).expand_as(student_old)
        student_log = F.log_softmax(student_old / T, dim=-1)
        teacher_prob = F.softmax(teacher_logits / T, dim=-1)
        kd = F.kl_div(student_log, teacher_prob, reduction="none") * (T**2)
        return (kd * valid).sum() / valid.sum().clamp(min=1)

    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        self.model.train()
        teacher = self.state.custom.get("teacher")
        if teacher is not None:
            teacher.to(self.device)
            teacher.eval()
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.get("lr", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        stopper = self.make_early_stopper(val_loader)

        total_loss = 0.0
        n_steps = 0
        for epoch in range(epochs):
            pbar = tqdm(
                train_loader, desc=f"CL-LoRA T{task.task_id} ep{epoch+1}/{epochs}", leave=False
            )
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                optimizer.zero_grad()

                outputs = self.model(**batch)
                ce_loss = outputs.loss
                # Task-specific subspace: O-LoRA orthogonality penalty.
                ortho_loss = self.lambda_ortho * self._ortho_loss()

                # Task-shared knowledge: KD toward the frozen previous-task teacher,
                # gradient-reassigned by shared_grad_scale.
                kd_loss = torch.zeros((), device=self.device)
                if teacher is not None and self.kd_alpha > 0:
                    with torch.no_grad():
                        teacher_out = teacher(**{k: v for k, v in batch.items() if k != "labels"})
                    mask = batch["labels"] != -100
                    kd_loss = self._kd_loss(outputs.logits, teacher_out.logits, mask)
                    kd_loss = self.shared_grad_scale * self.kd_alpha * kd_loss

                loss = ce_loss + ortho_loss + kd_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], max_grad_norm
                )
                optimizer.step()

                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix(
                    {
                        "ce": f"{ce_loss.item():.3f}",
                        "ortho": f"{float(ortho_loss):.4f}",
                        "kd": f"{float(kd_loss):.3f}",
                    }
                )
            if self._early_stop_after_epoch(stopper, val_loader, task, epoch):
                break

        stopper.restore_best(self.model)
        return TrainMetrics(
            task_id=task.task_id,
            loss=total_loss / max(n_steps, 1),
            n_steps=n_steps,
        )

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Snapshot LoRA-A matrices (O-LoRA) AND a frozen teacher (shared-KD)."""
        super().after_task(task, train_loader)  # O-LoRA: snapshot past_A_matrices
        teacher = copy.deepcopy(self.model)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        if hasattr(teacher.model, "gradient_checkpointing_disable"):
            teacher.model.gradient_checkpointing_disable()
        self.state.custom["teacher"] = teacher
