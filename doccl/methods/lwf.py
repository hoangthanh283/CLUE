"""Learning without Forgetting (LwF).

Reference: Li & Hoiem, "Learning without Forgetting", ECCV 2016, TPAMI 2017.

Knowledge distillation from a frozen teacher (snapshot of model after previous task).
Loss: L_total = L_CE(current task) + α * KL(σ(z_student/T) || σ(z_teacher/T))
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo, TrainMetrics


class LwF(NaiveFineTune):
    """Learning without Forgetting via knowledge distillation."""

    name = "lwf"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.alpha = config.get("alpha", 1.0)
        self.temperature = config.get("temperature", 2.0)
        self.state.custom["teacher"] = None  # set in after_task

    def _kd_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """KL divergence between temperature-softened distributions.

        Only the *previously known* logits are distilled — when classifier is
        expanded for new task, new heads have no teacher target.
        """
        T = self.temperature
        n_old = teacher_logits.shape[-1]
        student_logits_old = student_logits[..., :n_old]

        # Mask out -100 positions
        valid = mask.unsqueeze(-1).expand_as(student_logits_old)
        student_log = F.log_softmax(student_logits_old / T, dim=-1)
        teacher_log = F.softmax(teacher_logits / T, dim=-1)

        kd = F.kl_div(student_log, teacher_log, reduction="none") * (T ** 2)
        kd = (kd * valid).sum() / valid.sum().clamp(min=1)
        return kd

    def train_task(self, task: TaskInfo, train_loader: DataLoader) -> TrainMetrics:
        self.model.train()
        teacher = self.state.custom.get("teacher")
        optimizer = torch.optim.AdamW(
            self.trainable_parameters(),
            lr=self.config.get("lr", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)

        total_loss = 0.0
        n_steps = 0
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"LwF T{task.task_id} ep{epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                optimizer.zero_grad()

                outputs = self.model(**batch)
                ce_loss = outputs.loss
                kd_loss = torch.zeros((), device=self.device)

                if teacher is not None:
                    with torch.no_grad():
                        teacher_out = teacher(
                            **{k: v for k, v in batch.items() if k != "labels"}
                        )
                    mask = batch["labels"] != -100
                    kd_loss = self._kd_loss(outputs.logits, teacher_out.logits, mask)

                loss = ce_loss + self.alpha * kd_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), max_grad_norm)
                optimizer.step()

                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix({"ce": f"{ce_loss.item():.3f}", "kd": f"{kd_loss.item():.3f}"})

        return TrainMetrics(
            task_id=task.task_id,
            loss=total_loss / max(n_steps, 1),
            n_steps=n_steps,
        )

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Snapshot current model as teacher for the next task."""
        teacher = copy.deepcopy(self.model)
        for p in teacher.parameters():
            p.requires_grad = False
        teacher.eval()
        self.state.custom["teacher"] = teacher
