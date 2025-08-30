"""Learning without Forgetting (LwF)."""

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.cl_strategies.base import BaseCLStrategy


class LwF(BaseCLStrategy):
    """Distillation-based CL with a frozen teacher from previous task."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        cl_cfg = config.get("cl_strategy", {})
        self.alpha = float(cl_cfg.get("lwf_alpha", 0.5))
        self.temperature = float(cl_cfg.get("lwf_temperature", 2.0))
        self.teacher: Optional[nn.Module] = None
        self.kldiv = nn.KLDivLoss(reduction="batchmean")

    def before_task(self, model: nn.Module, task_id: int, train_loader=None):
        if task_id == 0:
            self.teacher = None
        else:
            self.teacher = copy.deepcopy(model).eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

    def compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]
                     ) -> torch.Tensor:
        base_loss = outputs["loss"]
        if self.teacher is None:
            return base_loss

        T = self.temperature
        student_logits = outputs["logits"]
        with torch.no_grad():
            teacher_out = self.teacher(**batch)
            teacher_logits = teacher_out["logits"].detach()

        student_log_probs = torch.log_softmax(student_logits / T, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / T, dim=-1)

        attn = batch.get("attention_mask")
        if attn is None:
            kd_loss = self.kldiv(
                student_log_probs.view(-1, student_log_probs.size(-1)),
                teacher_probs.view(-1, teacher_probs.size(-1)),
            )
        else:
            mask = attn.view(-1) == 1
            kd_loss = self.kldiv(
                student_log_probs.view(-1, student_log_probs.size(-1))[mask],
                teacher_probs.view(-1, teacher_probs.size(-1))[mask],
            )

        return self.alpha * base_loss + (1.0 - self.alpha) * (T * T) * kd_loss
