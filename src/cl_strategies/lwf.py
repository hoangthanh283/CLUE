"""Learning without Forgetting (LwF).

This is a vanilla implementation of Learning without Forgetting following:
    Li & Hoiem (2016) "Learning without Forgetting" (ECCV 2016)

The algorithm uses knowledge distillation to preserve old task knowledge
while learning new tasks, without storing any old task data:

    L_total = α * L_new + (1-α) * T² * KL(student || teacher)

Where:
    - L_new: Cross-entropy loss on new task data
    - KL: KL divergence between soft teacher and student predictions
    - T: Temperature for softening distributions (typical: 2.0)
    - α: Balance between new learning and knowledge retention

IMPORTANT: Requires unified label space (label_space.unified: true in config)
to ensure teacher and student have compatible output dimensions.

This implementation is suitable as a baseline for continual learning research.
"""

import copy
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from src.cl_strategies.base import BaseCLStrategy
from src.config import LwFConfig


class LwF(BaseCLStrategy):
    """Vanilla Learning without Forgetting with knowledge distillation.

    References:
        Li & Hoiem (2016) "Learning without Forgetting"
        https://arxiv.org/abs/1606.09282

    Algorithm:
        1. After training on task T, create frozen teacher = deepcopy(model)
        2. When training on task T+1:
           - Compute loss on new data: L_new
           - Compute KL divergence between student and teacher soft predictions
           - Total loss: α * L_new + (1-α) * T² * L_distill

    Key Features:
        - No storage of old task data (only model weights)
        - Knowledge distillation with temperature scaling
        - Frozen teacher model from previous task
        - Token-level distillation with attention masking

    Requirements:
        - Unified label space across all tasks (same output dimension)
        - Teacher and student must have identical architecture

    This is a basic implementation following the original paper, suitable
    as a vanilla baseline for comparisons.
    """

    def __init__(self, config: Union[LwFConfig, Dict[str, Any]]):
        super().__init__(config)
        if isinstance(config, dict):
            cl_cfg = config.get("cl_strategy", {})
            self.alpha = float(cl_cfg.get("lwf_alpha", 0.5))
            self.temperature = float(cl_cfg.get("lwf_temperature", 2.0))
            if not config.get("label_space", {}).get("unified", False):
                raise ValueError(
                    "LwF requires unified label space! "
                    "Set 'label_space.unified: true' in config. "
                    "This ensures teacher and student have same output dimensions."
                )
        else:
            self.alpha = config.lwf_alpha
            self.temperature = config.lwf_temperature
            # Validation already done in LwFConfig.__post_init__

        self.teacher: Optional[nn.Module] = None
        self.kldiv = nn.KLDivLoss(reduction="batchmean")

    def before_task(self, model: nn.Module, task_id: int, train_loader=None) -> None:
        """Prepare for new task by creating teacher from current model.

        Args:
            model: Current model (student) to be trained on new task.
            task_id: Index of the upcoming task (0-indexed).
            train_loader: Not used by LwF.

        Notes:
            - For task 0 (first task): No teacher, train normally
            - For task > 0: Create frozen teacher = deepcopy(current model)
            - Teacher captures knowledge from all previous tasks
        """
        super().before_task(model, task_id, train_loader)
        if task_id == 0:
            self.teacher = None  # No previous knowledge for first task
        else:
            # Create frozen teacher from current model (trained on previous tasks)
            self.teacher = copy.deepcopy(model).eval()
            for p in self.teacher.parameters():
                p.requires_grad = False  # Freeze teacher weights

    def compute_loss(self, model: nn.Module, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]
                     ) -> torch.Tensor:
        """Compute combined loss: new task loss + distillation loss.

        Args:
            model: Current model (student) being trained.
            batch: Input batch with keys like 'input_ids', 'attention_mask', 'labels'.
            outputs: Model outputs with 'loss' and 'logits' keys.

        Returns:
            Combined loss: α * L_new + (1-α) * T² * L_distill

        Formula:
            L_new = Cross-entropy loss on current task
            L_distill = KL(student_soft || teacher_soft)
            T² scaling compensates for gradient scaling with temperature

        Notes:
            - First task (no teacher): Returns L_new only
            - Subsequent tasks: Combines L_new with distillation loss
            - Teacher predictions are detached (no gradient flow)
            - Padding tokens excluded via attention mask
        """
        base_loss = outputs["loss"]
        if self.teacher is None:
            return base_loss  # First task: no distillation

        T = self.temperature
        student_logits = outputs["logits"]

        # Get teacher predictions (frozen, no gradients)
        with torch.no_grad():
            teacher_out = self.teacher(**batch)
            teacher_logits = teacher_out["logits"].detach()

        # Apply temperature scaling and compute soft distributions
        student_log_probs = torch.log_softmax(student_logits / T, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / T, dim=-1)

        # Compute KL divergence, excluding padding tokens if present
        attn = batch.get("attention_mask")
        if attn is None:
            # No mask: use all tokens
            kd_loss = self.kldiv(
                student_log_probs.view(-1, student_log_probs.size(-1)),
                teacher_probs.view(-1, teacher_probs.size(-1)),
            )
        else:
            # With mask: exclude padding tokens (mask==0)
            mask = attn.view(-1) == 1
            kd_loss = self.kldiv(
                student_log_probs.view(-1, student_log_probs.size(-1))[mask],
                teacher_probs.view(-1, teacher_probs.size(-1))[mask],
            )

        # Combine losses with T² scaling for distillation
        # α controls balance: higher α = more focus on new task
        return self.alpha * base_loss + (1.0 - self.alpha) * (T * T) * kd_loss
