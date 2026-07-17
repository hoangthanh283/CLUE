"""Marginal-anchored training objectives — RCA kill-tests #4/#5.

The RCA (docs/RCA_FORGETTING_BASELINES_2026-07.md) showed forgetting is a readout-marginal
snap: the head's output marginal on old tasks realigns to the just-trained task's gold
label marginal. Both methods here store ONLY each task's gold label marginal (num_labels
floats per task — no documents, no features) and reshape the training loss so the snap is
never learned. Pre-registered decision rules: docs/RCA_KILLTESTS_PREREG_2026-07.md.

MarginalAnchor  (``marginal_kl``):  CE + λ·KL(mixture-of-seen-marginals ‖ batch output
    marginal). H4′-vs-H2 disentangler — expected to fix O/VALUE confusion, not KEY/HEADER
    extinction.
LogitAdjust  (``logit_adjust``):  balanced-softmax CE on logits + τ(log q_task − log q_cum),
    so raw logits estimate the posterior under the CUMULATIVE seen prior rather than the
    current task's (Menon et al. ICLR'21 logit adjustment / Ren et al. NeurIPS'20 balanced
    softmax, transplanted to sequential DIL). Predict with raw logits.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo

_EPS = 1e-8


def gold_marginal(train_loader: DataLoader, num_labels: int) -> torch.Tensor:
    """Empirical label marginal over valid (!=-100) tokens; one CPU pass, no model."""
    counts = torch.zeros(num_labels, dtype=torch.float64)
    for batch in train_loader:
        labels = batch["labels"]
        valid = labels[labels != -100]
        counts += torch.bincount(valid.flatten(), minlength=num_labels).to(torch.float64)
    return (counts / counts.sum().clamp(min=1)).to(torch.float32)


class MarginalAnchor(NaiveFineTune):
    """CE + KL anchor pulling the batch output marginal toward the seen-task mixture."""

    name = "marginal_kl"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.task_marginals: list[torch.Tensor] = []

    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        n = self.model.model.config.num_labels
        self.task_marginals.append(gold_marginal(train_loader, n))

    def _mixture(self) -> torch.Tensor:
        return torch.stack(self.task_marginals).mean(0)

    def _loss(self, outputs, batch) -> torch.Tensor:
        labels = batch["labels"]
        valid = labels != -100
        if not valid.any() or len(self.task_marginals) == 0:
            return outputs.loss
        probs = F.softmax(outputs.logits[valid], dim=-1)
        batch_marginal = probs.mean(0).clamp(min=_EPS)
        target = self._mixture().to(batch_marginal.device).clamp(min=_EPS)
        # KL(target || batch_marginal): mode-covering — punishes zeroing any class the
        # seen-task mixture still contains (the snap's signature failure).
        kl = (target * (target.log() - batch_marginal.log())).sum()
        return outputs.loss + self.config.get("lambda_kl", 1.0) * kl


class LogitAdjust(NaiveFineTune):
    """Balanced-softmax CE under the cumulative seen prior. Buffer-free anti-snap."""

    name = "logit_adjust"

    def __init__(self, model, config):
        super().__init__(model, config)
        self._counts: torch.Tensor | None = None  # cumulative label counts
        self._task_marginal: torch.Tensor | None = None

    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        n = self.model.model.config.num_labels
        marginal = gold_marginal(train_loader, n)
        n_tokens = sum(int((b["labels"] != -100).sum()) for b in train_loader)
        counts = marginal.to(torch.float64) * n_tokens
        self._counts = counts if self._counts is None else self._counts + counts
        self._task_marginal = marginal

    def _loss(self, outputs, batch) -> torch.Tensor:
        labels = batch["labels"]
        q_task = self._task_marginal
        if q_task is None:
            return outputs.loss
        q_cum = (self._counts / self._counts.sum()).to(torch.float32)
        device = outputs.logits.device
        # Train on logits + τ(log q_task − log q_cum) ⇒ raw logits estimate the posterior
        # under q_cum (see module docstring); prediction path stays untouched.
        tau = self.config.get("tau", 1.0)
        adjust = tau * ((q_task.clamp(min=_EPS)).log() - (q_cum.clamp(min=_EPS)).log()).to(device)
        logits = outputs.logits + adjust
        return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
