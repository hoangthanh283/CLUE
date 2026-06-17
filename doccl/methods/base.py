"""Abstract base class for all continual learning methods.

The lifecycle is:
    for task in scenario.tasks:
        method.before_task(task)     # extend classifier, allocate adapters
        method.train_task(task)      # main training loop
        method.after_task(task)      # update buffer, compute Fisher, freeze adapters
        results = method.evaluate(eval_loaders_seen_so_far)

Subclasses override hooks they need; the rest are no-ops by default.
"""

from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from doccl.eval.metrics import compute_token_f1
from doccl.types import EvalMetrics, TaskInfo, TaskState, TrainMetrics

log = logging.getLogger(__name__)


class EarlyStopper:
    """Val-F1-based early stopping with best-weight restoration.

    After each training epoch a method calls ``step(val_f1, model)``. The
    stopper tracks the best validation F1 seen so far, snapshots the model's
    weights at that point, and returns ``True`` once ``patience`` consecutive
    epochs pass with no improvement. On stop (or at the end of training) the
    caller invokes ``restore_best(model)`` so the task ends on its best-val
    checkpoint rather than a possibly-overfit final epoch.

    Disabled (``enabled=False``) when no validation loader is available, in
    which case ``step`` always returns ``False`` and training runs the full
    epoch budget — preserving the previous fixed-budget behaviour.
    """

    def __init__(self, patience: int = 2, min_delta: float = 0.1, enabled: bool = True):
        # NOTE: min_delta is in F1 *points* on a 0-100 scale (compute_token_f1 returns
        # seqeval F1 * 100). 0.1 points is a sensible "real improvement" threshold; a
        # fractional value like 1e-4 would be swamped by batch-ordering noise (~0.01 pts)
        # and would reset the patience counter every epoch, defeating early stopping.
        self.patience = patience
        self.min_delta = min_delta
        self.enabled = enabled
        self.best_f1: float = -1.0
        self.best_epoch: int = -1
        self.best_state: dict[str, torch.Tensor] | None = None
        self.num_bad_epochs: int = 0

    def step(self, val_f1: float, model: nn.Module, epoch: int) -> bool:
        """Record this epoch's val-F1. Returns True if training should stop."""
        if not self.enabled:
            return False
        if val_f1 > self.best_f1 + self.min_delta:
            self.best_f1 = val_f1
            self.best_epoch = epoch
            # Snapshot on CPU to avoid holding a second model-sized copy on GPU.
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.num_bad_epochs = 0
            return False
        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Load the best-val weights back into the model (no-op if none kept)."""
        if self.best_state is None:
            if self.enabled:
                log.warning(
                    "EarlyStopper.restore_best: no snapshot saved (0 epochs?); "
                    "model left unchanged."
                )
            return
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device) for k, v in self.best_state.items()})
        self.best_state = None  # free the CPU snapshot


class ContinualMethod(ABC):
    """Base class for all CL methods. Subclasses override what they need."""

    name: str = "base"  # override in subclasses for W&B tagging

    def __init__(self, model: nn.Module, config: dict[str, Any]):
        self.model = model
        self.config = config
        self.state = TaskState()
        self.device = next(model.parameters()).device
        # Mixed precision (opt-in; set by train.py from cfg.training.amp). Inert by
        # default → existing runs are byte-identical. bf16 (no scaler) is preferred on
        # Ampere+, else fp16 with a GradScaler.
        self.amp_enabled: bool = False
        self._amp_dtype: torch.dtype | None = None
        self._amp_scaler: Any = None
        # Optional deep-diagnostic logger (set by train.py when
        # cfg.tensorboard.diagnostics is on). When present, the shared per-epoch
        # hook logs per-component / per-depth weight + grad histograms. Inert by
        # default so existing runs are byte-identical.
        self.tb_diag: Any = None
        self._epoch_counter: int = 0

    # ─── Lifecycle hooks ───────────────────────────────────────────────────────
    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Called once before training on a new task.

        Common uses:
            - Extend classifier head for class-incremental learning
            - Allocate new LoRA bank for parameter-efficient methods
            - Cache teacher model for distillation methods
        """
        pass

    @abstractmethod
    def train_task(
        self,
        task: TaskInfo,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> TrainMetrics:
        """Train on the current task. Returns training metrics.

        Args:
            task: current task descriptor.
            train_loader: batches for the current task.
            val_loader: held-out batches for the *current* task, used for
                val-F1 early stopping. If ``None`` (e.g. joint training), the
                method trains the full fixed epoch budget.
        """
        ...

    # ─── Early-stopping helpers (shared by all methods) ─────────────────────────
    def make_early_stopper(self, val_loader: DataLoader | None) -> EarlyStopper:
        """Construct an EarlyStopper from config; disabled if no val_loader."""
        return EarlyStopper(
            # min_delta in F1 points (0-100 scale) — see EarlyStopper.__init__.
            patience=int(self.config.get("early_stop_patience", 2)),
            min_delta=float(self.config.get("early_stop_min_delta", 0.1)),
            enabled=val_loader is not None and bool(self.config.get("early_stopping", True)),
        )

    def _early_stop_after_epoch(
        self,
        stopper: EarlyStopper,
        val_loader: DataLoader | None,
        task: TaskInfo,
        epoch: int,
    ) -> bool:
        """Evaluate val-F1 and update the stopper. Returns True to stop.

        One-line tail for every method's epoch loop. No-op (returns False) when
        early stopping is disabled or no val_loader is supplied.

        Doubles as the shared per-epoch diagnostic hook: when a ``tb_diag`` logger
        is attached, log per-component and per-depth *weight* histograms here (grads
        are method-local and already zeroed by this point). This runs once per epoch
        for every method without touching their individual loops.
        """
        self._log_weight_diagnostics()
        if not stopper.enabled or val_loader is None:
            return False
        val_f1 = self.current_task_val_f1(val_loader)
        should_stop = stopper.step(val_f1, self.model, epoch)
        log.info(
            "T%s ep%d val_f1=%.4f best=%.4f bad=%d%s",
            task.task_id,
            epoch + 1,
            val_f1,
            stopper.best_f1,
            stopper.num_bad_epochs,
            " -> STOP" if should_stop else "",
        )
        return should_stop

    def _log_weight_diagnostics(self) -> None:
        """Per-epoch weight histograms by component + depth (no-op without tb_diag).

        Reuses the model's ``param_groups`` / ``param_groups_by_depth`` so the
        forgetting story is bucketed identically to the Fisher/displacement plots.
        """
        if self.tb_diag is None:
            return
        step = self._epoch_counter
        self._epoch_counter += 1
        groups = getattr(self.model, "param_groups", None)
        depth = getattr(self.model, "param_groups_by_depth", None)
        if groups is not None:
            self.tb_diag.log_param_histograms(groups, "weights", step, which="weight")
        if depth is not None:
            self.tb_diag.log_param_histograms(depth, "weights_by_depth", step, which="weight")

    @torch.no_grad()
    def current_task_val_f1(self, val_loader: DataLoader) -> float:
        """Entity-F1 on the current task's held-out set (early-stop signal).

        Mirrors the prediction path in ``evaluate`` but for a single loader and
        restores ``train`` mode on exit so the caller's epoch loop is unaffected.
        """
        was_training = self.model.training
        self.model.eval()
        id_to_label = getattr(self.model, "id_to_label", None) or {
            i: str(i) for i in range(self.model.model.config.num_labels)
        }
        all_preds: list[int] = []
        all_labels: list[int] = []
        for batch in val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
            preds = outputs.logits.argmax(dim=-1)
            labels = batch["labels"]
            mask = labels != -100
            all_preds.extend(preds[mask].cpu().tolist())
            all_labels.extend(labels[mask].cpu().tolist())
        f1 = compute_token_f1(all_preds, all_labels, id_to_label)["f1"]
        if was_training:
            self.model.train()
        return f1

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Called once after training on a task.

        Common uses:
            - Update replay buffer (ER, DER++, iCaRL)
            - Compute Fisher information matrix (EWC)
            - Snapshot model as teacher (LwF)
            - Freeze current LoRA bank, allocate next
        """
        pass

    @abstractmethod
    def evaluate(self, eval_loaders: dict[int, DataLoader]) -> dict[int, EvalMetrics]:
        """Evaluate on all tasks seen so far.

        Args:
            eval_loaders: {task_id: DataLoader} for all tasks seen up to now.

        Returns:
            {task_id: EvalMetrics}
        """
        ...

    # ─── Mixed precision (opt-in; gated by self.amp_enabled) ────────────────────
    def _amp_setup(self) -> None:
        """Pick the AMP dtype + scaler for the current task. Call at train start.

        No-op unless ``amp_enabled`` and on CUDA. bf16 needs no loss scaling; fp16
        uses a fresh ``GradScaler``. Leaves everything None when disabled so the
        plain-precision path runs unchanged.
        """
        if not (self.amp_enabled and self.device.type == "cuda"):
            self._amp_dtype = None
            self._amp_scaler = None
            return
        if torch.cuda.is_bf16_supported():
            self._amp_dtype = torch.bfloat16
            self._amp_scaler = None
        else:
            self._amp_dtype = torch.float16
            self._amp_scaler = torch.cuda.amp.GradScaler()

    def _amp_autocast(self):
        """Autocast context for the forward+loss; nullcontext when AMP is disabled."""
        if self.amp_enabled and self.device.type == "cuda" and self._amp_dtype is not None:
            return torch.autocast(device_type="cuda", dtype=self._amp_dtype)
        return contextlib.nullcontext()

    def _amp_backward_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        params: list[nn.Parameter],
        max_grad_norm: float,
    ) -> None:
        """Backward + grad-clip + optimizer step, with fp16 loss scaling if active.

        When AMP is disabled (or bf16, which needs no scaler) this is exactly the
        plain ``loss.backward(); clip; optimizer.step()`` path.
        """
        if self._amp_scaler is not None:
            self._amp_scaler.scale(loss).backward()
            self._amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            self._amp_scaler.step(optimizer)
            self._amp_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()

    # ─── Helpers ───────────────────────────────────────────────────────────────
    def trainable_parameters(self) -> list[nn.Parameter]:
        """All parameters with requires_grad=True (default optimizer target)."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def state_dict(self) -> dict[str, Any]:
        """Serialize method state for checkpointing.

        Subclasses override to save buffers, prompt pools, Fisher matrices, etc.
        """
        return {
            "name": self.name,
            "config": self.config,
            "model_state": self.model.state_dict(),
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        """Restore method state from checkpoint."""
        self.model.load_state_dict(sd["model_state"])
