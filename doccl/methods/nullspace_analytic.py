"""Null-Space Analytic CL — hierarchical lexical memory + gradient projection + analytic head.

Core idea: store (feature, position, label) tuples per (task, class) in a hierarchical
memory. During backbone training, project gradients to prevent feature drift on stored
samples. After training, compute head weights analytically via least squares.

Two projection modes:
- ``surgery`` (default): PCGrad-style — remove conflicting component between task
  gradient and feature-preservation gradient. Fast, practical.
- ``off``: no projection (ablation control — backbone drifts freely, only analytic head).

The analytic head update solves min ||X·W - Y||² over all stored features, giving
zero forgetting on stored samples by construction (when features are stable).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo, TrainMetrics

log = logging.getLogger(__name__)

__all__ = ["NullSpaceAnalyticCL"]


class HierarchicalLexicalMemory:
    """Dict-based memory: {task_id: {class_name: [(feat, pos, label), ...]}}.

    Stores classifier-input features (hidden_size-dim), bounding boxes (4-dim),
    and label indices. Also stores raw batch inputs separately for re-extraction.
    Stratified sampling per class to handle imbalance.
    """

    def __init__(self, samples_per_class: int = 50):
        self.samples_per_class = samples_per_class
        self.memory: dict[int, dict[str, list[tuple]]] = {}
        self.raw_batches: dict[int, list[dict]] = {}

    def store(
        self,
        task_id: int,
        features: torch.Tensor,
        positions: torch.Tensor,
        labels: torch.Tensor,
        id_to_label: dict[int, str],
        raw_batches: list[dict] | None = None,
    ) -> None:
        task_mem: dict[str, list[tuple]] = {}
        unique_labels = labels.unique()
        for lbl in unique_labels:
            if lbl.item() == -100:
                continue
            mask = labels == lbl
            feats_cls = features[mask]
            pos_cls = positions[mask]
            n_take = min(feats_cls.shape[0], self.samples_per_class)
            if n_take == 0:
                continue
            idx = torch.randperm(feats_cls.shape[0])[:n_take]
            class_name = id_to_label.get(lbl.item(), str(lbl.item()))
            task_mem[class_name] = [
                (feats_cls[j].cpu().half(), pos_cls[j].cpu().half(), lbl.item()) for j in idx
            ]
        self.memory[task_id] = task_mem
        if raw_batches:
            self.raw_batches[task_id] = raw_batches
        n_total = sum(len(v) for v in task_mem.values())
        log.info(
            "memory: task %d stored %d tuples across %d classes, %d raw batches",
            task_id,
            n_total,
            len(task_mem),
            len(raw_batches) if raw_batches else 0,
        )

    def get_all(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats, positions, labels = [], [], []
        for task_mem in self.memory.values():
            for class_tuples in task_mem.values():
                for item in class_tuples:
                    feat, pos, lbl = item[0], item[1], item[2]
                    feats.append(feat.float())
                    positions.append(pos.float())
                    labels.append(lbl)
        if not feats:
            return torch.empty(0), torch.empty(0), torch.empty(0)
        return torch.stack(feats), torch.stack(positions), torch.tensor(labels)

    def get_all_raw_batches(self) -> list[dict]:
        """Return all stored raw batches across all tasks."""
        all_batches = []
        for batches in self.raw_batches.values():
            all_batches.extend(batches)
        return all_batches

    def has_data(self) -> bool:
        return bool(self.memory)

    def memory_bytes(self) -> int:
        total = 0
        for task_mem in self.memory.values():
            for class_tuples in task_mem.values():
                for item in class_tuples:
                    feat, pos = item[0], item[1]
                    total += feat.nelement() * feat.element_size()
                    total += pos.nelement() * pos.element_size()
                    total += 4
        return total


class NullSpaceAnalyticCL(NaiveFineTune):
    """Null-space projection during backbone training + analytic head update."""

    name = "nullspace_analytic"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.samples_per_class = int(config.get("samples_per_class", 50))
        self.projection_freq = int(config.get("projection_freq", 10))
        self.projection_mode = str(config.get("projection_mode", "surgery"))
        self.soft_lambda = float(config.get("soft_lambda", 0.01))
        self.freeze_backbone_after_t0 = bool(config.get("freeze_backbone_after_t0", False))
        self.probe_batches = int(config.get("probe_batches", 2))
        self.analytic_head = bool(config.get("analytic_head", True))

        self.memory = HierarchicalLexicalMemory(self.samples_per_class)
        self._head_frozen = False
        self._cur_feats: torch.Tensor | None = None
        self._hook_handle = None
        self._register_head_hook()
        self._probe_cache: list[dict] = []
        self._probe_feats0: torch.Tensor | None = None
        self._capture_grad: bool = False

    def _register_head_hook(self) -> None:
        cls = self.model.model.classifier
        self._hook_handle = cls.register_forward_pre_hook(self._capture_feats)

    def _capture_feats(self, _module, inp):
        self._cur_feats = inp[0] if self._capture_grad else inp[0].detach()

    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Re-register hook after classifier expansion (expand_classifier orphans it)."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
        self._register_head_hook()

    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        if task.task_id == 0:
            return super().train_task(task, train_loader, val_loader)
        if self.freeze_backbone_after_t0:
            return self._train_head_only(task, train_loader, val_loader)
        return self._train_with_projection(task, train_loader, val_loader)

    def _train_head_only(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None
    ) -> TrainMetrics:
        """Backbone frozen — skip backbone training, analytic head update in after_task."""
        return TrainMetrics(task_id=task.task_id, loss=0.0, n_steps=0)

    def _train_with_projection(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None
    ) -> TrainMetrics:
        self.model.train()
        params = self.trainable_parameters()
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.get("lr", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        stopper = self.make_early_stopper(val_loader)
        self._amp_setup()

        total_loss, n_steps = 0.0, 0
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"T{task.task_id} ep{epoch+1}/{epochs}", leave=False)
            for step, batch in enumerate(pbar):
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                optimizer.zero_grad()
                with self._amp_autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                loss.backward()

                if (
                    step % self.projection_freq == 0
                    and self._probe_feats0 is not None
                    and self.projection_mode == "surgery"
                ):
                    self._project_gradient_surgery(params)

                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()

                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if self._early_stop_after_epoch(stopper, val_loader, task, epoch):
                break

        stopper.restore_best(self.model)
        return TrainMetrics(
            task_id=task.task_id, loss=total_loss / max(n_steps, 1), n_steps=n_steps
        )

    def _project_gradient_surgery(self, params: list[nn.Parameter]) -> None:
        """PCGrad-style: remove task-grad component conflicting with drift-preservation.

        Runs ONE stored probe batch through backbone, measures MSE drift from task-0
        snapshot, computes preservation gradient, then projects task gradient.
        Uses only the first probe batch to stay within VRAM budget.
        """
        if not self._probe_cache:
            return
        self.model.eval()
        self._capture_grad = True
        batch = {
            k: v.to(self.device) for k, v in self._probe_cache[0].items() if torch.is_tensor(v)
        }
        self.model(**{k: v for k, v in batch.items() if k != "labels"})
        cur = self._cur_feats
        mask = batch.get("attention_mask")
        if mask is not None and mask.shape[:2] == cur.shape[:2]:
            cur = cur[mask.bool()]
        else:
            cur = cur.reshape(-1, cur.shape[-1])
        n = min(cur.shape[0], self._probe_feats0.shape[0])
        drift_loss = ((cur[:n] - self._probe_feats0[:n].to(self.device)) ** 2).mean()
        self._capture_grad = False

        preserve_grads = torch.autograd.grad(
            drift_loss, params, retain_graph=False, allow_unused=True
        )
        del drift_loss

        for p, g_preserve in zip(params, preserve_grads, strict=False):
            if p.grad is None or g_preserve is None:
                continue
            dot = (p.grad * g_preserve).sum()
            if dot < 0:
                p.grad = p.grad - (dot / (g_preserve.norm() ** 2 + 1e-8)) * g_preserve

        self.model.train()

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        feats, positions, labels, raw_batches = self._extract_features(train_loader)
        self.memory.store(
            task.task_id, feats, positions, labels, self.model.id_to_label, raw_batches
        )

        if task.task_id == 0:
            self._snapshot_probe(train_loader)

        if task.task_id > 0 and self.analytic_head:
            self._analytic_head_update()

        if self.freeze_backbone_after_t0 and task.task_id == 0:
            self.model.freeze_backbone()
            log.info("nullspace_analytic: backbone frozen after task 0")

        if not self._head_frozen and task.task_id > 0 and self.analytic_head:
            for p in self.model.model.classifier.parameters():
                p.requires_grad = False
            self._head_frozen = True

        self._save_artifact()

    def _snapshot_probe(self, loader: DataLoader) -> None:
        """Cache probe batches + their task-0 feature snapshot for drift measurement."""
        self._probe_cache = []
        for i, batch in enumerate(loader):
            if i >= self.probe_batches:
                break
            self._probe_cache.append(
                {k: v.detach().cpu().clone() for k, v in batch.items() if torch.is_tensor(v)}
            )
        self._probe_feats0 = self._get_probe_feats()
        log.info(
            "nullspace_analytic: probe cached (%d batches, %d token feats)",
            len(self._probe_cache),
            self._probe_feats0.shape[0],
        )

    def _get_probe_feats(self) -> torch.Tensor:
        """Extract current classifier-input features on cached probe batches."""
        was_training = self.model.training
        self.model.eval()
        out: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in self._probe_cache:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                self.model(**{k: v for k, v in batch.items() if k != "labels"})
                f = self._cur_feats
                mask = batch.get("attention_mask")
                if mask is not None and mask.shape[:2] == f.shape[:2]:
                    f = f[mask.bool()]
                else:
                    f = f.reshape(-1, f.shape[-1])
                out.append(f.detach().float().cpu())
        if was_training:
            self.model.train()
        return torch.cat(out)

    def _extract_features(
        self, loader: DataLoader
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
        feats_all, pos_all, labels_all, raw_all = [], [], [], []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="extracting features", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                self._cur_feats = None
                self.model(**{k: v for k, v in batch.items() if k != "labels"})
                if self._cur_feats is None:
                    continue
                f = self._cur_feats
                mask = batch.get("attention_mask")
                if mask is not None and mask.shape[:2] == f.shape[:2]:
                    f = f[mask.bool()]
                    pos = (
                        batch["bbox"][mask.bool()]
                        if "bbox" in batch
                        else torch.zeros(f.shape[0], 4)
                    )
                    lbl = batch["labels"][mask.bool()]
                else:
                    f = f.reshape(-1, f.shape[-1])
                    pos = (
                        batch["bbox"].reshape(-1, 4)
                        if "bbox" in batch
                        else torch.zeros(f.shape[0], 4)
                    )
                    lbl = batch["labels"].reshape(-1)
                feats_all.append(f.cpu())
                pos_all.append(pos.cpu())
                labels_all.append(lbl.cpu())
                raw_all.append({k: v.cpu() for k, v in batch.items()})
        if was_training:
            self.model.train()
        if not feats_all:
            return (
                torch.empty(0, self.model.hidden_size),
                torch.empty(0, 4),
                torch.empty(0),
                [],
            )
        return torch.cat(feats_all), torch.cat(pos_all), torch.cat(labels_all), raw_all

    def _analytic_head_update(self) -> None:
        """Compute head weights via least squares: W = lstsq(X, Y).

        Re-extracts features from stored raw inputs using current backbone to avoid
        stale feature drift. Falls back to stored features if raw inputs unavailable.

        Soft constraint: adds λ||W - W_old||² regularization to allow
        small drift and improve conditioning.
        """
        raw_batches = self.memory.get_all_raw_batches()
        if raw_batches:
            x_all, y_all = self._reextract_features(raw_batches)
            if x_all is None or x_all.shape[0] == 0:
                log.warning("nullspace_analytic: re-extraction failed, skipping head update")
                return
        else:
            x_all, _, y_all = self.memory.get_all()
            if x_all.shape[0] == 0:
                return

        x_all = x_all.to(self.device)
        y_all = y_all.to(self.device)

        n_classes = self.model.model.classifier.out_features
        y_oh = torch.zeros(x_all.shape[0], n_classes, device=self.device)
        y_oh.scatter_(1, y_all.unsqueeze(1).long(), 1.0)

        if self.soft_lambda > 0:
            w_old = self.model.model.classifier.weight.data  # (C, D)  # noqa: N806
            lam_sqrt = self.soft_lambda**0.5
            x_reg = lam_sqrt * torch.eye(x_all.shape[1], device=self.device)  # (D, D)  # noqa: N806
            y_reg = lam_sqrt * w_old.T  # (D, C)  # noqa: N806
            x_aug = torch.cat([x_all, x_reg], dim=0)  # noqa: N806
            y_aug = torch.cat([y_oh, y_reg], dim=0)  # noqa: N806
            sol = torch.linalg.lstsq(x_aug, y_aug).solution
        else:
            sol = torch.linalg.lstsq(x_all, y_oh).solution

        w_new = sol.T  # noqa: N806
        with torch.no_grad():
            self.model.model.classifier.weight.copy_(w_new)
            self.model.model.classifier.bias.zero_()
        log.info(
            "nullspace_analytic: head updated via lstsq (%d samples, lambda=%.4f)",
            x_all.shape[0],
            self.soft_lambda,
        )

    def _reextract_features(
        self, raw_batches: list[dict]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Re-extract features from stored raw inputs using current backbone.

        Returns (features, labels) aligned with valid tokens (labels != -100).
        """
        feats_all = []
        labels_all = []
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch in raw_batches:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                self._cur_feats = None
                self.model(**{k: v for k, v in batch.items() if k != "labels"})
                if self._cur_feats is None:
                    continue
                f = self._cur_feats
                mask = batch.get("attention_mask")
                if mask is not None and mask.shape[:2] == f.shape[:2]:
                    f = f[mask.bool()]
                    lbl = batch["labels"][mask.bool()]
                else:
                    f = f.reshape(-1, f.shape[-1])
                    lbl = batch["labels"].reshape(-1)
                feats_all.append(f.cpu())
                labels_all.append(lbl.cpu())
        if was_training:
            self.model.train()

        if not feats_all:
            return None, None

        feats_cat = torch.cat(feats_all)
        labels_cat = torch.cat(labels_all)

        valid_mask = labels_cat != -100
        if not valid_mask.any():
            return None, None

        return feats_cat[valid_mask], labels_cat[valid_mask]

    def _save_artifact(self) -> None:
        out_dir = getattr(self, "out_dir", None)
        if not out_dir:
            return
        artifact = {
            "samples_per_class": self.samples_per_class,
            "projection_freq": self.projection_freq,
            "projection_mode": self.projection_mode,
            "soft_lambda": self.soft_lambda,
            "memory_bytes": self.memory.memory_bytes(),
            "tasks_stored": list(self.memory.memory.keys()),
        }
        path = Path(out_dir) / "nullspace_analytic.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact))

    def memory_bytes(self) -> int:
        return self.memory.memory_bytes()
