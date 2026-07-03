"""Gaussian head-replay on the drift-controlled trunk (LexMem-v4 family).

Experiment #2 of the analysis program, redesigned after the head-refit oracle:
on a NAIVE trunk, head-only recovery is ceiling-bounded at AA ~55 (features
themselves forget — the oracle's low branch), so distributional head replay is
only well-founded on a trunk whose features stay put. That trunk is v3b's:
freeze the late-layer locus, EWC the plastic early/mid bucket (measured task-0
CKA ~0.99 across the sequence).

Mechanism (exemplar-free — stores statistics, not documents):
  - After each task, extract classifier-input token features from the task's
    train split and fit per-(class x task) diagonal Gaussians (means/vars +
    token counts). O is huge and multimodal, hence task-conditioning.
  - For tasks >= 1 the head stays TRAINABLE (``freeze_head: false``) and is
    excluded from EWC (``ewc_exclude_head``); every training step adds a
    replay term: CE of the head on features SAMPLED from all previous tasks'
    Gaussians, giving the head balanced gradients from every seen task — the
    structural cure replay applies, at a storage cost of ~(2 x 768 + 1) floats
    per class-task instead of raw documents.
  - No KV slot memory (``mem_enabled: false``); trunk stability comes from the
    inherited freeze map + EWC.

Gate bars (dil): AA >= ~80 => direction lives (between v3b 66.1±3.0 and replay
~88); 70-80 => needs covariance/conditioning work; <= 70 => exemplar-free head
realignment is falsified even on a drift-controlled trunk (fourth level of the
falsification chain).
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.lexmem import LexMem
from doccl.types import TaskInfo, TrainMetrics

log = logging.getLogger(__name__)

__all__ = ["GaussReplay"]

_VAR_FLOOR = 1e-4  # shrinkage floor on per-dim variances (fp16 feature extraction)


class GaussReplay(LexMem):
    """Drift-controlled trunk + per-(class x task) Gaussian head replay."""

    name = "gauss_replay"

    def __init__(self, model, config):
        config = dict(config)
        config.setdefault("mem_enabled", False)
        config.setdefault("freeze_head", False)
        config.setdefault("ewc_exclude_head", True)
        super().__init__(model, config)
        self.replay_tokens = int(config.get("replay_tokens", 1024))
        self.replay_weight = float(config.get("replay_weight", 1.0))
        self.gauss_max_tokens = int(config.get("gauss_max_tokens", 50_000))
        # (task_id, class_id) -> {"mean": (d,), "var": (d,), "count": int}
        self._gaussians: dict[tuple[int, int], dict] = {}

    # ── Gaussian bank ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _fit_gaussians(self, task: TaskInfo, loader) -> None:
        """Per-(class x task) diagonal Gaussians from classifier-input features."""
        was_training = self.model.training
        self.model.eval()
        xs, ys = [], []
        n = 0
        for batch in tqdm(loader, desc=f"T{task.task_id} gauss-fit", leave=False):
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            self.model(**{k: v for k, v in batch.items() if k != "labels"})
            f = self._cur_feats
            labels = batch["labels"]
            mask = labels != -100
            am = batch.get("attention_mask")
            if am is not None and am.shape == labels.shape:
                mask = mask & am.bool()
            xs.append(f[mask].detach().float().cpu())
            ys.append(labels[mask].detach().cpu())
            n += int(mask.sum())
            if n >= 2 * self.gauss_max_tokens:
                break
        if was_training:
            self.model.train()
        if not xs:
            return
        x, y = torch.cat(xs), torch.cat(ys)
        if x.shape[0] > self.gauss_max_tokens:
            keep = torch.randperm(x.shape[0])[: self.gauss_max_tokens]
            x, y = x[keep], y[keep]
        for c in y.unique().tolist():
            xc = x[y == c]
            if xc.shape[0] < 2:
                continue
            self._gaussians[(task.task_id, int(c))] = {
                "mean": xc.mean(0),
                "var": xc.var(0).clamp(min=_VAR_FLOOR),
                "count": int(xc.shape[0]),
            }
        log.info(
            "gauss_replay: task %d — %d Gaussians fitted (%d tokens); bank now %d",
            task.task_id,
            len([k for k in self._gaussians if k[0] == task.task_id]),
            x.shape[0],
            len(self._gaussians),
        )

    def _sample_replay(self, cur_task: int, n: int):
        """Sample (feats, labels) from all PRIOR tasks' Gaussians, count-weighted."""
        keys = [k for k in self._gaussians if k[0] < cur_task]
        if not keys:
            return None
        counts = torch.tensor([self._gaussians[k]["count"] for k in keys], dtype=torch.float)
        pick = torch.multinomial(counts / counts.sum(), n, replacement=True)
        feats = torch.empty(n, self.hidden_dim)
        labels = torch.empty(n, dtype=torch.long)
        for i, ki in enumerate(pick.tolist()):
            g = self._gaussians[keys[ki]]
            feats[i] = g["mean"] + g["var"].sqrt() * torch.randn(self.hidden_dim)
            labels[i] = keys[ki][1]
        return feats.to(self.device), labels.to(self.device)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def after_task(self, task: TaskInfo, train_loader) -> None:
        # Freeze map / EWC / drift probes first (super), THEN fit this task's
        # Gaussians on the post-task trunk (loader re-passed; features final).
        super().after_task(task, train_loader)
        self._fit_gaussians(task, train_loader)

    def _train_memory(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None
    ) -> TrainMetrics:
        """Tasks >= 1: CE on real batches + Gaussian-replay CE on the head."""
        self.model.train()
        # freeze_head=False -> the head is already in the trainable set.
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": params,
                    "lr": float(self.config.get("lr", 5e-5)),
                    "weight_decay": float(self.config.get("weight_decay", 0.01)),
                }
            ]
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        stopper = self.make_early_stopper(val_loader)
        snap = torch.nn.ModuleList([self.model])
        self._amp_setup()

        total_loss, n_steps = 0.0, 0
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"T{task.task_id} gr ep{epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                optimizer.zero_grad()
                with self._amp_autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    if self.ewc_lambda > 0 and self._fisher:
                        loss = loss + self._ewc_penalty()
                    replay = self._sample_replay(task.task_id, self.replay_tokens)
                    if replay is not None:
                        rf, rl = replay
                        r_logits = self.model.model.classifier(rf)
                        loss = loss + self.replay_weight * torch.nn.functional.cross_entropy(
                            r_logits, rl
                        )
                self._amp_backward_step(loss, optimizer, params, max_grad_norm)
                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            self._log_weight_diagnostics()
            if stopper.enabled and val_loader is not None:
                val_f1 = self.current_task_val_f1(val_loader)
                should_stop = stopper.step(val_f1, snap, epoch)
                log.info(
                    "T%s ep%d val_f1=%.4f best=%.4f bad=%d%s",
                    task.task_id,
                    epoch + 1,
                    val_f1,
                    stopper.best_f1,
                    stopper.num_bad_epochs,
                    " -> STOP" if should_stop else "",
                )
                if should_stop:
                    break
        stopper.restore_best(snap)
        return TrainMetrics(
            task_id=task.task_id, loss=total_loss / max(n_steps, 1), n_steps=n_steps
        )
