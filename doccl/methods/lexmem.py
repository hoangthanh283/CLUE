"""LexMem — sparse lexical-memory head for continual document IE.

Adaptation of "Continual Learning via Sparse Memory Finetuning" (Lin et al.,
arXiv 2510.15103) to encoder token classification, built from the LexSlot RCA:

  - **Task 0**: full fine-tune (plain NaiveFineTune loop — full plasticity, which
    kills LexSlot-FM's frozen-probe capacity failure). Then the backbone AND the
    base classifier head are frozen forever — the base predictor can never drift
    (which kills standalone LexSlot's base-drift collapse).
  - **Memory head**: a flat key-value memory (``LexicalMemoryHead``) attached via
    classifier forward hooks; keys are anchored in task-0 token features and
    frozen; values live in logit space and add to the frozen base head's logits.
    Routing is per-token top-k retrieval — the forward pass itself, train == eval,
    no external gate.
  - **Tasks >= 1**: a counting pass records per-slot access counts; TF-IDF against
    a cumulative background (task 0 + all previously learned tasks) selects the
    top-t task-specific slots; ONLY those value rows train (plain SGD — no
    momentum/decay leakage into masked rows). Interference between tasks is
    bounded by slot-access overlap, which for documents tracks lexical overlap.

Fallback knob for frozen-feature plasticity: ``unfreeze_late_n`` unfreezes the
last-N encoder layers for tasks >= 1 (default 0 = fully frozen; turning it on
trades away the staleness guarantee).

CIL caveat (pilot targets DIL first): after ``expand_classifier`` replaces the
head module, hooks are re-registered in ``before_task``; the zero-shot FWT
reading train.py takes between expansion and ``before_task`` therefore runs
without the memory delta on CIL only.

Artifact: ``results/<run>/lexmem_slots.json`` — per-task selected slots, owner
histogram, pairwise selection Jaccard (the routing-interpretability evidence).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.lexmem_memory import LexicalMemoryHead
from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo, TrainMetrics

log = logging.getLogger(__name__)

__all__ = ["LexMem"]


class LexMem(NaiveFineTune):
    """Frozen base + sparsely-finetuned lexical memory head (SMF-style)."""

    name = "lexmem"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.n_slots = int(config.get("n_slots", 65536))
        self.top_k = int(config.get("top_k", 32))
        self.top_t = int(config.get("top_t", 4096))
        self.temp = float(config.get("temp", 0.05))
        self.key_init = str(config.get("key_init", "kmeans"))
        self.kmeans_iters = int(config.get("kmeans_iters", 8))
        self.key_sample_cap = int(config.get("key_sample_cap", 200_000))
        self.select_mode = str(config.get("select", "tfidf"))
        self.lr_mem = float(config.get("lr_mem", 0.05))
        self.unfreeze_late_n = int(config.get("unfreeze_late_n", 0))

        self.hidden_dim = model.hidden_size
        n_labels = self.model.model.classifier.out_features
        self.mem = LexicalMemoryHead(
            self.n_slots, self.hidden_dim, n_labels, top_k=self.top_k, temp=self.temp
        ).to(self.device)

        # Inactive until task 0 is trained and keys are anchored — the hook is an
        # exact no-op during task-0 training (zero-init values + _mem_active gate).
        self._mem_active = False
        self._cur_feats: torch.Tensor | None = None
        self._hook_handles: list = []
        self._register_head_hooks()

        self._selected: dict[int, list[int]] = {}  # task_id -> selected slot indices

    # ── hooks (classifier capture + additive memory delta) ──────────────────

    def _register_head_hooks(self) -> None:
        self._hook_handles = []
        cls = self.model.model.classifier
        self._hook_handles.append(cls.register_forward_pre_hook(self._capture_feats))
        self._hook_handles.append(cls.register_forward_hook(self._add_mem))

    def _detach_head_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def _capture_feats(self, _module, inp):
        self._cur_feats = inp[0]
        return None

    def _add_mem(self, _module, _inp, output):
        if (
            not self._mem_active
            or self._cur_feats is None
            or self._cur_feats.shape[1] != output.shape[1]
        ):
            return output
        return output + self.mem.delta(self._cur_feats).to(output.dtype)

    # ── task lifecycle ───────────────────────────────────────────────────────

    def before_task(self, task: TaskInfo, train_loader) -> None:
        # CIL: expand_classifier may have REPLACED the classifier module (orphaning
        # the hooks) and re-created it trainable. Re-point hooks; grow value columns;
        # re-freeze the base head for tasks >= 1 (it froze at end of task 0).
        self.mem.expand_labels(self.model.model.classifier.out_features)
        self._detach_head_hooks()
        self._register_head_hooks()
        if task.task_id == 0:
            return
        for p in self.model.model.classifier.parameters():
            p.requires_grad = False

        counts = self._count_pass(train_loader, desc=f"T{task.task_id} slot-count")
        idx = self.mem.select_topt(counts, self.top_t, task.task_id, self.select_mode)
        self._selected[task.task_id] = idx.cpu().tolist()
        n_accessed = int((counts > 0).sum())
        log.info(
            "lexmem: task %d — %d/%d slots accessed, %d selected (%s); " "bg batches=%d",
            task.task_id,
            n_accessed,
            self.n_slots,
            len(idx),
            self.select_mode,
            self.mem.n_bg_batches,
        )

    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        if task.task_id == 0:
            return super().train_task(task, train_loader, val_loader)
        return self._train_memory(task, train_loader, val_loader)

    def _train_memory(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None
    ) -> TrainMetrics:
        """Sparse memory finetuning: SGD on the selected value rows only."""
        self.model.train()
        params: list[nn.Parameter] = [self.mem.values]
        groups = [{"params": [self.mem.values], "lr": self.lr_mem, "weight_decay": 0.0}]
        snap = nn.ModuleList([self.mem])  # what the early stopper snapshots/restores
        if self.unfreeze_late_n > 0:
            late = self._unfreeze_late_layers(self.unfreeze_late_n)
            params += late
            groups.append({"params": late, "lr": float(self.config.get("lr", 5e-5))})
            snap.append(self.model)
        optimizer = torch.optim.SGD(groups)
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        stopper = self.make_early_stopper(val_loader)
        self._amp_setup()

        total_loss, n_steps = 0.0, 0
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"T{task.task_id} mem ep{epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                optimizer.zero_grad()
                with self._amp_autocast():
                    outputs = self.model(**batch)  # hook adds the memory delta
                    loss = outputs.loss
                self._amp_backward_step(loss, optimizer, params, max_grad_norm)
                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            # Early-stop tail mirroring base._early_stop_after_epoch, but snapshotting
            # the memory (the only thing training) instead of the frozen model.
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

    def after_task(self, task: TaskInfo, train_loader) -> None:
        if task.task_id == 0:
            # Freeze the base predictor forever: backbone AND classifier head.
            self.model.freeze_backbone()
            for p in self.model.model.classifier.parameters():
                p.requires_grad = False
            feats = self._collect_feats(train_loader)
            self.mem.init_keys(feats, mode=self.key_init, iters=self.kmeans_iters)
            self._mem_active = True
            log.info(
                "lexmem: task 0 done — base frozen; keys anchored from %d token feats "
                "(%s); memory active.",
                feats.shape[0],
                self.key_init,
            )
        # Fold this task's data into the IDF background so later tasks avoid the
        # slots that now carry learned knowledge (cumulative "existing capabilities").
        self._merge_background(train_loader, desc=f"T{task.task_id} bg-count")
        self._save_slot_artifact()

    # ── counting / feature-collection passes (no_grad, eval mode) ────────────

    def _count_pass(self, loader, desc: str) -> torch.Tensor:
        """One pass accumulating total per-slot access counts for a task."""
        was_training = self.model.training
        self.model.eval()
        self.mem.start_counting()
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                self.mem._count_mask = batch.get("attention_mask")
                self.model(**batch)
        counts = self.mem.stop_counting()
        self.mem._count_mask = None
        if was_training:
            self.model.train()
        return counts

    def _merge_background(self, loader, desc: str) -> None:
        """Per-batch counting pass folded into the IDF background statistics."""
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc, leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                self.mem._count_mask = batch.get("attention_mask")
                self.mem.start_counting()
                self.model(**batch)
                self.mem.note_background_batch(self.mem.stop_counting())
        self.mem._count_mask = None
        if was_training:
            self.model.train()

    def _collect_feats(self, loader) -> torch.Tensor:
        """Gather attention-masked classifier-input features for key anchoring."""
        chunks: list[torch.Tensor] = []
        n = 0
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="T0 key-feats", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                self.model(**{k: v for k, v in batch.items() if k != "labels"})
                if self._cur_feats is None:
                    continue
                f = self._cur_feats
                mask = batch.get("attention_mask")
                if mask is not None and mask.shape[:2] == f.shape[:2]:
                    f = f[mask.bool()]
                else:
                    f = f.reshape(-1, f.shape[-1])
                chunks.append(f.detach().half().cpu())
                n += f.shape[0]
                if n >= 2 * self.key_sample_cap:
                    break
        if was_training:
            self.model.train()
        if not chunks:
            return torch.empty(0, self.hidden_dim)
        feats = torch.cat(chunks)
        if feats.shape[0] > self.key_sample_cap:
            feats = feats[torch.randperm(feats.shape[0])[: self.key_sample_cap]]
        return feats.to(self.device)

    # ── fallback plasticity knob ─────────────────────────────────────────────

    def _unfreeze_late_layers(self, n: int) -> list[nn.Parameter]:
        inner = getattr(self.model, "_inner", None)
        if inner is None:
            inner = getattr(self.model.model, "layoutlmv3", None)
        if inner is None:
            raise RuntimeError("lexmem: cannot locate inner encoder for unfreeze_late_n")
        layers = inner.encoder.layer
        params: list[nn.Parameter] = []
        for layer in list(layers)[-n:]:
            for p in layer.parameters():
                p.requires_grad = True
                params.append(p)
        log.info("lexmem: unfroze last %d encoder layers (staleness guarantee off)", n)
        return params

    # ── artifact ─────────────────────────────────────────────────────────────

    def _save_slot_artifact(self) -> None:
        out_dir = getattr(self, "out_dir", None)
        if not out_dir:
            return
        owners = self.mem.slot_owner.cpu()
        hist = {int(t): int((owners == t).sum()) for t in owners.unique().tolist() if t >= 0}
        sets = {t: set(v) for t, v in self._selected.items()}
        overlap = {}
        tids = sorted(sets)
        for i, a in enumerate(tids):
            for b in tids[i + 1 :]:
                union = len(sets[a] | sets[b])
                overlap[f"{a}-{b}"] = len(sets[a] & sets[b]) / union if union else 0.0
        artifact = {
            "n_slots": self.n_slots,
            "top_k": self.top_k,
            "top_t": self.top_t,
            "select": self.select_mode,
            "n_bg_batches": self.mem.n_bg_batches,
            "owner_histogram": hist,
            "selected": {str(t): v for t, v in self._selected.items()},
            "selection_jaccard": overlap,
        }
        path = Path(out_dir) / "lexmem_slots.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact))
        log.info("lexmem: slot artifact -> %s (jaccard=%s)", path, overlap)
