"""LexMem — sparse lexical-memory head for continual document IE.

Adaptation of "Continual Learning via Sparse Memory Finetuning" (Lin et al.,
arXiv 2510.15103) to encoder token classification, built from the LexSlot RCA
and refined by the v1 pilot (perfect retention, no plasticity):

  - **Task 0**: full fine-tune (plain NaiveFineTune loop — full plasticity).
    Then a **diagnosis-guided freeze map** is applied: the classifier head and
    the last ``freeze_late_n`` encoder layers — the components the forgetting
    diagnosis (Fisher-weighted displacement + CKA) implicates — are frozen;
    the early/mid layers the diagnosis certifies as low-drift stay TRAINABLE
    and supply plasticity for later tasks. (``freeze_late_n: -1`` freezes the
    whole backbone — the v1 pilot regime, kept for the ablation row.)
  - **Memory head**: a flat key-value memory (``LexicalMemoryHead``) attached
    via classifier hooks; keys are anchored in task-0 token features and
    frozen. ``value_space: feature`` (v2 default) adds a hidden-dim delta to
    the classifier INPUT (capacity d per slot); ``logit`` adds directly to the
    logits (the v1 regime, C per slot). Routing is per-token top-k retrieval —
    the forward pass itself, train == eval, no external gate.
  - **Tasks >= 1**: a counting pass records per-slot access counts; TF-IDF
    against a cumulative background (task 0 + previously learned tasks)
    selects the top-t task-specific slots; only those value rows train,
    jointly with the diagnosed-plastic backbone layers.

Control arm: ``mem_enabled: false`` runs the identical freeze map with a plain
frozen head and NO memory — isolating what the freeze map alone contributes.

Drift probe (``drift_probe_batches > 0``): a few task-0 batches are cached; at
every task boundary the classifier-input features on those batches are compared
to their task-0 snapshot (linear CKA + mean top-1 key cosine), measuring the
pressure-redirection / key-staleness risk the freeze map introduces. Logged and
saved into the slot artifact.

CIL caveats (pilots target DIL): hooks are re-registered in ``before_task``
after ``expand_classifier`` replaces the head; with ``value_space: feature``
labels introduced after task 0 sit in frozen zero-init head rows and are not
reachable — CIL needs ``logit`` values (which grow via ``expand_labels``).

Artifact: ``results/<run>/lexmem_slots.json`` — per-task selected slots, owner
histogram, pairwise selection Jaccard, drift-probe readings.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — canonical torch alias (repo-wide)
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.eval.fisher import empirical_fisher_diagonal
from doccl.methods.lexmem_memory import LexicalMemoryHead
from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo, TrainMetrics

log = logging.getLogger(__name__)

__all__ = ["LexMem"]


def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA between two (n, d) feature matrices (column-centered)."""
    x = (x - x.mean(0)).float()
    y = (y - y.mean(0)).float()
    num = (x.T @ y).norm() ** 2
    den = (x.T @ x).norm() * (y.T @ y).norm()
    return float(num / den.clamp(min=1e-12))


class LexMem(NaiveFineTune):
    """Diagnosis-guided freeze map + sparsely-finetuned lexical memory head."""

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
        self.select_exclude_owned = bool(config.get("select_exclude_owned", False))
        self.lr_mem = float(config.get("lr_mem", 0.05))
        # v2 knobs. Defaults reproduce the v1 pilot exactly (logit values, SGD,
        # full backbone freeze, no probe) so the saved v1 run stays reproducible.
        self.value_space = str(config.get("value_space", "logit"))
        self.mem_enabled = bool(config.get("mem_enabled", True))
        self.mem_optimizer = str(config.get("mem_optimizer", "sgd"))
        self.freeze_late_n = int(config.get("freeze_late_n", -1))
        self.drift_probe_batches = int(config.get("drift_probe_batches", 0))
        # v3 knob: EWC on the PLASTIC bucket only (the diagnosed-stable early/mid
        # layers). The v2 pilot measured massive pressure-redirection drift there
        # (CKA 0.19); this is the stability regularizer where plasticity lives,
        # while the memory is the architecture where forgetting lives.
        self.ewc_lambda = float(config.get("ewc_lambda", 0.0))
        self.fisher_n_samples = int(config.get("fisher_n_samples", 200))
        self._fisher: dict[str, torch.Tensor] = {}
        self._theta_star: dict[str, torch.Tensor] = {}
        if not self.mem_enabled and self.freeze_late_n < 0:
            raise ValueError(
                "lexmem: mem_enabled=false with freeze_late_n=-1 leaves nothing "
                "trainable after task 0 — set freeze_late_n >= 0 for the control arm."
            )

        self.hidden_dim = model.hidden_size
        n_labels = self.model.model.classifier.out_features
        value_dim = self.hidden_dim if self.value_space == "feature" else None
        self.mem = LexicalMemoryHead(
            self.n_slots,
            self.hidden_dim,
            n_labels,
            top_k=self.top_k,
            temp=self.temp,
            value_dim=value_dim,
        ).to(self.device)

        # Inactive until task 0 is trained and keys are anchored — the hook is an
        # exact no-op during task-0 training (zero-init values + _mem_active gate).
        self._mem_active = False
        self._cur_feats: torch.Tensor | None = None
        self._hook_handles: list = []
        # Hooks are registered even in the control arm (mem_enabled=false): the
        # capture pre-hook feeds the drift probe; the memory paths stay gated off
        # because _mem_active is never set true without the memory.
        self._register_head_hooks()

        self._selected: dict[int, list[int]] = {}  # task_id -> selected slot indices
        # Drift probe state: cached task-0 batches + their task-0 feature snapshot.
        self._probe_batches: list[dict] = []
        self._probe_feats0: torch.Tensor | None = None
        self._drift: dict[str, dict[str, float]] = {}

    # ── hooks (classifier capture + memory delta) ────────────────────────────

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
        """Stash the RAW classifier input; in feature mode, return it shifted.

        The raw (pre-delta) features are what key anchoring, counting, and the
        drift probe read; the memory's correction is applied on top. Gradient
        flows to the trainable backbone through ``inp[0]`` and to the selected
        value rows through the delta (whose query side is detached).
        """
        self._cur_feats = inp[0]
        if self._mem_active and self.value_space == "feature":
            return (inp[0] + self.mem.delta(inp[0]).to(inp[0].dtype),)
        return None

    def _add_mem(self, _module, _inp, output):
        if (
            not self._mem_active
            or self.value_space != "logit"
            or self._cur_feats is None
            or self._cur_feats.shape[1] != output.shape[1]
        ):
            return output
        return output + self.mem.delta(self._cur_feats).to(output.dtype)

    # ── task lifecycle ───────────────────────────────────────────────────────

    def before_task(self, task: TaskInfo, train_loader) -> None:
        # CIL: expand_classifier may have REPLACED the classifier module (orphaning
        # the hooks) and re-created it trainable. Re-point hooks; grow value columns
        # (logit space only); re-freeze the base head for tasks >= 1.
        if self.mem_enabled and self.value_space == "logit":
            self.mem.expand_labels(self.model.model.classifier.out_features)
        self._detach_head_hooks()
        self._register_head_hooks()
        if task.task_id == 0:
            return
        for p in self.model.model.classifier.parameters():
            p.requires_grad = False
        if not self.mem_enabled:
            return

        counts = self._count_pass(train_loader, desc=f"T{task.task_id} slot-count")
        idx = self.mem.select_topt(
            counts,
            self.top_t,
            task.task_id,
            self.select_mode,
            exclude_owned=self.select_exclude_owned,
        )
        self._selected[task.task_id] = idx.cpu().tolist()
        n_accessed = int((counts > 0).sum())
        log.info(
            "lexmem: task %d — %d/%d slots accessed, %d selected (%s); bg batches=%d",
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
        """Tasks >= 1: selected memory rows + diagnosed-plastic backbone layers."""
        self.model.train()
        params: list[nn.Parameter] = []
        groups: list[dict] = []
        snap = nn.ModuleList()
        if self.mem_enabled:
            groups.append({"params": [self.mem.values], "lr": self.lr_mem, "weight_decay": 0.0})
            params.append(self.mem.values)
            snap.append(self.mem)
        backbone = [p for p in self.model.parameters() if p.requires_grad]
        if backbone:
            groups.append(
                {
                    "params": backbone,
                    "lr": float(self.config.get("lr", 5e-5)),
                    "weight_decay": float(self.config.get("weight_decay", 0.01)),
                }
            )
            params += backbone
            snap.append(self.model)
        opt_cls = torch.optim.AdamW if self.mem_optimizer == "adamw" else torch.optim.SGD
        optimizer = opt_cls(groups)
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
                    outputs = self.model(**batch)  # hook applies the memory delta
                    loss = outputs.loss
                    if self.ewc_lambda > 0 and self._fisher:
                        loss = loss + self._ewc_penalty()
                self._amp_backward_step(loss, optimizer, params, max_grad_norm)
                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            # Early-stop tail mirroring base._early_stop_after_epoch, but snapshotting
            # exactly the modules that train (memory and/or plastic backbone).
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
            self._apply_freeze_map()
            if self.mem_enabled:
                feats = self._collect_feats(train_loader)
                self.mem.init_keys(feats, mode=self.key_init, iters=self.kmeans_iters)
                self._mem_active = True
                log.info(
                    "lexmem: task 0 done — freeze map applied; keys anchored from %d "
                    "token feats (%s); memory active.",
                    feats.shape[0],
                    self.key_init,
                )
            if self.drift_probe_batches > 0:
                self._snapshot_probe(train_loader)
        if self.mem_enabled:
            # Fold this task's data into the IDF background so later tasks avoid the
            # slots that now carry learned knowledge (cumulative existing capabilities).
            self._merge_background(train_loader, desc=f"T{task.task_id} bg-count")
        if self.ewc_lambda > 0:
            self._accumulate_fisher(task, train_loader)
        if task.task_id > 0:
            self._measure_drift(task.task_id)
        self._save_slot_artifact()

    # ── EWC on the plastic bucket ────────────────────────────────────────────

    def _accumulate_fisher(self, task: TaskInfo, train_loader) -> None:
        """Fisher diagonal + θ* snapshot for the PLASTIC (trainable) params only.

        Runs after the freeze map, so frozen params receive no grad and drop out
        naturally; the memory lives outside ``model.named_parameters()``. Online
        EWC: Fishers sum across tasks (as in lexslot_fm/DocCL).
        """
        log.info(
            "lexmem: computing EWC Fisher on task %d (%d samples)",
            task.task_id,
            self.fisher_n_samples,
        )
        new_fisher = empirical_fisher_diagonal(
            self.model, train_loader, n_samples=self.fisher_n_samples, device=self.device
        )
        params = dict(self.model.named_parameters())
        n_prot = 0
        for name, f_val in new_fisher.items():
            p = params.get(name)
            if p is None or not p.requires_grad:
                continue
            self._theta_star[name] = p.detach().clone()
            old = self._fisher.get(name)
            self._fisher[name] = f_val if old is None else old + f_val
            n_prot += 1
        log.info("lexmem: EWC Fisher accumulated — %d plastic tensors protected", n_prot)

    def _ewc_penalty(self) -> torch.Tensor:
        """(λ/2)·Σ F_i (θ_i − θ*_i)² over the plastic bucket."""
        params = dict(self.model.named_parameters())
        penalty = torch.zeros((), device=self.device)
        for name, fisher_val in self._fisher.items():
            p = params.get(name)
            star = self._theta_star.get(name)
            if p is None or star is None:
                continue
            penalty = penalty + (fisher_val * (p - star) ** 2).sum()
        return (self.ewc_lambda / 2) * penalty

    def _apply_freeze_map(self) -> None:
        """Freeze the diagnosed forgetting locus; leave the low-drift bucket plastic.

        ``freeze_late_n == -1``: freeze the whole backbone (v1 / full-stability).
        ``freeze_late_n == n >= 0``: freeze only the last-n encoder layers; the
        embeddings and early/mid layers stay trainable. The classifier head is
        always frozen — its role is taken by the memory (or, in the control arm,
        it acts as the fixed task-0 anchor).
        """
        if self.freeze_late_n < 0:
            self.model.freeze_backbone()
        elif self.freeze_late_n > 0:
            layers = list(self._inner_encoder().encoder.layer)
            for layer in layers[len(layers) - self.freeze_late_n :]:
                for p in layer.parameters():
                    p.requires_grad = False
        for p in self.model.model.classifier.parameters():
            p.requires_grad = False
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        log.info(
            "lexmem: freeze map — freeze_late_n=%d, head frozen; trainable backbone " "params=%d",
            self.freeze_late_n,
            n_train,
        )

    def _inner_encoder(self):
        inner = getattr(self.model, "_inner", None)
        if inner is None:
            inner = getattr(self.model.model, "layoutlmv3", None)
        if inner is None:
            raise RuntimeError("lexmem: cannot locate the inner encoder module")
        return inner

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

    # ── drift probe (pressure-redirection / key-staleness measurement) ───────

    def _probe_feats(self) -> torch.Tensor:
        """Raw classifier-input features on the cached probe batches (masked)."""
        was_training = self.model.training
        self.model.eval()
        out: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in self._probe_batches:
                batch = {k: v.to(self.device) for k, v in batch.items()}
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

    def _snapshot_probe(self, loader) -> None:
        """Cache a few task-0 batches + their post-freeze feature snapshot."""
        for i, batch in enumerate(loader):
            if i >= self.drift_probe_batches:
                break
            self._probe_batches.append(
                {k: v.detach().cpu().clone() for k, v in batch.items() if torch.is_tensor(v)}
            )
        self._probe_feats0 = self._probe_feats()
        log.info(
            "lexmem: drift probe cached (%d batches, %d token feats)",
            len(self._probe_batches),
            self._probe_feats0.shape[0],
        )

    def _measure_drift(self, task_id: int) -> None:
        """CKA + top-1 key cosine of task-0 probe features vs their t0 snapshot."""
        if self._probe_feats0 is None or not self._probe_batches:
            return
        cur = self._probe_feats()
        cka = _linear_cka(self._probe_feats0, cur)
        reading = {"cka_t0_feats": round(cka, 4)}
        if self.mem_enabled:
            q = F.normalize(cur.to(self.device), dim=-1)
            keys = self.mem.keys.to(q.dtype)
            top1 = torch.cat(
                [(q[i : i + 4096] @ keys.T).max(dim=-1).values for i in range(0, q.shape[0], 4096)]
            )
            reading["mean_top1_key_cos"] = round(float(top1.mean()), 4)
        self._drift[str(task_id)] = reading
        log.info("lexmem: drift probe after task %d — %s", task_id, reading)

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
            "value_space": self.value_space,
            "mem_enabled": self.mem_enabled,
            "freeze_late_n": self.freeze_late_n,
            "n_bg_batches": self.mem.n_bg_batches,
            "owner_histogram": hist,
            "selected": {str(t): v for t, v in self._selected.items()},
            "selection_jaccard": overlap,
            "drift": self._drift,
        }
        path = Path(out_dir) / "lexmem_slots.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact))
        log.info("lexmem: slot artifact -> %s (jaccard=%s)", path, overlap)
