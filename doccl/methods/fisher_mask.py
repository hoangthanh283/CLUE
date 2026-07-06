"""Fisher-masked parameter freezing — the parameter-granularity migration test.

Experiment #6. Experiment #4 established that forgetting is conserved under
*layer*-level protection (freeze vs free within 1-2 AA across orders x maps x
horizons). This method probes the law's remaining attack surface: granularity.
Within EVERY parameter tensor (backbone and head), the top-``mask_top_p``
fraction of entries by accumulated Fisher importance over all seen tasks is
hard-frozen; the interleaved remainder stays plastic. Per-tensor (not global)
selection is deliberate: global top-p would concentrate in the head/late layers
(the diagnosed locus) and collapse back to the layer-level maps already tested —
per-tensor forces the fine-grained interleaving that PackNet/WSN-style
lottery-ticket arguments say could share capacity where whole-layer freezing
cannot.

Pre-registered readings (STATE.md experiment #6):
- conservation holds at every p -> the migration law is granularity-independent,
  and the PackNet/HAT/WSN family is explained away for task-ID-free doc-IE;
- some p yields AA meaningfully above the naive floor with retention -> the law
  breaks at fine grain, and a mask-based method is justified by mechanism.

Freezing mechanics: masked gradient entries are zeroed each step, and the masked
weight entries are snapshotted before ``optimizer.step()`` and restored after it
(``masked_scatter_``), which makes the freeze exact — immune to AdamW's decoupled
weight decay, which would otherwise shrink zero-gradient entries. The mask is
rebuilt after every task from the running Fisher sum (one importance language:
``empirical_fisher_diagonal``, the same estimator EWC and the localization
diagnostics use), and each task trains a fresh optimizer, so Adam moments for
frozen entries are exactly zero.
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.eval.fisher import empirical_fisher_diagonal
from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo, TrainMetrics

log = logging.getLogger(__name__)


class FisherMaskFreeze(NaiveFineTune):
    """Hard-freeze the per-tensor top-p Fisher-important entries of every tensor."""

    name = "fisher_mask"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.mask_top_p = float(config.get("mask_top_p", 0.8))
        self.fisher_n_samples = int(config.get("fisher_n_samples", 100))
        self.fisher_gamma = float(config.get("fisher_gamma", 1.0))
        if not 0.0 < self.mask_top_p < 1.0:
            raise ValueError(f"mask_top_p must be in (0, 1), got {self.mask_top_p}")
        self.fisher_acc: dict[str, torch.Tensor] = {}  # CPU, running gamma-sum
        self.masks: dict[str, torch.Tensor] = {}  # device bool, True = frozen entry

    # ------------------------------------------------------------------ train
    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.trainable_parameters(),
            lr=self.config.get("lr", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        stopper = self.make_early_stopper(val_loader)

        total_loss = 0.0
        n_steps = 0
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"FM T{task.task_id} ep{epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}

                optimizer.zero_grad()
                out = self.model(**batch)
                out.loss.backward()
                self._mask_grads()
                torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), max_grad_norm)
                saved = self._snapshot_frozen()
                optimizer.step()
                self._restore_frozen(saved)

                total_loss += float(out.loss.item())
                n_steps += 1
                pbar.set_postfix({"loss": f"{out.loss.item():.3f}"})
            if self._early_stop_after_epoch(stopper, val_loader, task, epoch):
                break

        stopper.restore_best(self.model)
        return TrainMetrics(
            task_id=task.task_id,
            loss=total_loss / max(n_steps, 1),
            n_steps=n_steps,
        )

    def _named_masked(self):
        """(name, param, mask-sliced-to-param) for every masked parameter.

        A CIL head expansion grows ``classifier.*`` after the mask was built; the
        stored mask then covers only the old top-left block, which is exactly
        right — new rows carry no old-task importance and must stay plastic. The
        mask is applied to the matching leading slice (EWC's min-shape pattern).
        """
        params = dict(self.model.named_parameters())
        for name, mask in self.masks.items():
            p = params.get(name)
            if p is None:
                continue
            if p.shape == mask.shape:
                yield name, p, mask
            else:
                idx = tuple(slice(0, min(a, b)) for a, b in zip(p.shape, mask.shape, strict=True))
                yield name, p, (idx, mask[tuple(slice(0, s.stop) for s in idx)])

    def _mask_grads(self) -> None:
        for _, p, mask in self._named_masked():
            if p.grad is None:
                continue
            if isinstance(mask, tuple):
                idx, sub = mask
                p.grad[idx] = p.grad[idx].masked_fill(sub, 0)
            else:
                p.grad.masked_fill_(mask, 0)

    def _snapshot_frozen(self) -> list[tuple[torch.Tensor, object, torch.Tensor]]:
        saved = []
        for _, p, mask in self._named_masked():
            if isinstance(mask, tuple):
                idx, sub = mask
                saved.append((p, mask, p.data[idx].masked_select(sub)))
            else:
                saved.append((p, mask, p.data.masked_select(mask)))
        return saved

    @staticmethod
    def _restore_frozen(saved) -> None:
        with torch.no_grad():
            for p, mask, vals in saved:
                if isinstance(mask, tuple):
                    idx, sub = mask
                    block = p.data[idx]
                    block.masked_scatter_(sub, vals)
                    p.data[idx] = block
                else:
                    p.data.masked_scatter_(mask, vals)

    # ------------------------------------------------------------- after task
    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Accumulate this task's Fisher (gamma-sum, CPU) and rebuild the masks."""
        fisher = empirical_fisher_diagonal(
            self.model, train_loader, n_samples=self.fisher_n_samples, device=self.device
        )
        for name, f in fisher.items():
            f = f.cpu()
            old = self.fisher_acc.get(name)
            if old is None:
                self.fisher_acc[name] = f
            elif old.shape == f.shape:
                self.fisher_acc[name] = self.fisher_gamma * old + f
            else:  # head grew (CIL): zero-pad the old accumulator into the new shape
                padded = torch.zeros_like(f)
                idx = tuple(slice(0, s) for s in old.shape)
                padded[idx] = old
                self.fisher_acc[name] = self.fisher_gamma * padded + f
        self._rebuild_masks()

    def _rebuild_masks(self) -> None:
        n_frozen = 0
        n_total = 0
        for name, acc in self.fisher_acc.items():
            flat = acc.flatten()
            k = round(self.mask_top_p * flat.numel())
            n_total += flat.numel()
            if k <= 0:
                continue
            # Exact top-k indices, not a >=threshold test (Fisher ties at exactly 0
            # would over-freeze) and not torch.quantile (fails >2^24 elements — the
            # 38M-entry word embedding trips it).
            mask_flat = torch.zeros(flat.numel(), dtype=torch.bool)
            mask_flat[flat.topk(k).indices] = True
            mask = mask_flat.view(acc.shape).to(self.device)
            self.masks[name] = mask
            n_frozen += int(mask.sum())
        log.info(
            "fisher_mask: masks rebuilt — %.1f%% of %d params frozen (target %.0f%%)",
            100.0 * n_frozen / max(n_total, 1),
            n_total,
            100.0 * self.mask_top_p,
        )
