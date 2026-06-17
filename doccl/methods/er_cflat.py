"""C-Flat++ on Experience Replay (``er_cflat``).

The 2025 "currency" member of the optimisation / flat-minima family. C-Flat is a
sharpness-aware optimiser step that is *plug-and-play* on top of any CL method;
following the strongest-cheap-base principle we bolt it onto ER (replay buffer +
flat-minima updates). See ``doccl.methods.sam_step`` for the two-step itself.

References:
    - Foret et al., SAM, ICLR 2021 (arXiv:2010.01412).
    - Bian et al., C-Flat, NeurIPS 2024 (arXiv:2404.00986); C-Flat++ (arXiv:2508.18860).

Cost: ~2× forward-backward per step (the SAM ascent needs a second forward). This is
the documented price of the flat-minima estimate and is why the config notes 2× compute.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.er import ER
from doccl.methods.sam_step import SAMStep
from doccl.types import TaskInfo, TrainMetrics


class ERCFlat(ER):
    """Experience Replay with a C-Flat (sharpness-aware) two-step update."""

    name = "er_cflat"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.rho = config.get("rho", 0.05)
        self.cflat_lambda = config.get("cflat_lambda", 0.1)

    def _ce_plus_replay(self, batch: dict) -> torch.Tensor:
        """ER's per-step loss: current-task CE + (if buffer non-empty) replay CE.

        Factored out so the two SAM forward passes share one definition. Mirrors
        ``ER.train_task`` (er.py) exactly — same buffer sampling, same summed loss.
        """
        cur_out = self.model(**batch)
        loss = cur_out.loss
        replay_batch = self.state.buffer.sample(self.replay_batch_size)
        if replay_batch is not None:
            replay_batch = {k: v.to(self.device) for k, v in replay_batch.items()}
            loss = loss + self.model(**replay_batch).loss
        return loss

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
        params = self.trainable_parameters()

        total_loss = 0.0
        n_steps = 0
        for epoch in range(epochs):
            pbar = tqdm(
                train_loader, desc=f"ER-CFlat T{task.task_id} ep{epoch+1}/{epochs}", leave=False
            )
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                sam = SAMStep(params, rho=self.rho, cflat_lambda=self.cflat_lambda)

                # ─ Step 1: clean gradient g₀ at θ, then ascend to θ+ε ─
                optimizer.zero_grad()
                loss = self._ce_plus_replay(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                eps = sam.first_step()

                # ─ Step 2: gradient g₊ at θ+ε, restore θ, blend curvature, step ─
                optimizer.zero_grad()
                loss2 = self._ce_plus_replay(batch)
                loss2.backward()
                sam.second_step(eps)
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()

                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if self._early_stop_after_epoch(stopper, val_loader, task, epoch):
                break

        stopper.restore_best(self.model)
        return TrainMetrics(
            task_id=task.task_id,
            loss=total_loss / max(n_steps, 1),
            n_steps=n_steps,
        )
