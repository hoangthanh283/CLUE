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

    def _sample_replay(self) -> dict | None:
        """Sample ONE replay batch (moved to device), or None if the buffer is empty.

        Sampled once per optimizer step and reused across BOTH SAM forwards — SAM
        requires the two passes to evaluate the SAME loss L(θ, batch); re-sampling a
        different replay batch in the second forward would make the ascent ε invalid
        and reduce er_cflat to noisy ER, silently invalidating the C-Flat comparison.
        """
        replay_batch = self.state.buffer.sample(self.replay_batch_size)
        if replay_batch is None:
            return None
        return {k: v.to(self.device) for k, v in replay_batch.items()}

    def _ce_plus_replay(self, batch: dict, replay_batch: dict | None) -> torch.Tensor:
        """ER's per-step loss: current-task CE + (if given) the SAME replay CE.

        ``replay_batch`` is the per-step fixed sample from ``_sample_replay`` — shared
        by both SAM forwards so the loss surface is identical across the two passes.
        """
        loss = self.model(**batch).loss
        if replay_batch is not None:
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
                # Sample the replay batch ONCE; reuse it for both SAM forwards.
                replay_batch = self._sample_replay()

                optimizer.zero_grad()
                loss = self._ce_plus_replay(batch, replay_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                eps = sam.first_step()

                optimizer.zero_grad()
                loss2 = self._ce_plus_replay(batch, replay_batch)
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
