"""Experience Replay (ER).

Reference: Rolnick et al., "Experience Replay for Continual Learning", NeurIPS 2019.

At each training step, mix a batch from the current task with a batch sampled
from the replay buffer (containing examples from past tasks).
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.buffer import ReservoirBuffer
from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo, TrainMetrics


class ER(NaiveFineTune):
    """Experience Replay with reservoir-sampled buffer."""

    name = "er"

    def __init__(self, model, config):
        super().__init__(model, config)
        capacity = config.get("buffer_size", 200)
        self.state.buffer = ReservoirBuffer(capacity=capacity, store_logits=False)
        self.replay_batch_size = config.get("replay_batch_size", 8)

    def train_task(self, task: TaskInfo, train_loader: DataLoader) -> TrainMetrics:
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.trainable_parameters(),
            lr=self.config.get("lr", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)

        total_loss = 0.0
        n_steps = 0
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"ER T{task.task_id} ep{epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}

                optimizer.zero_grad()
                # Forward on current task batch
                cur_out = self.model(**batch)
                ce_loss = cur_out.loss
                replay_loss = torch.zeros((), device=self.device)

                # Forward on replay batch (if buffer non-empty)
                replay_batch = self.state.buffer.sample(self.replay_batch_size)
                if replay_batch is not None:
                    replay_batch = {k: v.to(self.device) for k, v in replay_batch.items()}
                    replay_out = self.model(**replay_batch)
                    replay_loss = replay_out.loss

                loss = ce_loss + replay_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), max_grad_norm)
                optimizer.step()

                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix(
                    {"ce": f"{ce_loss.item():.3f}", "replay": f"{replay_loss.item():.3f}"}
                )

        return TrainMetrics(
            task_id=task.task_id,
            loss=total_loss / max(n_steps, 1),
            n_steps=n_steps,
        )

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Add this task's examples to the buffer via reservoir sampling."""
        for batch in train_loader:
            self.state.buffer.add_batch(batch)
