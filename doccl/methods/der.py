"""Dark Experience Replay++ (DER++).

Reference: Buzzega et al., "Dark Experience for General Continual Learning:
a Strong, Simple Baseline", NeurIPS 2020, arXiv:2004.07211.

Loss: L = L_CE(current) + α * MSE(student_logits_replay, cached_logits_replay)
                       + β * L_CE(replay_with_labels)

The MSE term distills the dark knowledge (full output distribution) from the
moment a sample was observed; the CE term keeps replay grounded in true labels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.buffer import ReservoirBuffer
from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo, TrainMetrics


class DERpp(NaiveFineTune):
    """Dark Experience Replay++ — strongest replay baseline."""

    name = "der_pp"

    def __init__(self, model, config):
        super().__init__(model, config)
        capacity = config.get("buffer_size", 200)
        self.state.buffer = ReservoirBuffer(capacity=capacity, store_logits=True)
        self.replay_batch_size = config.get("replay_batch_size", 8)
        self.alpha = config.get("alpha", 0.5)  # MSE weight
        self.beta = config.get("beta", 0.5)  # CE weight

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
            pbar = tqdm(train_loader, desc=f"DER++ T{task.task_id} ep{epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}

                optimizer.zero_grad()
                # Current task forward — capture logits for buffer insertion
                cur_out = self.model(**batch)
                ce_loss = cur_out.loss
                cur_logits = cur_out.logits

                mse_loss = torch.zeros((), device=self.device)
                replay_ce_loss = torch.zeros((), device=self.device)

                # Replay (if buffer non-empty) — sample TWO batches per DER++ paper
                replay1 = self.state.buffer.sample(self.replay_batch_size)
                replay2 = self.state.buffer.sample(self.replay_batch_size)

                if replay1 is not None:
                    replay1 = {k: v.to(self.device) for k, v in replay1.items()}
                    cached_logits = replay1.pop("_logits")
                    r_out = self.model(**{k: v for k, v in replay1.items() if k != "labels"})
                    # MSE — only on shared output dims (handle classifier expansion)
                    n_shared = min(r_out.logits.shape[-1], cached_logits.shape[-1])
                    mse_loss = F.mse_loss(
                        r_out.logits[..., :n_shared], cached_logits[..., :n_shared]
                    )

                if replay2 is not None:
                    replay2 = {k: v.to(self.device) for k, v in replay2.items()}
                    replay2.pop("_logits", None)  # don't need logits for CE replay
                    r_out2 = self.model(**replay2)
                    replay_ce_loss = r_out2.loss

                loss = ce_loss + self.alpha * mse_loss + self.beta * replay_ce_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), max_grad_norm)
                optimizer.step()

                # Add current batch to buffer with its logits
                self.state.buffer.add_batch(
                    {k: v for k, v in batch.items() if torch.is_tensor(v)},
                    logits=cur_logits,
                )

                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix({
                    "ce": f"{ce_loss.item():.3f}",
                    "mse": f"{mse_loss.item():.3f}",
                    "rce": f"{replay_ce_loss.item():.3f}",
                })

        return TrainMetrics(
            task_id=task.task_id,
            loss=total_loss / max(n_steps, 1),
            n_steps=n_steps,
        )
