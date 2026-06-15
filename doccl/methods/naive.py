"""Naive sequential FT and Joint multi-task baselines.

Naive: lower bound — sequential fine-tuning with no CL strategy. Exhibits maximal
       catastrophic forgetting.
Joint: upper bound — multi-task training on union of all task data. Oracle.

These two bracket the "forgetting gap" that motivates CL.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.eval.metrics import compute_token_f1
from doccl.methods.base import ContinualMethod
from doccl.types import EvalMetrics, TaskInfo, TrainMetrics


class NaiveFineTune(ContinualMethod):
    """Sequential fine-tuning, no CL. Lower bound."""

    name = "naive"

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
            pbar = tqdm(train_loader, desc=f"T{task.task_id} ep{epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), max_grad_norm)
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

    def evaluate(
        self, eval_loaders: dict[int, DataLoader]
    ) -> dict[int, EvalMetrics]:
        self.model.eval()
        results: dict[int, EvalMetrics] = {}

        # Build label map for seqeval — assume model wrapper exposes id_to_label
        id_to_label = getattr(self.model, "id_to_label", None)
        if not id_to_label:
            # Fallback: assume HuggingFace config has id2label
            id_to_label = {i: str(i) for i in range(self.model.model.config.num_labels)}

        with torch.no_grad():
            for tid, loader in eval_loaders.items():
                all_preds, all_labels = [], []
                for batch in loader:
                    batch = {
                        k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)
                    }
                    outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
                    preds = outputs.logits.argmax(dim=-1)  # (B, L)
                    labels = batch["labels"]

                    mask = labels != -100
                    all_preds.extend(preds[mask].cpu().tolist())
                    all_labels.extend(labels[mask].cpu().tolist())

                metrics = compute_token_f1(all_preds, all_labels, id_to_label)
                results[tid] = EvalMetrics(
                    task_id=tid,
                    f1=metrics["f1"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    n_samples=len(all_labels),
                )
        return results


class JointMultiTask(NaiveFineTune):
    """Joint multi-task training. Same training loop as Naive — but train_loader
    must be constructed externally to contain ALL tasks' data concatenated.
    """

    name = "joint"
