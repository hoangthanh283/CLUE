"""CoLaR with post-task, retention-only lexical head residuals."""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from doccl.methods.colaslot import CoLaSlot
from doccl.types import EvalMetrics, TaskInfo, TrainMetrics

log = logging.getLogger(__name__)

__all__ = ["CoLaSlotRF"]


class CoLaSlotRF(CoLaSlot):
    """Keep CoLaR training unchanged; refit lexical residuals after shared-head drift."""

    name = "colaslot_rf"

    def __init__(self, model, config):
        self._active_task_id = -1
        self._bank_task_id = -1
        self._store_task_ids: list[int] = []
        self._slot_reads_enabled = False
        self._forced_slot_owner: int | None = None
        self.diagnostic_metrics: dict = {"base_only_by_stage": {}, "refit": {}}
        super().__init__(model, config)
        if self.slot_depth != "head_only":
            raise ValueError("colaslot_rf requires slot_depth=head_only")
        if not self.infer_gate or not config.get("store_input_ids", False):
            raise ValueError("colaslot_rf requires infer_gate=true and store_input_ids=true")
        self.refit_epochs = int(config.get("refit_epochs", 10))
        self.refit_lr = float(config.get("refit_lr", 1e-3))
        self.refit_null_weight = float(config.get("refit_null_weight", 1.0))
        if self.refit_epochs < 1 or self.refit_lr <= 0 or self.refit_null_weight < 0:
            raise ValueError("colaslot_rf refit settings must be non-negative and epochs/lr > 0")
        self._slot_reads_enabled = True

    def _slot_parameters(self) -> list[torch.nn.Parameter]:
        return [p for module in self._all_slot_modules() for p in module.parameters()]

    def trainable_parameters(self):
        """Normal CoLaR optimizer excludes every residual parameter."""
        slot_ids = (
            {id(p) for p in self._slot_parameters()} if hasattr(self, "head_slots") else set()
        )
        return [p for p in self.model.parameters() if p.requires_grad and id(p) not in slot_ids]

    def _set_owner_gate(self, batch_size: int, device: torch.device, owner_id: int | None) -> None:
        for module in self._all_slot_modules():
            owners = torch.as_tensor(module.slot_owner, device=device)
            if owner_id is None:
                module._infer_gate = torch.zeros(batch_size, module.n_slots, device=device)
            else:
                module._infer_gate = (
                    (owners == owner_id).float().unsqueeze(0).expand(batch_size, -1)
                )

    def _install_infer_gate(self, module, args, kwargs) -> None:
        ids = kwargs.get("input_ids")
        if not torch.is_tensor(ids):
            return super()._install_infer_gate(module, args, kwargs)
        if self._forced_slot_owner is not None:
            self._set_owner_gate(ids.shape[0], ids.device, self._forced_slot_owner)
            return
        if not self._slot_reads_enabled:
            self._set_owner_gate(ids.shape[0], ids.device, None)
            return

        super()._install_infer_gate(module, args, kwargs)
        for slot_module in self._all_slot_modules():
            gate = slot_module._infer_gate
            if gate is None:
                continue
            owners = torch.as_tensor(slot_module.slot_owner, device=gate.device)
            gate[:, owners >= self._active_task_id] = 0

    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        self._active_task_id = task.task_id
        super().before_task(task, train_loader)

    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        self._slot_reads_enabled = False
        try:
            return super().train_task(task, train_loader, val_loader)
        finally:
            self._slot_reads_enabled = True

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        self._bank_task_id = task.task_id
        super().after_task(task, train_loader)
        self._refit_prior_slots(task.task_id)

    def _capture_task(self, train_loader: DataLoader) -> None:
        before = len(self.store)
        super()._capture_task(train_loader)
        self._store_task_ids.extend([self._bank_task_id] * (len(self.store) - before))

    def _positive_refit_loss(
        self, replay: dict[str, torch.Tensor], docs: list[dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        return self._replay_forward(replay).loss

    def _refit_prior_slots(self, current_task_id: int) -> None:
        owners = sorted({owner for owner in self._store_task_ids if owner < current_task_id})
        if not owners:
            return

        was_training = self.model.training
        original_requires_grad = [(p, p.requires_grad) for p in self.model.parameters()]
        original_mask = self.head_slots.grad_mask.detach().clone()
        slot_params = self._slot_parameters()
        for parameter, _ in original_requires_grad:
            parameter.requires_grad = False
        for parameter in slot_params:
            parameter.requires_grad = True
        self.model.eval()

        try:
            for owner in owners:
                positives = [
                    doc
                    for doc, task_id in zip(self.store, self._store_task_ids, strict=True)
                    if task_id == owner
                ]
                negatives = [
                    doc
                    for doc, task_id in zip(self.store, self._store_task_ids, strict=True)
                    if task_id != owner
                ]
                mask = torch.tensor(
                    [slot_owner == owner for slot_owner in self.head_slots.slot_owner],
                    device=self.head_slots.grad_mask.device,
                    dtype=self.head_slots.grad_mask.dtype,
                )
                self.head_slots.set_grad_mask(mask)
                optimizer = torch.optim.AdamW(slot_params, lr=self.refit_lr, weight_decay=0.0)
                positive_loss = 0.0
                null_loss = 0.0
                steps = 0
                self._forced_slot_owner = owner

                for _ in range(self.refit_epochs):
                    for start in range(0, len(positives), self.replay_batch_size):
                        positive_docs = positives[start : start + self.replay_batch_size]
                        replay = self._stack_replay(positive_docs)
                        optimizer.zero_grad(set_to_none=True)
                        loss = self._positive_refit_loss(replay, positive_docs)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            slot_params, self.config.get("max_grad_norm", 1.0)
                        )
                        optimizer.step()
                        positive_loss += float(loss.item())
                        steps += 1

                    for start in range(0, len(negatives), self.replay_batch_size):
                        replay = self._stack_replay(
                            negatives[start : start + self.replay_batch_size]
                        )
                        optimizer.zero_grad(set_to_none=True)
                        self._replay_forward(replay)
                        delta = self.head_slots.logits_delta(self._cur_feats)
                        valid = replay["attention_mask"].to(delta.device).bool().unsqueeze(-1)
                        denom = valid.sum().clamp_min(1) * delta.shape[-1]
                        loss = self.refit_null_weight * (delta.square() * valid).sum() / denom
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            slot_params, self.config.get("max_grad_norm", 1.0)
                        )
                        optimizer.step()
                        null_loss += float(loss.item())

                self.diagnostic_metrics["refit"][str(current_task_id) + ":" + str(owner)] = {
                    "positive_loss": positive_loss / max(steps, 1),
                    "null_loss": null_loss / max(self.refit_epochs, 1),
                    "positive_docs": len(positives),
                    "negative_docs": len(negatives),
                }
                log.info(
                    "colaslot_rf: refit owner=%d after task=%d on %d positive/%d negative docs",
                    owner,
                    current_task_id,
                    len(positives),
                    len(negatives),
                )
        finally:
            self._forced_slot_owner = None
            self.head_slots.set_grad_mask(original_mask)
            for parameter, requires_grad in original_requires_grad:
                parameter.requires_grad = requires_grad
            if was_training:
                self.model.train()

    def evaluate(self, eval_loaders: dict[int, DataLoader]) -> dict[int, EvalMetrics]:
        results = super().evaluate(eval_loaders)
        expected = set(range(self._active_task_id + 1))
        if self._active_task_id >= 0 and set(eval_loaders) == expected:
            self._slot_reads_enabled = False
            rng_state = torch.random.get_rng_state()
            try:
                base = super().evaluate(eval_loaders)
            finally:
                torch.random.set_rng_state(rng_state)
                self._slot_reads_enabled = True
            self.diagnostic_metrics["base_only_by_stage"][str(self._active_task_id)] = {
                str(task_id): metric.f1 for task_id, metric in base.items()
            }
        return results

    def memory_bytes(self) -> int:
        return super().memory_bytes() + len(self._store_task_ids) * 8
