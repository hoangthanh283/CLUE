"""CoLaR with online, replay-only lexical residual tracking."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from doccl.methods.colaslot_rf import CoLaSlotRF
from doccl.types import TaskInfo, TrainMetrics

__all__ = ["CoLaSlotRO"]


class CoLaSlotRO(CoLaSlotRF):
    """Update prior-owner slots after each slot-free CoLaR optimizer step."""

    name = "colaslot_ro"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.online_lr = float(config.get("online_lr", 5e-5))
        self.online_weight_decay = float(config.get("online_weight_decay", 0.01))
        if self.online_lr <= 0 or self.online_weight_decay < 0:
            raise ValueError("colaslot_ro online optimizer settings must be non-negative")
        self._online_optimizer: torch.optim.Optimizer | None = None
        self._online_docs: list[dict[str, torch.Tensor]] = []
        self.diagnostic_metrics["online"] = {}

    def _sample_replay(self) -> dict[str, torch.Tensor] | None:
        if not self.store:
            self._online_docs = []
            return None
        indices = self._sample_indices()
        self._online_docs = [self.store[index] for index in indices]
        return self._stack_replay(self._online_docs)

    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        self._online_optimizer = torch.optim.AdamW(
            self._slot_parameters(),
            lr=self.online_lr,
            weight_decay=self.online_weight_decay,
        )
        try:
            return super().train_task(task, train_loader, val_loader)
        finally:
            self._online_optimizer = None
            self._online_docs = []

    def _post_optimizer_step(self) -> None:
        super()._post_optimizer_step()
        if self._online_optimizer is None or not self._online_docs:
            return

        owner_by_id = {
            id(doc): owner for doc, owner in zip(self.store, self._store_task_ids, strict=True)
        }
        docs_by_owner: dict[int, list[dict[str, torch.Tensor]]] = {}
        for doc in self._online_docs:
            owner = owner_by_id[id(doc)]
            if owner < self._active_task_id:
                docs_by_owner.setdefault(owner, []).append(doc)
        if not docs_by_owner:
            return

        was_training = self.model.training
        original_requires_grad = [
            (parameter, parameter.requires_grad) for parameter in self.model.parameters()
        ]
        original_mask = self.head_slots.grad_mask.detach().clone()
        slot_params = self._slot_parameters()
        for parameter, _ in original_requires_grad:
            parameter.requires_grad = False
        for parameter in slot_params:
            parameter.requires_grad = True
        self.model.eval()

        try:
            self._online_optimizer.zero_grad(set_to_none=True)
            active_rows = torch.zeros(
                self.head_slots.n_slots, device=self.head_slots.grad_mask.device, dtype=torch.bool
            )
            for owner, docs in docs_by_owner.items():
                mask = torch.tensor(
                    [slot_owner == owner for slot_owner in self.head_slots.slot_owner],
                    device=self.head_slots.grad_mask.device,
                    dtype=self.head_slots.grad_mask.dtype,
                )
                active_rows |= mask.bool()
                self.head_slots.set_grad_mask(mask)
                self._forced_slot_owner = owner
                loss = self._replay_forward(self._stack_replay(docs)).loss
                loss.backward()

                key = str(self._active_task_id) + ":" + str(owner)
                entry = self.diagnostic_metrics["online"].setdefault(
                    key, {"loss_sum": 0.0, "steps": 0}
                )
                entry["loss_sum"] += float(loss.item())
                entry["steps"] += 1
                entry["mean_loss"] = entry["loss_sum"] / entry["steps"]

            torch.nn.utils.clip_grad_norm_(slot_params, self.config.get("max_grad_norm", 1.0))
            inactive = ~active_rows
            inactive_values = [parameter.detach()[inactive].clone() for parameter in slot_params]
            self._online_optimizer.step()
            with torch.no_grad():
                for parameter, saved in zip(slot_params, inactive_values, strict=True):
                    parameter[inactive] = saved
        finally:
            self._forced_slot_owner = None
            self.head_slots.set_grad_mask(original_mask)
            for parameter, requires_grad in original_requires_grad:
                parameter.requires_grad = requires_grad
            if was_training:
                self.model.train()

    def _refit_prior_slots(self, current_task_id: int) -> None:
        """Online tracking replaces the post-task RF refit."""
