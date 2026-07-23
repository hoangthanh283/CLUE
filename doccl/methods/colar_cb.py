"""CoLaR-CB — RCA-balanced loss on replayed whole documents."""

from __future__ import annotations

import random

import torch
from torch.nn import functional

from doccl.methods.colar import CoLaR

__all__ = ["CoLaRCB"]


class CoLaRCB(CoLaR):
    """Balance old tasks and sparse labels without changing CoLaR's store."""

    name = "colar_cb"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.replay_balance_power = float(config.get("replay_balance_power", 0.5))
        self._bank_task_id = 0
        self._store_task_ids: list[int] = []

    def after_task(self, task, train_loader) -> None:
        self._bank_task_id = task.task_id
        super().after_task(task, train_loader)

    def _capture_task(self, train_loader) -> None:
        n_before = len(self.store)
        super()._capture_task(train_loader)
        self._store_task_ids.extend([self._bank_task_id] * (len(self.store) - n_before))

    def _sample_indices(self) -> list[int]:
        if not self.config.get("replay_task_balance", False) or len(self._store_task_ids) != len(
            self.store
        ):
            return super()._sample_indices()

        groups: dict[int, list[int]] = {}
        for index, task_id in enumerate(self._store_task_ids):
            groups.setdefault(task_id, []).append(index)
        pools = [list(indices) for _, indices in sorted(groups.items())]
        for pool in pools:
            random.shuffle(pool)
        random.shuffle(pools)
        chosen: list[int] = []
        quota = min(self.replay_batch_size, len(self.store))
        while len(chosen) < quota:
            for pool in pools:
                if pool and len(chosen) < quota:
                    chosen.append(pool.pop())
        return chosen

    def _replay_forward(self, replay: dict[str, torch.Tensor]):
        out = super()._replay_forward(replay)
        labels = replay["labels"].to(self.device)
        valid = labels != -100
        if not valid.any() or self.replay_balance_power == 0:
            return out

        all_labels = torch.cat([doc["labels"].reshape(-1) for doc in self.store])
        all_labels = all_labels[all_labels != -100]
        n_classes = out.logits.shape[-1]
        counts = torch.bincount(all_labels, minlength=n_classes)[:n_classes].to(
            device=self.device, dtype=out.logits.dtype
        )
        weights = counts.clamp_min(1).pow(-self.replay_balance_power)
        weights[counts == 0] = 0
        loss = functional.cross_entropy(
            out.logits.reshape(-1, n_classes),
            labels.reshape(-1),
            weight=weights,
            ignore_index=-100,
        )
        return type("Out", (), {"loss": loss, "logits": out.logits})()
