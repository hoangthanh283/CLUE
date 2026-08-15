"""Support-gated analytic compensation for measured replay-logit drift."""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from doccl.methods.colaslot_fdp import CoLaSlotFDP
from doccl.types import TaskInfo

log = logging.getLogger(__name__)

__all__ = ["CoLaSlotFDA"]


class CoLaSlotFDA(CoLaSlotFDP):
    """Fit token-gated owner residuals by a tiny relative-ridge solve."""

    name = "colaslot_fda"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.analytic_ridge = float(config.get("analytic_ridge", 1e-3))
        self.support_min_precision = float(config.get("support_min_precision", 0.9))
        if self.analytic_ridge <= 0 or not 0 <= self.support_min_precision <= 1:
            raise ValueError("colaslot_fda needs analytic_ridge > 0 and support precision in [0,1]")
        self._support_keys: dict[int, torch.Tensor] = {}
        self._support_enabled: dict[int, bool] = {}
        self.diagnostic_metrics["analytic"] = {}
        self.diagnostic_metrics["support_gate"] = {}

    def _token_support(self, feats: torch.Tensor) -> torch.Tensor:
        support = torch.zeros(feats.shape[:2], device=feats.device, dtype=torch.bool)
        route = self.head_slots._infer_gate
        if route is None:
            return support
        owners = torch.as_tensor(self.head_slots.slot_owner, device=route.device)
        normalized = functional.normalize(feats.float(), dim=-1)
        for owner, keys in self._support_keys.items():
            rows = owners.eq(owner)
            selected = route[:, rows].gt(0).any(dim=1)
            if not self._support_enabled.get(owner, False) or not selected.any():
                continue
            scores = normalized[selected] @ keys.to(feats.device).T
            support[selected] = scores.max(dim=-1).values.gt(0)
        return support

    def _gated_pathway_features(self, feats: torch.Tensor) -> torch.Tensor:
        pathways = self._pathway_features(feats)
        return pathways * self._token_support(feats).unsqueeze(-1).to(pathways.dtype)

    def _head_delta(self, feats: torch.Tensor) -> torch.Tensor:
        return self._gated_pathway_features(feats) @ self.head_slots.proj

    @staticmethod
    def _contrast_keys(
        docs: list[tuple[torch.Tensor, torch.Tensor]], o_id: int, max_keys: int
    ) -> torch.Tensor:
        if not docs or max_keys < 1:
            return torch.empty(0, docs[0][0].shape[-1] if docs else 0)
        feats = torch.cat([doc[0] for doc in docs])
        labels = torch.cat([doc[1] for doc in docs])
        o_feats = feats[labels.eq(o_id)]
        if not o_feats.numel():
            return feats.new_empty((0, feats.shape[-1]))
        o_mean = o_feats.mean(dim=0)
        keys = []
        for class_id in sorted(set(labels.tolist()) - {o_id}):
            class_feats = feats[labels.eq(class_id)]
            if class_feats.numel():
                contrast = class_feats.mean(dim=0) - o_mean
                if contrast.norm() > 1e-8:
                    keys.append(functional.normalize(contrast, dim=0))
            if len(keys) == max_keys:
                break
        return torch.stack(keys) if keys else feats.new_empty((0, feats.shape[-1]))

    def _fit_support_gate(
        self, owner: int, docs: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, float | int | bool]:
        rows = sum(slot_owner == owner for slot_owner in self.head_slots.slot_owner)
        o_id = int(getattr(self.model, "label_to_id", {}).get("O", 0))
        true_positive = false_positive = false_negative = 0
        if len(docs) > 1:
            for held_out in range(len(docs)):
                keys = self._contrast_keys(
                    [doc for index, doc in enumerate(docs) if index != held_out], o_id, rows
                )
                feats, labels = docs[held_out]
                predicted = (
                    (feats @ keys.T).max(dim=-1).values.gt(0)
                    if keys.numel()
                    else torch.zeros_like(labels, dtype=torch.bool)
                )
                entity = labels.ne(o_id)
                true_positive += int((predicted & entity).sum())
                false_positive += int((predicted & ~entity).sum())
                false_negative += int((~predicted & entity).sum())
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        keys = self._contrast_keys(docs, o_id, rows)
        enabled = bool(keys.numel() and precision >= self.support_min_precision and f1 > 0)
        self._support_keys[owner] = keys.cpu()
        self._support_enabled[owner] = enabled
        return {
            "precision": 100 * precision,
            "recall": 100 * recall,
            "f1": 100 * f1,
            "enabled": enabled,
            "keys": int(keys.shape[0]),
            "documents": len(docs),
        }

    def _refresh_support_gates(self) -> None:
        grouped: dict[int, list[dict[str, torch.Tensor]]] = {}
        for doc, owner in zip(self.store, self._store_task_ids, strict=True):
            grouped.setdefault(owner, []).append(doc)
        if not grouped:
            return

        was_training = self.model.training
        reads_enabled = self._slot_reads_enabled
        self.model.eval()
        self._slot_reads_enabled = False
        try:
            with torch.no_grad():
                for owner, stored_docs in grouped.items():
                    docs = []
                    for doc in stored_docs:
                        replay = self._stack_replay([doc])
                        self._replay_forward(replay)
                        labels = replay["labels"].to(self.device)
                        valid = replay["attention_mask"].to(self.device).bool() & labels.ne(-100)
                        docs.append(
                            (
                                functional.normalize(self._cur_feats[valid].float(), dim=-1).cpu(),
                                labels[valid].cpu(),
                            )
                        )
                    entry = self._fit_support_gate(owner, docs)
                    self.diagnostic_metrics["support_gate"][str(owner)] = entry
                    log.info(
                        "colaslot_fda: owner=%d LODO support precision=%.2f f1=%.2f enabled=%s",
                        owner,
                        entry["precision"],
                        entry["f1"],
                        entry["enabled"],
                    )
        finally:
            self._forced_slot_owner = None
            self._slot_reads_enabled = reads_enabled
            if was_training:
                self.model.train()

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        super().after_task(task, train_loader)
        self._refresh_support_gates()

    @staticmethod
    def _ridge_increment(
        entity_x: torch.Tensor,
        target: torch.Tensor,
        null_groups: list[torch.Tensor],
        ridge: float,
        null_weight: float,
    ) -> tuple[torch.Tensor, float, float] | None:
        if not entity_x.numel() or not target.numel():
            return None
        x_blocks = [entity_x / math.sqrt(entity_x.shape[0])]
        y_blocks = [target / math.sqrt(target.shape[0])]
        nonempty = [group for group in null_groups if group.numel()]
        for group in nonempty:
            scale = math.sqrt(null_weight / (len(nonempty) * group.shape[0]))
            x_blocks.append(group * scale)
            y_blocks.append(target.new_zeros((group.shape[0], target.shape[1])))
        x = torch.cat(x_blocks).float()
        y = torch.cat(y_blocks).float()
        gram = x.T @ x
        relative_scale = gram.diagonal().mean().clamp_min(1e-6)
        system = gram + ridge * relative_scale * torch.eye(gram.shape[0], device=gram.device)
        delta = torch.linalg.solve(system, x.T @ y)
        if not torch.isfinite(delta).all():
            return None
        before = float(target.float().square().mean())
        after = float((target.float() - entity_x.float() @ delta).square().mean())
        if after >= before:
            return None
        return delta, before, after

    def _post_optimizer_step(self) -> None:
        if self._online_optimizer is None or not self._online_docs:
            return
        owner_by_id = {
            id(doc): owner for doc, owner in zip(self.store, self._store_task_ids, strict=True)
        }
        grouped: dict[int, list[dict[str, torch.Tensor]]] = {}
        for doc in self._online_docs:
            owner = owner_by_id[id(doc)]
            if owner < self._active_task_id and self._support_enabled.get(owner, False):
                grouped.setdefault(owner, []).append(doc)
        if not grouped:
            return

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for owner, docs in grouped.items():
                    replay = self._stack_replay(docs)
                    self._forced_slot_owner = owner
                    logits = self._replay_forward(replay).logits
                    pathway = self._gated_pathway_features(self._cur_feats).float()
                    rows = torch.tensor(
                        [slot_owner == owner for slot_owner in self.head_slots.slot_owner],
                        device=pathway.device,
                        dtype=torch.bool,
                    )
                    labels = replay["labels"].to(logits.device)
                    valid = replay["attention_mask"].to(logits.device).bool() & labels.ne(-100)
                    o_id = int(getattr(self.model, "label_to_id", {}).get("O", 0))
                    entity = valid & labels.ne(o_id)
                    target = self._teacher_logits[owner].to(logits) - logits
                    target = target - target.mean(dim=-1, keepdim=True)
                    entity_x = pathway[entity][:, rows]
                    null_groups = [pathway[valid & labels.eq(o_id)][:, rows]]

                    if self._current_feats is not None and self._current_mask is not None:
                        self._set_owner_gate(
                            self._current_feats.shape[0], self._current_feats.device, owner
                        )
                        current = self._gated_pathway_features(self._current_feats).float()
                        null_groups.append(current[self._current_mask.bool()][:, rows])
                    for foreign_owner, feats in self._teacher_feats.items():
                        if foreign_owner == owner:
                            continue
                        self._set_owner_gate(feats.shape[0], feats.device, owner)
                        foreign = self._gated_pathway_features(feats).float()
                        null_groups.append(
                            foreign[self._teacher_masks[foreign_owner].bool()][:, rows]
                        )

                    solved = self._ridge_increment(
                        entity_x,
                        target[entity],
                        null_groups,
                        self.analytic_ridge,
                        self.functional_null_weight,
                    )
                    if solved is None:
                        continue
                    delta, before, after = solved
                    self.head_slots.proj[rows] += delta.to(self.head_slots.proj)
                    key = f"{self._active_task_id}:{owner}"
                    entry = self.diagnostic_metrics["analytic"].setdefault(
                        key, {"before_sum": 0.0, "after_sum": 0.0, "steps": 0}
                    )
                    entry["before_sum"] += before
                    entry["after_sum"] += after
                    entry["steps"] += 1
                    entry["cancellation"] = 1 - entry["after_sum"] / entry["before_sum"]
        finally:
            self._forced_slot_owner = None
            if was_training:
                self.model.train()

    def memory_bytes(self) -> int:
        return super().memory_bytes() + sum(
            keys.numel() * 4 for keys in self._support_keys.values()
        )
