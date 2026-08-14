"""CoLaR with support-bounded one-step functional-drift residuals."""

from __future__ import annotations

import torch

from doccl.methods.colaslot_ro import CoLaSlotRO

__all__ = ["CoLaSlotFD"]


class CoLaSlotFD(CoLaSlotRO):
    """Cancel measured replay-logit drift without fitting already-satisfied labels."""

    name = "colaslot_fd"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.functional_null_weight = float(config.get("functional_null_weight", 1.0))
        if self.functional_null_weight < 0:
            raise ValueError("colaslot_fd functional_null_weight must be non-negative")
        self._latest_current_mask: torch.Tensor | None = None
        self._current_feats: torch.Tensor | None = None
        self._current_mask: torch.Tensor | None = None
        self._teacher_logits: dict[int, torch.Tensor] = {}
        self._teacher_feats: dict[int, torch.Tensor] = {}
        self._teacher_masks: dict[int, torch.Tensor] = {}
        self.diagnostic_metrics["functional"] = {}

    def _install_infer_gate(self, module, args, kwargs) -> None:
        attention_mask = kwargs.get("attention_mask")
        if (
            self._online_optimizer is not None
            and self._inject is None
            and self._forced_slot_owner is None
            and not self._slot_reads_enabled
            and torch.is_tensor(attention_mask)
        ):
            self._latest_current_mask = attention_mask.detach()
        super()._install_infer_gate(module, args, kwargs)

    def _sample_replay(self) -> dict[str, torch.Tensor] | None:
        self._current_feats = self._cur_feats.detach() if self._cur_feats is not None else None
        self._current_mask = self._latest_current_mask
        replay = super()._sample_replay()
        self._teacher_logits = {}
        self._teacher_feats = {}
        self._teacher_masks = {}
        if replay is None:
            return None

        owner_by_id = {
            id(doc): owner for doc, owner in zip(self.store, self._store_task_ids, strict=True)
        }
        docs_by_owner: dict[int, list[dict[str, torch.Tensor]]] = {}
        for doc in self._online_docs:
            owner = owner_by_id[id(doc)]
            if owner < self._active_task_id:
                docs_by_owner.setdefault(owner, []).append(doc)

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for owner, docs in docs_by_owner.items():
                    owner_replay = self._stack_replay(docs)
                    self._forced_slot_owner = owner
                    self._teacher_logits[owner] = self._replay_forward(owner_replay).logits.detach()
                    self._teacher_feats[owner] = self._cur_feats.detach()
                    self._teacher_masks[owner] = owner_replay["attention_mask"].detach()
        finally:
            self._forced_slot_owner = None
            if was_training:
                self.model.train()
        return replay

    @staticmethod
    def _masked_square_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = mask.to(device=values.device, dtype=torch.bool).unsqueeze(-1)
        denominator = valid.sum().clamp_min(1) * values.shape[-1]
        return (values.square() * valid).sum() / denominator

    def _head_delta(self, feats: torch.Tensor) -> torch.Tensor:
        """Head residual hook point; FDP only changes how existing slots are mixed."""
        return self.head_slots.logits_delta(feats)

    def _online_owner_loss(self, owner: int, docs: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        replay = self._stack_replay(docs)
        logits = self._replay_forward(replay).logits
        owner_delta = self._head_delta(self._cur_feats)
        teacher = self._teacher_logits[owner].to(device=logits.device, dtype=logits.dtype)
        labels = replay["labels"].to(logits.device)
        valid = replay["attention_mask"].to(logits.device).bool() & labels.ne(-100)
        o_id = int(getattr(self.model, "label_to_id", {}).get("O", 0))
        entity = valid & labels.ne(o_id)
        error = logits - teacher
        error = error - error.mean(dim=-1, keepdim=True)
        drift_loss = self._masked_square_mean(error, entity)

        null_losses = [self._masked_square_mean(owner_delta, valid & labels.eq(o_id))]
        if self._current_feats is not None and self._current_mask is not None:
            self._set_owner_gate(self._current_feats.shape[0], self._current_feats.device, owner)
            null_losses.append(
                self._masked_square_mean(self._head_delta(self._current_feats), self._current_mask)
            )
        for foreign_owner, feats in self._teacher_feats.items():
            if foreign_owner == owner:
                continue
            self._set_owner_gate(feats.shape[0], feats.device, owner)
            null_losses.append(
                self._masked_square_mean(
                    self._head_delta(feats), self._teacher_masks[foreign_owner]
                )
            )
        null_loss = torch.stack(null_losses).mean()
        loss = drift_loss + self.functional_null_weight * null_loss

        key = f"{self._active_task_id}:{owner}"
        entry = self.diagnostic_metrics["functional"].setdefault(
            key,
            {
                "drift_loss_sum": 0.0,
                "null_loss_sum": 0.0,
                "entity_tokens": 0,
                "steps": 0,
            },
        )
        entry["drift_loss_sum"] += float(drift_loss.item())
        entry["null_loss_sum"] += float(null_loss.item())
        entry["entity_tokens"] += int(entity.sum().item())
        entry["steps"] += 1
        entry["mean_drift_loss"] = entry["drift_loss_sum"] / entry["steps"]
        entry["mean_null_loss"] = entry["null_loss_sum"] / entry["steps"]
        return loss
