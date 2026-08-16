"""Owner-utility-gated prototype readout."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as functional

from doccl.methods.colaslot_proto import CoLaSlotProto

__all__ = ["CoLaSlotProtoUtility"]

log = logging.getLogger(__name__)


class CoLaSlotProtoUtility(CoLaSlotProto):
    """Keep only prototype owners that improve leave-one-document-out replay F1."""

    name = "colaslot_proto_u"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.diagnostic_metrics["prototype_utility"] = {}

    @staticmethod
    def _entity_f1(predicted: torch.Tensor, labels: torch.Tensor, o_id: int) -> float:
        entity = labels.ne(o_id)
        correct = predicted.eq(labels)
        true_positive = int((correct & entity).sum())
        false_positive = int((predicted.ne(o_id) & ~correct).sum())
        false_negative = int((entity & ~correct).sum())
        return 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)

    @staticmethod
    def _fold_prototypes(feats: torch.Tensor, labels: torch.Tensor) -> dict[int, torch.Tensor]:
        return {
            int(cls): functional.normalize(feats[labels.eq(cls)].mean(dim=0), dim=0)
            for cls in labels.unique().tolist()
        }

    @staticmethod
    def _fold_scale(
        prototypes: dict[int, torch.Tensor],
        feats: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        o_id: int,
    ) -> float | None:
        if len(prototypes) < 2 or o_id not in prototypes:
            return None
        margins = []
        proto_margins = []
        for cls, proto in prototypes.items():
            mask = labels.eq(cls)
            if cls == o_id or not mask.any():
                continue
            margins.append((logits[:, cls] - logits[:, o_id])[mask].abs())
            proto_margins.append((feats @ proto - feats @ prototypes[o_id])[mask].abs())
        if not margins:
            return None
        return torch.cat(margins).mean().item() / max(torch.cat(proto_margins).mean().item(), 1e-6)

    def _owner_utility(
        self, owner: int, docs: list[dict[str, torch.Tensor]]
    ) -> tuple[float, float]:
        if len(docs) < 2:
            return 0.0, 0.0
        records = []
        for doc in docs:
            replay = self._stack_replay([doc])
            output = self._replay_forward(replay)
            labels = replay["labels"].to(self.device)
            valid = replay["attention_mask"].to(self.device).bool() & labels.ne(-100)
            records.append(
                (
                    functional.normalize(self._cur_feats[valid].float(), dim=-1).cpu(),
                    labels[valid].cpu(),
                    output.logits[valid].float().cpu(),
                )
            )

        base_predictions = []
        utility_predictions = []
        held_labels = []
        rows = sum(slot_owner == owner for slot_owner in self.head_slots.slot_owner)
        o_id = int(getattr(self.model, "label_to_id", {}).get("O", 0))
        for held_out, (feats, labels, logits) in enumerate(records):
            train = [record for index, record in enumerate(records) if index != held_out]
            train_feats = torch.cat([record[0] for record in train])
            train_labels = torch.cat([record[1] for record in train])
            train_logits = torch.cat([record[2] for record in train])
            prototypes = self._fold_prototypes(train_feats, train_labels)
            scale = self._fold_scale(prototypes, train_feats, train_labels, train_logits, o_id)
            corrected = logits.clone()
            keys = self._contrast_keys([(record[0], record[1]) for record in train], o_id, rows)
            if scale is not None and keys.numel():
                support = (feats @ keys.T).max(dim=-1).values.gt(0)
                o_score = feats @ prototypes[o_id]
                for cls, proto in prototypes.items():
                    if cls != o_id:
                        corrected[:, cls] += support * scale * (feats @ proto - o_score)
            base_predictions.append(logits.argmax(dim=-1))
            utility_predictions.append(corrected.argmax(dim=-1))
            held_labels.append(labels)

        labels = torch.cat(held_labels)
        return (
            self._entity_f1(torch.cat(base_predictions), labels, o_id),
            self._entity_f1(torch.cat(utility_predictions), labels, o_id),
        )

    def _fit_prototypes(self, task_id: int) -> None:
        super()._fit_prototypes(task_id)
        if not self._prototypes:
            return

        was_training = self.model.training
        reads_enabled = self._slot_reads_enabled
        rng_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=rng_devices):
            self.model.eval()
            self._slot_reads_enabled = False
            try:
                for owner in list(self._prototypes):
                    docs = [
                        doc
                        for doc, doc_owner in zip(self.store, self._store_task_ids, strict=True)
                        if doc_owner == owner
                    ]
                    self._forced_slot_owner = owner
                    with torch.no_grad():
                        base_f1, utility_f1 = self._owner_utility(owner, docs)
                    enabled = utility_f1 > base_f1
                    self.diagnostic_metrics["prototype_utility"][str(owner)] = {
                        "base_f1": 100 * base_f1,
                        "prototype_f1": 100 * utility_f1,
                        "enabled": enabled,
                        "documents": len(docs),
                    }
                    log.info(
                        "colaslot_proto_u: owner=%d LODO base_f1=%.2f prototype_f1=%.2f enabled=%s",
                        owner,
                        100 * base_f1,
                        100 * utility_f1,
                        enabled,
                    )
                    if not enabled:
                        self._prototypes.pop(owner, None)
                        self._prototype_scale.pop(owner, None)
            finally:
                self._forced_slot_owner = None
                self._slot_reads_enabled = reads_enabled
                if was_training:
                    self.model.train()
