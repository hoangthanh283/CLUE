"""Route-conditioned prototype readout for compressed latent replay."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from doccl.methods.colaslot_fda import CoLaSlotFDA
from doccl.types import TaskInfo

log = logging.getLogger(__name__)

__all__ = ["CoLaSlotProto"]


class CoLaSlotProto(CoLaSlotFDA):
    """Keep CoLaR training unchanged and add a frozen owner prototype margin."""

    name = "colaslot_proto"

    def __init__(self, model, config):
        super().__init__(model, config)
        self._prototypes: dict[int, dict[int, torch.Tensor]] = {}
        self._prototype_scale: dict[int, float] = {}
        self.diagnostic_metrics["prototype"] = {}

    def _post_optimizer_step(self) -> None:
        # Prototype readout is deliberately absent from shared-weight training.
        return None

    def _refit_prior_slots(self, _current_task_id: int) -> None:
        # Keep LexSlot parameters at their exact zero initialization.
        return None

    def _fit_prototypes(self, task_id: int) -> None:
        was_training = self.model.training
        reads_enabled = self._slot_reads_enabled
        grouped: dict[int, list[dict[str, torch.Tensor]]] = {}
        for doc, owner in zip(self.store, self._store_task_ids, strict=True):
            if owner < task_id and self._support_enabled.get(owner, False):
                grouped.setdefault(owner, []).append(doc)
        self._prototypes = {}
        self._prototype_scale = {}
        rng_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=rng_devices):
            self.model.eval()
            self._slot_reads_enabled = False
            try:
                for owner, docs in grouped.items():
                    replay = self._stack_replay(docs)
                    self._forced_slot_owner = owner
                    try:
                        with torch.no_grad():
                            _ = self._replay_forward(replay)
                            feats = functional.normalize(self._cur_feats.float(), dim=-1)
                            labels = replay["labels"].to(feats.device)
                            valid = replay["attention_mask"].to(feats.device).bool() & labels.ne(
                                -100
                            )
                            prototypes: dict[int, torch.Tensor] = {}
                            for cls in labels[valid].unique().tolist():
                                cls_mask = valid & labels.eq(int(cls))
                                if cls_mask.any():
                                    prototypes[int(cls)] = functional.normalize(
                                        feats[cls_mask].mean(dim=0), dim=0
                                    ).cpu()
                            if len(prototypes) < 2 or 0 not in prototypes:
                                continue
                            entity = valid & labels.ne(0)
                            base_logits = self._replay_forward(replay).logits.float()
                            margins = []
                            proto_margins = []
                            for cls, proto in prototypes.items():
                                if cls == 0:
                                    continue
                                mask = entity & labels.eq(cls)
                                if not mask.any():
                                    continue
                                margins.append(
                                    (base_logits[..., cls] - base_logits[..., 0])[mask].abs()
                                )
                                proto_margins.append(
                                    (
                                        feats @ proto.to(feats.device)
                                        - feats @ prototypes[0].to(feats.device)
                                    )[mask].abs()
                                )
                            if not margins:
                                continue
                            base_mag = torch.cat(margins).mean().item()
                            proto_mag = torch.cat(proto_margins).mean().item()
                            self._prototypes[owner] = prototypes
                            self._prototype_scale[owner] = base_mag / max(proto_mag, 1e-6)
                            self.diagnostic_metrics["prototype"][str(owner)] = {
                                "classes": len(prototypes),
                                "documents": len(docs),
                                "scale": self._prototype_scale[owner],
                            }
                    finally:
                        self._forced_slot_owner = None
            finally:
                self._slot_reads_enabled = reads_enabled
                if was_training:
                    self.model.train()

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        super().after_task(task, train_loader)
        self._fit_prototypes(task.task_id)

    def _owner_support(self, feats: torch.Tensor, owner: int) -> torch.Tensor:
        route = self.head_slots._infer_gate
        keys = self._support_keys.get(owner)
        if route is None or keys is None or not self._support_enabled.get(owner, False):
            return torch.zeros(feats.shape[:2], dtype=torch.bool, device=feats.device)
        owners = torch.as_tensor(self.head_slots.slot_owner, device=route.device)
        selected = route[:, owners.eq(owner)].gt(0).any(dim=1)
        support = torch.zeros(feats.shape[:2], dtype=torch.bool, device=feats.device)
        if selected.any():
            normalized = functional.normalize(feats[selected].float(), dim=-1)
            scores = normalized @ keys.to(feats.device).T
            support[selected] = scores.max(dim=-1).values.gt(0)
        return support

    def _head_delta(self, feats: torch.Tensor) -> torch.Tensor:
        if not self._slot_reads_enabled or not self._prototypes:
            return torch.zeros(
                *feats.shape[:-1],
                self.model.model.classifier.out_features,
                device=feats.device,
                dtype=feats.dtype,
            )
        normalized = functional.normalize(feats.float(), dim=-1)
        delta = torch.zeros(
            *feats.shape[:-1],
            self.model.model.classifier.out_features,
            device=feats.device,
            dtype=torch.float32,
        )
        for owner, prototypes in self._prototypes.items():
            support = self._owner_support(feats, owner)
            if not support.any():
                continue
            proto_o = prototypes.get(0)
            if proto_o is None:
                continue
            o_score = normalized @ proto_o.to(feats.device)
            scale = self._prototype_scale[owner]
            for cls, proto in prototypes.items():
                if cls == 0:
                    continue
                delta[..., cls] += (
                    support.to(delta.dtype)
                    * scale
                    * (normalized @ proto.to(feats.device) - o_score)
                )
        return delta.to(feats.dtype)

    def memory_bytes(self) -> int:
        total = super().memory_bytes()
        total += sum(
            proto.numel() * 4 for values in self._prototypes.values() for proto in values.values()
        )
        return total
