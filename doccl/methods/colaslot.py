"""CoLaR with task-isolated additive slots.

This composes CoLaR replay with additive slots. The historical ``colaslot`` config keeps
routing disabled; ``colaslot_r`` opts into stored replay token IDs, strict ownership,
padding-aware task-block routing, and low-confidence abstention.
"""

from __future__ import annotations

import torch

from doccl.methods.colar import CoLaR
from doccl.methods.lexslot import LexSlot

__all__ = ["CoLaSlot"]


class CoLaSlot(CoLaR, LexSlot):
    """Compressed latent replay plus ungated, gradient-isolated head/late slots."""

    name = "colaslot"

    def __init__(self, model, config):
        if config.get("infer_gate", False) and not config.get("store_input_ids", False):
            raise ValueError("colaslot inference routing needs store_input_ids=true")
        if config.get("freeze_lower", False):
            raise ValueError("colaslot must use CoLaR's split-layer freeze map")
        rng_state = torch.random.get_rng_state()
        try:
            super().__init__(model, config)
        finally:
            torch.random.set_rng_state(rng_state)
        self._routing_input_ids: torch.Tensor | None = None

        # LexSlot keeps slots on the method object. Register them on the model too so
        # EarlyStopper snapshots/restores them with the backbone and classifier.
        self.model.add_module("_colaslot_head_slots", self.head_slots)
        self.model.add_module("_colaslot_late_slots", self.late_slots)

    def before_task(self, task, train_loader) -> None:
        """Keep the shuffled LexSlot signature pass from changing CoLaR training RNG."""
        rng_state = torch.random.get_rng_state()
        try:
            super().before_task(task, train_loader)
        finally:
            torch.random.set_rng_state(rng_state)

    def _install_infer_gate(self, module, args, kwargs) -> None:
        if self._routing_input_ids is not None:
            kwargs = dict(kwargs, input_ids=self._routing_input_ids)
        super()._install_infer_gate(module, args, kwargs)

    def _replay_forward(self, replay: dict[str, torch.Tensor]):
        route_ids = replay.get("input_ids")
        if route_ids is None:
            return super()._replay_forward(replay)
        self._routing_input_ids = route_ids.to(self.device)
        try:
            return super()._replay_forward({k: v for k, v in replay.items() if k != "input_ids"})
        finally:
            self._routing_input_ids = None

    def trainable_parameters(self):
        # Slots are registered model children above; returning model parameters avoids
        # handing AdamW duplicate references through LexSlot.trainable_parameters().
        return [p for p in self.model.parameters() if p.requires_grad]
