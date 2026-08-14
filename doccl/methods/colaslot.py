"""CoLaR with task-isolated additive slots.

This composes CoLaR replay with additive slots. The historical ``colaslot`` config keeps
routing disabled; ``colaslot_r`` opts into stored replay token IDs, strict ownership,
padding-aware task-block routing, and low-confidence abstention.
"""

from __future__ import annotations

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
        super().__init__(model, config)

        # LexSlot keeps slots on the method object. Register them on the model too so
        # EarlyStopper snapshots/restores them with the backbone and classifier.
        self.model.add_module("_colaslot_head_slots", self.head_slots)
        self.model.add_module("_colaslot_late_slots", self.late_slots)

    def trainable_parameters(self):
        # Slots are registered model children above; returning model parameters avoids
        # handing AdamW duplicate references through LexSlot.trainable_parameters().
        return [p for p in self.model.parameters() if p.requires_grad]
