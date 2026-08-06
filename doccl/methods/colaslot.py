"""CoLaR with task-isolated additive slots.

This is the smallest clean test of whether task-specific capacity can move CoLaR beyond
its retention frontier. It intentionally disables lexical inference routing: CoLaR's
replay pass uses dummy token ids, so routing from ``input_ids`` would train/evaluate the
stored document under different slot states.
"""

from __future__ import annotations

from doccl.methods.colar import CoLaR
from doccl.methods.lexslot import LexSlot

__all__ = ["CoLaSlot"]


class CoLaSlot(CoLaR, LexSlot):
    """Compressed latent replay plus ungated, gradient-isolated head/late slots."""

    name = "colaslot"

    def __init__(self, model, config):
        if config.get("infer_gate", False):
            raise ValueError("colaslot inference routing needs original replay input_ids")
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
