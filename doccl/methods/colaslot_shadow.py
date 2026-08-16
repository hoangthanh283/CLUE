"""CoLaR with an exact-base, replay-trained low-rank shadow branch."""

from __future__ import annotations

import torch
from torch import nn

from doccl.methods.colaslot_sidecar import CoLaSlotSidecar
from doccl.methods.lexslot_memory import ReprSlots


class CoLaSlotShadow(CoLaSlotSidecar):
    """Route aged owners through a shared low-rank upper branch; keep CoLaR exact."""

    name = "colaslot_shadow"

    def __init__(self, model, config):
        super().__init__(model, config)
        layers = self._encoder_layers()
        rng_state = torch.random.get_rng_state()
        try:
            self.shadow_slots = nn.ModuleDict(
                {
                    str(i): ReprSlots(1, self.hidden_dim, self.repr_rank).to(self.device)
                    for i in range(self.split_layer_k, len(layers))
                }
            )
        finally:
            torch.random.set_rng_state(rng_state)
        self.model.add_module("_colaslot_shadow_slots", self.shadow_slots)
        self._shadow_hook_handles = [
            layers[i].register_forward_hook(self._make_layer_hook(self.shadow_slots[str(i)]))
            for i in range(self.split_layer_k, len(layers))
        ]

    def _slot_parameters(self):
        return super()._slot_parameters() + [
            parameter for module in self.shadow_slots.values() for parameter in module.parameters()
        ]

    @staticmethod
    def _apply_shadow_gate(shadow_slots, owner_gate: torch.Tensor | None) -> None:
        active = None
        if owner_gate is not None:
            active = owner_gate.ne(0).any(dim=1, keepdim=True).to(owner_gate.dtype)
        for module in shadow_slots.values():
            module._infer_gate = active

    def _install_infer_gate(self, module, args, kwargs) -> None:
        super()._install_infer_gate(module, args, kwargs)
        self._apply_shadow_gate(self.shadow_slots, self.head_slots._infer_gate)

    def _sidecar_replay_loss(self, replay):
        return self._replay_forward(replay).loss


class CoLaSlotShadowReplay(CoLaSlotShadow):
    """Train the shared shadow only on replay; current examples train owner heads."""

    name = "colaslot_shadow_replay"

    def _install_infer_gate(self, module, args, kwargs) -> None:
        super()._install_infer_gate(module, args, kwargs)
        if not self._stage_eval and self._routing_input_ids is None:
            for shadow in self.shadow_slots.values():
                if shadow._infer_gate is not None:
                    shadow._infer_gate.zero_()
