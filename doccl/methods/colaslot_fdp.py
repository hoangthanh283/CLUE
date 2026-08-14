"""Functional-drift slots with token-level pathway-energy routing."""

from __future__ import annotations

import torch

from doccl.methods.colaslot_fd import CoLaSlotFD

__all__ = ["CoLaSlotFDP"]


class CoLaSlotFDP(CoLaSlotFD):
    """Mix owner slots per token using their own rank-one activation energy."""

    name = "colaslot_fdp"

    def _head_delta(self, feats: torch.Tensor) -> torch.Tensor:
        act = feats @ self.head_slots.values.T
        gate = self.head_slots._infer_gate
        if gate is None:
            return act @ self.head_slots.proj

        routed = gate.unsqueeze(1).to(device=act.device, dtype=act.dtype)
        active = routed.gt(0)
        energy = act.square().masked_fill(~active, torch.finfo(act.dtype).min)
        weights = energy.softmax(dim=-1)
        weights = weights * active.sum(dim=-1, keepdim=True).to(act.dtype) * routed
        return (act * weights) @ self.head_slots.proj

    def _add_head_slots(self, _module, _inp, output):
        if self._cur_feats is None or self._cur_feats.shape[1] != output.shape[1]:
            return output
        return output + self._head_delta(self._cur_feats.to(output.dtype))
