"""CoLaR-SBR + LexSlot — the session-blocked readout with lexically-routed head slots on top.

User-requested combination (2026-11-09). Composes ``CoLaRSBR`` with ``CoLaSlot`` (CoLaR +
LexSlot's gradient-isolated, OCR-routed slots): the slot bias lands on the shared tag logits
that every block reads, while each block's own background logit stays slot-free. Expected
inert — SBR already partitions the head per session with a label-correct rule, and LexSlot
alone scores 19.0 on CIL-CORD — so this exists for a "we tried it" line.
"""

from __future__ import annotations

from doccl.methods.colar_sbr import CoLaRSBR
from doccl.methods.colaslot import CoLaSlot

__all__ = ["CoLaRSBRSlot"]


class CoLaRSBRSlot(CoLaRSBR, CoLaSlot):
    """MRO: SBR losses/readout -> CoLaSlot routing + replay-id plumbing -> CoLaR -> LexSlot."""

    name = "colar_sbr_slot"
