"""CoLaR-SBR + LexSlot — composition order check (no backbone)."""

from doccl.methods.colar_sbr import CoLaRSBR
from doccl.methods.colar_sbr_slot import CoLaRSBRSlot
from doccl.methods.colaslot import CoLaSlot
from doccl.methods.lexslot import LexSlot


def test_mro_sbr_wraps_slot_wraps_lexslot():
    mro = CoLaRSBRSlot.__mro__
    assert mro.index(CoLaRSBR) < mro.index(CoLaSlot) < mro.index(LexSlot)
    # SBR owns the replay loss; CoLaSlot's override must still sit under it for routing ids.
    assert CoLaRSBRSlot._replay_forward is CoLaRSBR._replay_forward
    assert CoLaRSBRSlot._install_infer_gate is CoLaSlot._install_infer_gate
