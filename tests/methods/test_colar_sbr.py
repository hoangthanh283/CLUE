"""CoLaR-SBR (session-blocked readout) — synthetic-tensor checks, no backbone."""

import torch
import torch.nn.functional as F

from doccl.methods.colar_sbr import aggregate_blocks, block_targets, supervision_plan


def test_block_targets_maps_o_inblock_and_other_block():
    labels = torch.tensor([[0, 3, 4, 7, -100]])
    idx = torch.tensor([3, 4])  # block owns head ids 3, 4
    tgt = block_targets(labels, idx, n_head=9)
    assert tgt.tolist() == [[0, 1, 2, 0, -100]]  # annotated other-class tag 7 = true negative


def test_single_block_ce_equals_plain_ce_on_its_own_label_space():
    logits = torch.randn(2, 5, 3)  # [O_b, tag3, tag4] already block-shaped
    labels = torch.tensor([[0, 3, 4, 3, 0], [4, 0, 0, 3, 4]])
    tgt = block_targets(labels, torch.tensor([3, 4]), n_head=5)
    assert torch.equal(tgt, torch.where(labels == 0, 0, labels - 2))
    ce = F.cross_entropy(logits.reshape(-1, 3), tgt.reshape(-1), ignore_index=-100)
    assert torch.isfinite(ce)


def test_aggregate_blocks_claims_restricts_and_abstains():
    # two blocks: A owns {1,2}, B owns {3}; 4 tokens
    a = torch.full((1, 4, 3), -5.0)
    b = torch.full((1, 4, 2), -5.0)
    a[0, 0, 2] = 5.0  # token0: A says tag 2 confidently
    b[0, 1, 1] = 5.0  # token1: B says tag 3 confidently
    a[0, 2, 0] = 5.0  # token2: A says O_A, B says nothing -> O
    b[0, 2, 0] = 5.0
    a[0, 3, 1] = 2.0  # token3: both claim, B more confident
    b[0, 3, 1] = 4.0
    idx = [torch.tensor([1, 2]), torch.tensor([3])]
    preds = aggregate_blocks([a, b], idx, 0.5, None, n_head=4)
    assert preds.tolist() == [[2, 3, 0, 3]]
    restricted = aggregate_blocks([a, b], idx, 0.5, {0, 1, 2}, n_head=4)
    assert restricted.tolist() == [[2, 0, 0, 0]]  # B's claims mapped to O after the decision


def test_supervision_plan_cil_dil_and_mixed():
    # CIL-CORD: same schema, disjoint sessions -> each session trains only its own block
    blocks, doms = [{1, 2}, {3, 4}], ["cord", "cord"]
    assert supervision_plan({3, 4}, "cord", blocks, doms) == [(1, "full")]
    # DIL: one block covering every tag; a later domain re-annotates all of it -> full
    assert supervision_plan({1, 2, 3}, "sroie", [{1, 2, 3}], ["funsd"]) == [(0, "full")]
    # Mixed: SROIE docs are true negatives for FUNSD blocks; a FUNSD session skips its sibling
    blocks, doms = [{1, 2}, {3}, {4, 5}], ["funsd", "funsd", "sroie"]
    assert supervision_plan({4, 5}, "sroie", blocks, doms) == [(0, "neg"), (1, "neg"), (2, "full")]
    assert supervision_plan({1, 2}, "funsd", blocks, doms) == [(0, "full"), (2, "neg")]
    # FUNSD revisit annotates both FUNSD blocks
    assert supervision_plan({1, 2, 3}, "funsd", blocks, doms) == [
        (0, "full"),
        (1, "full"),
        (2, "neg"),
    ]
    # unknown domain -> never assume cross-schema negatives
    assert supervision_plan({4, 5}, "", blocks, doms) == [(2, "full")]


def test_offline_calibration_default_rule_matches_aggregate_blocks():
    """scripts/rca/sbr_calibrate.py (zero bias, tau 0.5) must reproduce the deployed rule."""
    import importlib.util

    import numpy as np

    from doccl.methods.colar_sbr import block_scores

    spec = importlib.util.spec_from_file_location("cal", "scripts/rca/sbr_calibrate.py")
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    g = torch.Generator().manual_seed(0)
    idx = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    bls = [torch.randn(2, 7, 1 + len(i), generator=g) * 3 for i in idx]
    for allowed in (None, {0, 3, 4, 5}):
        ref = aggregate_blocks(bls, idx, 0.5, allowed, 6)
        e, tags = block_scores(bls, idx)
        d = {"e": e.reshape(-1, 2), "tag": tags.reshape(-1, 2), "allowed": allowed, "n_head": 6}
        d["allowed"] = sorted(allowed) if allowed is not None else None
        assert torch.equal(cal.predict(d, np.zeros(2), 0.5), ref.reshape(-1))
