"""CoLaR-SBR (session-blocked readout) — synthetic-tensor checks, no backbone."""

import torch
import torch.nn.functional as F

from doccl.methods.colar_sbr import aggregate_blocks, block_targets


def test_block_targets_maps_o_inblock_and_other_block():
    labels = torch.tensor([[0, 3, 4, 7, -100]])
    idx = torch.tensor([3, 4])  # block owns head ids 3, 4
    tgt = block_targets(labels, idx, n_head=9)
    assert tgt.tolist() == [[0, 1, 2, -100, -100]]


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
