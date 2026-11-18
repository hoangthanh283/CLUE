"""Latent replay on LiLT must inject the layout stream at layer k together with the text stream.

LiLT's two streams share attention scores in every layer. With only the text stream stored, the
layout stream of a replayed document is recomputed from the dummy token ids below k and the
replayed features above k no longer match what the model computes for the real document.
"""

from __future__ import annotations

import torch
from transformers import LiltConfig, LiltForTokenClassification

from doccl.methods.latent_replay import LatentReplay
from doccl.models.lilt_wrapper import LiLTWrapper


def _tiny_lilt(tmp_path):
    cfg = LiltConfig(
        vocab_size=50,
        hidden_size=48,  # divisible by 6 (box embeddings) and by 4 (layout width)
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=40,
        max_2d_position_embeddings=64,
        num_labels=3,
    )
    LiltForTokenClassification(cfg).save_pretrained(tmp_path)
    return LiLTWrapper(model_name=str(tmp_path), num_labels=3)


def test_lilt_replay_matches_direct_forward_only_with_layout_stream(tmp_path):
    torch.manual_seed(0)
    model = _tiny_lilt(tmp_path)
    for p in model.parameters():  # default init (std 0.02) gives near-uniform attention, under
        if p.dim() > 1:  # which the layout stream hardly depends on the tokens
            torch.nn.init.normal_(p, std=0.3)
    model.eval()
    m = LatentReplay(model, {"split_layer_k": 2, "docs_per_task": 2, "replay_batch_size": 2})
    m.device = torch.device("cpu")
    xy = torch.randint(0, 32, (2, 16, 2))
    batch = {
        "input_ids": torch.randint(3, 50, (2, 16)),
        "bbox": torch.cat([xy, xy + torch.randint(1, 32, (2, 16, 2))], dim=-1),
        "attention_mask": torch.ones(2, 16, dtype=torch.long),
        "labels": torch.randint(0, 3, (2, 16)),
    }
    with torch.no_grad():
        m._capture = []
        direct = model(**batch).logits
        hidden, m._capture = m._capture[0], None
        layout, m._capture_layout = m._capture_layout, None
        assert layout is not None and layout.shape[:2] == hidden.shape[:2]
        replay = {k: batch[k] for k in ("bbox", "attention_mask", "labels")} | {"hidden": hidden}
        text_only = (m._replay_forward(replay).logits - direct).abs().max()
        both = (m._replay_forward(replay | {"layout_hidden": layout}).logits - direct).abs().max()
    scale = direct.abs().max()
    assert both < 0.02 * scale, f"replay with both streams deviates from the direct forward: {both}"
    assert text_only > 10 * both, f"text-only replay reproduces the direct forward: {text_only}"
