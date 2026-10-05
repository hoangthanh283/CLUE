"""Latent replay must give identical replay gradients with and without gradient checkpointing.

HF checkpointing recomputes ``layer.__call__`` in backward, re-firing the injection pre-hook
after the injection was cleared — the recompute then used the dummy input, silently
corrupting layer-k gradients on every replay step.
"""

from __future__ import annotations

import torch
from transformers import ViTConfig, ViTForImageClassification

from doccl.methods.latent_replay import LatentReplay
from doccl.models.vit_wrapper import ViTWrapper


def _tiny_vit(tmp_path):
    cfg = ViTConfig(
        image_size=32,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
        num_labels=3,
    )
    ViTForImageClassification(cfg).save_pretrained(tmp_path)
    return ViTWrapper(model_name=str(tmp_path), num_labels=3)


def _replay_grads(model, method, replay):
    model.zero_grad(set_to_none=True)
    method._replay_forward(replay).loss.backward()
    layer_k = method._encoder_layers()[method.split_layer_k]
    return [p.grad.clone() for p in layer_k.parameters()]


def test_replay_grads_identical_with_checkpointing(tmp_path):
    torch.manual_seed(0)
    model = _tiny_vit(tmp_path)
    model.train()
    m = LatentReplay(model, {"split_layer_k": 2, "docs_per_task": 2, "replay_batch_size": 2})
    m.device = torch.device("cpu")
    m._pixel_shape = (3, 32, 32)
    replay = {"hidden": torch.randn(2, 17, 32), "labels": torch.tensor([0, 2])}

    plain = _replay_grads(model, m, replay)
    model.enable_gradient_checkpointing()
    model.train()
    ckpt = _replay_grads(model, m, replay)

    for a, b in zip(plain, ckpt, strict=True):
        assert torch.allclose(a, b, atol=1e-6), "replay gradients differ under checkpointing"
