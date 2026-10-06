"""CoLaR++ components on a tiny offline ViT (no downloads)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import ViTConfig, ViTForImageClassification

from doccl.methods.colar_pp import CoLaRPP, _pool_tokens, _q8
from doccl.models.vit_wrapper import ViTWrapper
from doccl.types import TaskInfo


def _tiny_vit(tmp_path, n=4):
    cfg = ViTConfig(
        image_size=32,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
        num_labels=n,
    )
    ViTForImageClassification(cfg).save_pretrained(tmp_path)
    m = ViTWrapper(model_name=str(tmp_path), num_labels=n)
    m.id_to_label = {i: f"c{i}" for i in range(n)}
    m.label_to_id = {v: k for k, v in m.id_to_label.items()}
    return m


def _loader(labels, n=8):
    items = [
        {"pixel_values": torch.randn(3, 32, 32), "labels": torch.tensor(labels[i % len(labels)])}
        for i in range(n)
    ]
    return DataLoader(items, batch_size=4)


def _method(model, **cfg):
    base = {"split_layer_k": 2, "docs_per_task": 8, "replay_batch_size": 4, "rank_r": 0}
    m = CoLaRPP(model, {**base, **cfg})
    m.device = torch.device("cpu")
    return m


def test_q8_roundtrip_error_small():
    x = torch.randn(17, 32)
    q, s = _q8(x)
    rel = ((q.float() * s.float()) - x).norm() / x.norm()
    assert q.dtype == torch.int8 and rel < 0.01


def test_pool_tokens_shape():
    h = torch.randn(1 + 14 * 14, 768)
    assert _pool_tokens(h, 2).shape == (1 + 7 * 7, 768)
    assert torch.equal(_pool_tokens(h, 2)[0], h[0])


def test_head_alignment_changes_only_head(tmp_path):
    model = _tiny_vit(tmp_path)
    m = _method(model, head_align_epochs=2)
    trunk0 = {n: p.clone() for n, p in model.named_parameters() if "classifier" not in n}
    head0 = model.model.classifier.weight.clone()
    m.after_task(TaskInfo(task_id=0, task_name="t0", label_set=["c0"]), _loader([0, 1, 2, 3]))
    assert not torch.equal(head0, model.model.classifier.weight)
    for n, p in model.named_parameters():
        if "classifier" not in n:
            assert torch.equal(trunk0[n], p), n


def test_weight_align_equalises_row_norms(tmp_path):
    model = _tiny_vit(tmp_path)
    m = _method(model, weight_align=True)
    with torch.no_grad():
        model.model.classifier.weight[2:] *= 5.0
    m._weight_align(2)
    w = model.model.classifier.weight
    assert torch.allclose(w[:2].norm(dim=1).mean(), w[2:].norm(dim=1).mean(), rtol=1e-4)


def test_class_balanced_sampler_covers_classes(tmp_path):
    model = _tiny_vit(tmp_path)
    m = _method(model, replay_balance="class", replay_batch_size=64)
    m.store = [
        {"hidden": torch.zeros(17, 32), "labels": torch.tensor(c)} for c in [0] * 30 + [1, 2, 3]
    ]
    seen = {int(m.store[i]["labels"]) for i in m._sample_indices()}
    assert seen == {0, 1, 2, 3}


def test_compressed_store_roundtrip_and_memory(tmp_path):
    model = _tiny_vit(tmp_path)
    m = _method(model, rank_r=4, quant="int8")
    m.after_task(TaskInfo(task_id=0, task_name="t0", label_set=["c0"]), _loader([0, 1]))
    replay = m._sample_replay()
    assert replay["hidden"].shape[-2:] == (17, 32) and "bbox" not in replay
    raw_bytes = len(m.store) * 17 * 32 * 2
    assert 0 < m.memory_bytes() < raw_bytes
