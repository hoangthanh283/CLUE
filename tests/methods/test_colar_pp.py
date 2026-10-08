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


def test_token_drop_shortens_replay_only_when_augmenting(tmp_path):
    model = _tiny_vit(tmp_path)
    m = _method(model, token_drop=0.5, replay_batch_size=4)
    m.store = [{"hidden": torch.randn(17, 32), "labels": torch.tensor(c)} for c in range(4)]
    assert m._stack_replay(m.store)["hidden"].shape[1] == 17  # CA / logit banking: no drop
    replay = m._sample_replay()
    assert replay["hidden"].shape[1] == 1 + 8  # CLS + half of 16 patches
    stored_cls = {tuple(d["hidden"][0].half().tolist()) for d in m.store}
    for row in replay["hidden"]:
        assert tuple(row[0].tolist()) in stored_cls  # CLS token always kept


def test_latent_distill_banks_logits_and_adds_loss(tmp_path):
    model = _tiny_vit(tmp_path)
    m = _method(model, latent_distill=0.5)
    m.after_task(TaskInfo(task_id=0, task_name="t0", label_set=["c0"]), _loader([0, 1]))
    assert all("logits" in d and d["logits"].shape == (4,) for d in m.store)
    replay = m._sample_replay()
    model.train()
    with_ld = m._replay_forward(replay).loss
    m.latent_distill = 0.0
    without = m._replay_forward(replay).loss
    assert torch.isfinite(with_ld) and with_ld >= without - 1e-6


def test_analytic_head_is_fit_and_used_for_evaluation(tmp_path):
    model = _tiny_vit(tmp_path)
    m = _method(model, analytic_head="rp", rp_dim=64)
    task = TaskInfo(task_id=0, task_name="t0", label_set=["c0"])
    m.after_task(task, _loader([0, 1, 2, 3]))
    assert m._rp._fitted()
    res = m.evaluate({0: _loader([0, 1, 2, 3])})
    assert 0.0 <= res[0].f1 <= 100.0


def test_feature_anchor_is_zero_when_trunk_unchanged(tmp_path):
    model = _tiny_vit(tmp_path)
    m = _method(model, feature_anchor=1.0)
    m.after_task(TaskInfo(task_id=0, task_name="t0", label_set=["c0"]), _loader([0, 1]))
    assert all("feat" in d for d in m.store)
    replay = m._stack_replay(m.store[:4])
    model.eval()  # no dropout: same trunk ⇒ same features as at banking
    with torch.enable_grad():
        anchored = m._replay_forward(replay).loss
    m.feature_anchor = 0.0
    plain = m._replay_forward(replay).loss
    assert torch.allclose(anchored, plain, atol=1e-3)


def test_lora_linear_exact_at_init_and_merge_preserves_function():
    from doccl.methods.colar_pp import LoRALinear

    torch.manual_seed(0)
    base = torch.nn.Linear(8, 4)
    x = torch.randn(3, 8)
    ref = base(x).detach().clone()
    lora = LoRALinear(base, r=2)
    assert torch.allclose(lora(x), ref, atol=1e-6)  # B = 0 ⇒ identical
    with torch.no_grad():
        lora.lora_b.normal_()
    before = lora(x).detach().clone()
    lora.merge_and_reset()
    assert torch.allclose(lora(x), before, atol=1e-5)  # merged weights reproduce the output
    assert torch.count_nonzero(lora.lora_b) == 0


def test_lora_freeze_map_trains_only_adapters_and_head(tmp_path):
    from doccl.methods.colar_pp import LoRALinear

    model = _tiny_vit(tmp_path)
    m = _method(model, trunk_adapt="lora", lora_rank=2)
    m._apply_freeze_map()
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert names and all(("lora_" in n) or ("classifier" in n) for n in names)
    assert any(isinstance(mod, LoRALinear) for mod in model.modules())


def test_drift_comp_builds_stats_and_aligns_head(tmp_path):
    model = _tiny_vit(tmp_path)
    m = _method(model, drift_comp=True, head_align_epochs=1, ca_samples_per_cls=8)
    t0 = TaskInfo(task_id=0, task_name="t0", label_set=["c0", "c1"])
    m.before_task(t0, _loader([0, 1]))
    m.after_task(t0, _loader([0, 1]))
    assert set(m._cls_mean) == {0, 1}
    mean0 = m._cls_mean[0].clone()
    t1 = TaskInfo(task_id=1, task_name="t1", label_set=["c2", "c3"])
    m.before_task(t1, _loader([2, 3]))
    assert m._f_before is not None and m._f_before.shape[0] == len(m.store)
    with torch.no_grad():  # simulate trunk drift
        for p in m._encoder_layers()[-1].parameters():
            p.add_(0.05 * torch.randn_like(p))
    m.after_task(t1, _loader([2, 3]))
    assert set(m._cls_mean) == {0, 1, 2, 3}
    assert not torch.allclose(mean0, m._cls_mean[0])  # old mean shifted by measured drift


def _doc(n_text=6, n_vis=3, d=32, labelled=(1, 3, 4)):
    labels = torch.full((n_text,), -100)
    for i, c in zip(labelled, (0, 1, 2), strict=False):
        labels[i] = c
    return {
        "hidden": torch.randn(n_text + n_vis, d),
        "labels": labels,
        "bbox": torch.arange(n_text * 4).reshape(n_text, 4),
        "attention_mask": torch.ones(n_text, dtype=torch.long),
    }


def test_position_compression_keeps_labelled_text_only(tmp_path):
    model = _tiny_vit(tmp_path)
    m = _method(model, keep_tokens="labeled", visual_tokens="drop")
    d = _doc()
    h = m._compress_positions(d, d.pop("hidden").float())
    assert h.shape[0] == 3 and d["n_text"] == 3 and d["vis_drop"]
    assert d["labels"].tolist() == [0, 1, 2] and d["bbox"].shape == (3, 4)
    m2 = _method(model, keep_tokens="labeled", visual_tokens="keep")
    d2 = _doc()
    h2 = m2._compress_positions(d2, d2.pop("hidden").float())
    assert h2.shape[0] == 3 + 3 and "vis_drop" not in d2


def test_stack_replay_pads_variable_length_documents(tmp_path):
    model = _tiny_vit(tmp_path)
    m = _method(model, keep_tokens="labeled", visual_tokens="keep", rank_r=0)
    docs = []
    for labelled in ((1, 3, 4), (0, 2)):
        d = _doc(labelled=labelled)
        h = m._compress_positions(d, d.pop("hidden").float())
        d["hidden"] = h.to(torch.float16)
        docs.append(d)
    r = m._stack_replay(docs)
    assert r["hidden"].shape == (2, 3 + 3, 32)  # max text (3) + visual (3)
    assert r["labels"].shape == (2, 3) and r["labels"][1].tolist()[-1] == -100
    assert r["attention_mask"][1].tolist() == [1, 1, 0]
    # dropped-visual store: text only, hook pads with the live visual block
    m3 = _method(model, keep_tokens="labeled", visual_tokens="drop", rank_r=0)
    d3 = _doc()
    d3["hidden"] = m3._compress_positions(d3, d3.pop("hidden").float()).to(torch.float16)
    r3 = m3._stack_replay([d3])
    assert r3["hidden"].shape == (1, 3, 32)
    m3._inject = r3["hidden"]
    live = torch.zeros(1, 3 + 5, 32)
    out = m3._pre_hook(None, (live,), {})
    assert out[0][0].shape == (1, 8, 32) and torch.equal(out[0][0][:, 3:], live[:, 3:])
    m3._inject = None
