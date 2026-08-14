"""Unit tests for CoLaR (per-doc SVD compressed latent replay) and the latent_replay
doc_selection knob — synthetic, fast, GPU-free.

Covers: compression replaces raw hiddens with per-doc factors, reconstruction is exact at
full rank and injectable at any rank, the plastic-only gradient invariant survives, the
compressed footprint beats raw, k-center selection maximizes coverage, and the default
random path stays byte-identical (first-N banking) so prior runs remain reproducible.
"""

from __future__ import annotations

import torch
from torch import nn

from doccl.methods.base import EarlyStopper
from doccl.methods.colar import CoLaR
from doccl.methods.colar_cb import CoLaRCB
from doccl.methods.colar_wsvd import CoLaRWSVD
from doccl.methods.colaslot import CoLaSlot
from doccl.methods.colaslot_fd import CoLaSlotFD
from doccl.methods.colaslot_ra import CoLaSlotRA
from doccl.methods.colaslot_rf import CoLaSlotRF
from doccl.methods.colaslot_ro import CoLaSlotRO
from doccl.methods.latent_replay import LatentReplay
from doccl.types import TaskInfo

D, L, NL, N_LAYERS, K = 16, 6, 4, 4, 2


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(D, D)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return (self.lin(hidden_states),)


class _Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = D
        inner = nn.Module()
        inner.embeddings = nn.Embedding(10, D)
        encoder = nn.Module()
        encoder.layer = nn.ModuleList(_Layer() for _ in range(N_LAYERS))
        inner.encoder = encoder
        hf = nn.Module()
        hf.layoutlmv3 = inner
        hf.classifier = nn.Linear(D, NL)
        hf.config = type("Config", (), {"vocab_size": 10})()
        self.model = hf
        self.num_layers = N_LAYERS
        self.processor = type("P", (), {"tokenizer": type("T", (), {"pad_token_id": 1})()})()

    def freeze_backbone(self):
        for p in self.model.layoutlmv3.parameters():
            p.requires_grad = False

    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, bbox, pixel_values=None, attention_mask=None, labels=None):
        h = self.model.layoutlmv3.embeddings(input_ids)
        for layer in self.model.layoutlmv3.encoder.layer:
            h = layer(h, attention_mask)[0]
        logits = self.model.classifier(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, NL), labels.reshape(-1), ignore_index=-100
            )
        return type("Out", (), {"loss": loss, "logits": logits})()


def _batch(b=2, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    return {
        "input_ids": torch.randint(0, 10, (b, L)),
        "bbox": torch.randint(0, 100, (b, L, 4)),
        "attention_mask": torch.ones(b, L, dtype=torch.long),
        "labels": torch.randint(0, NL, (b, L)),
    }


def _colar(rank=2, docs=2):
    torch.manual_seed(0)
    return CoLaR(
        _Wrapper(),
        {"split_layer_k": K, "docs_per_task": docs, "replay_batch_size": 2, "rank_r": rank},
    )


def test_compression_replaces_hidden_with_factors():
    m = _colar(rank=2, docs=2)
    m._capture_task([_batch(2)])
    d = m.store[0]
    assert "hidden" not in d
    assert d["us"].shape == (L, 2) and d["v"].shape == (2, D)
    assert d["us"].dtype == torch.float16 and d["v"].dtype == torch.float16


def test_full_rank_reconstruction_is_near_exact():
    m = _colar(rank=min(L, D), docs=2)  # full rank -> lossless up to fp16
    m._capture_task([_batch(2)])
    replay = m._sample_replay()
    assert replay["hidden"].shape == (2, L, D)
    d = m.store[0]
    recon = d["us"].float() @ d["v"].float()
    assert recon.shape == (L, D)  # injectable full-width hidden per doc


def test_replay_grads_reach_only_plastic_layers():
    m = _colar(rank=2, docs=2)
    m._apply_freeze_map()
    m._capture_task([_batch(2)])
    out = m._replay_forward(m._sample_replay())
    out.loss.backward()
    layers = m._encoder_layers()
    assert all(p.grad is None for la in layers[:K] for p in la.parameters())
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for la in layers[K:] for p in la.parameters()
    )


def test_compressed_footprint_below_raw():
    m = _colar(rank=2, docs=2)
    m._capture_task([_batch(2)])
    raw_bytes_equiv = 2 * (L * D) * 2  # what the raw store would hold, fp16
    factor_bytes = sum((d["us"].numel() + d["v"].numel()) * 2 for d in m.store)
    assert factor_bytes < raw_bytes_equiv


def _colaslot_config(**overrides):
    config = {
        "split_layer_k": K,
        "docs_per_task": 2,
        "replay_batch_size": 2,
        "rank_r": 2,
        "freeze_lower": False,
        "infer_gate": False,
        "slot_depth": "head_late",
        "slot_sharing": "off",
        "n_tasks": 3,
        "n_slots_head": 6,
        "n_slots_late": 6,
        "repr_rank": 2,
    }
    config.update(overrides)
    return config


def _colaslot(**overrides):
    return CoLaSlot(_Wrapper(), _colaslot_config(**overrides))


def _colaslot_rf(**overrides):
    config = _colaslot_config(
        infer_gate=True,
        store_input_ids=True,
        routing_mode="hard_top1",
        route_margin=0.05,
        slot_depth="head_only",
        refit_epochs=2,
        refit_lr=1e-2,
        refit_null_weight=1.0,
        **overrides,
    )
    return CoLaSlotRF(_Wrapper(), config)


def _colaslot_ra(**overrides):
    config = _colaslot_config(
        infer_gate=True,
        store_input_ids=True,
        routing_mode="hard_top1",
        route_margin=0.05,
        slot_depth="head_only",
        refit_epochs=2,
        refit_lr=1e-2,
        refit_null_weight=0.0,
        **overrides,
    )
    return CoLaSlotRA(_Wrapper(), config)


def _colaslot_ro(**overrides):
    config = _colaslot_config(
        infer_gate=True,
        store_input_ids=True,
        routing_mode="hard_top1",
        route_margin=0.05,
        slot_depth="head_only",
        online_lr=1e-2,
        online_weight_decay=0.0,
        **overrides,
    )
    return CoLaSlotRO(_Wrapper(), config)


def _colaslot_fd(**overrides):
    config = _colaslot_config(
        infer_gate=True,
        store_input_ids=True,
        routing_mode="hard_top1",
        route_margin=0.05,
        slot_depth="head_only",
        online_lr=1e-2,
        online_weight_decay=0.0,
        functional_null_weight=1.0,
        **overrides,
    )
    return CoLaSlotFD(_Wrapper(), config)


def test_colaslot_uses_colar_freeze_map_and_unique_optimizer_params():
    m = _colaslot(split_layer_k=1)
    assert m.slot_sharing == "off"
    m._apply_freeze_map()
    layers = m._encoder_layers()
    assert all(not p.requires_grad for p in layers[0].parameters())
    assert all(p.requires_grad for layer in layers[1:] for p in layer.parameters())
    params = m.trainable_parameters()
    assert len(params) == len({id(p) for p in params})


def test_colaslot_slots_participate_in_replay_and_early_stop_restore():
    m = _colaslot()
    m._apply_freeze_map()
    m._capture_task([_batch(2)])
    m._replay_forward(m._sample_replay()).loss.backward()
    assert m.head_slots.proj.grad is not None and m.head_slots.proj.grad.abs().sum() > 0
    assert any(
        rs.up.grad is not None and rs.up.grad.abs().sum() > 0 for rs in m.late_slots.values()
    )

    stopper = EarlyStopper()
    stopper.step(1.0, m.model, epoch=0)
    saved = m.head_slots.proj.detach().clone()
    with torch.no_grad():
        m.head_slots.proj.add_(1.0)
    stopper.restore_best(m.model)
    assert torch.equal(m.head_slots.proj, saved)


def test_colaslot_r_claims_before_mask_and_replays_real_token_ids():
    m = _colaslot(
        infer_gate=True,
        store_input_ids=True,
        routing_mode="hard_top1",
        route_margin=0.05,
        slot_depth="head_only",
    )
    batch = _batch(2, seed=7)
    task = TaskInfo(task_id=0, task_name="t0", label_set=["O", "KEY", "VALUE"])
    m.before_task(task, [batch])

    assert m.head_slots.slot_owner == [0, 0, -1, -1, -1, -1]
    assert torch.equal(m.head_slots.grad_mask.cpu(), torch.tensor([1, 1, 0, 0, 0, 0]))

    m._apply_freeze_map()
    m._capture_task([batch])
    replay = m._sample_replay()
    assert replay is not None and "input_ids" in replay
    m._replay_forward(replay)
    gate = m.head_slots._infer_gate
    assert gate is not None
    assert torch.equal(gate[:, :2], torch.ones_like(gate[:, :2]))
    assert torch.count_nonzero(gate[:, 2:]) == 0


def test_colaslot_rf_normal_path_is_slot_free_and_uses_only_colar_parameters():
    m = _colaslot_rf()
    batch = _batch(2, seed=5)
    task = TaskInfo(task_id=0, task_name="t0", label_set=["O", "KEY", "VALUE"])
    m.before_task(task, [batch])
    with torch.no_grad():
        m.head_slots.proj.normal_()

    slot_ids = {id(p) for p in m._slot_parameters()}
    assert slot_ids.isdisjoint(id(p) for p in m.trainable_parameters())

    m._slot_reads_enabled = False
    with torch.no_grad():
        slot_free = m.model(**batch).logits
        m._detach_slot_hooks()
        base = m.model(**batch).logits
        m._register_slot_hooks()
    assert torch.equal(slot_free, base)


def test_colaslot_rf_reads_prior_owner_only():
    m = _colaslot_rf()
    task0 = TaskInfo(task_id=0, task_name="t0", label_set=["O", "KEY", "VALUE"])
    task1 = TaskInfo(task_id=1, task_name="t1", label_set=["O", "KEY", "VALUE"])
    batch0 = _batch(1, seed=1)
    batch1 = _batch(1, seed=2)
    batch0["input_ids"].fill_(2)
    batch1["input_ids"].fill_(3)
    m.before_task(task0, [batch0])
    m.before_task(task1, [batch1])

    ids = torch.stack([batch0["input_ids"][0], batch1["input_ids"][0]])
    attention = torch.ones_like(ids)
    m._install_infer_gate(None, None, {"input_ids": ids, "attention_mask": attention})
    gate = m.head_slots._infer_gate
    assert gate is not None
    assert torch.equal(gate[0, :2], torch.ones(2))
    assert torch.count_nonzero(gate[0, 2:]) == 0
    assert torch.count_nonzero(gate[1]) == 0


def test_colaslot_rf_refits_prior_owner_without_changing_colar():
    m = _colaslot_rf()
    task0 = TaskInfo(task_id=0, task_name="t0", label_set=["O", "KEY", "VALUE"])
    task1 = TaskInfo(task_id=1, task_name="t1", label_set=["O", "KEY", "VALUE"])
    batch0 = _batch(2, seed=3)
    batch1 = _batch(2, seed=4)
    batch0["input_ids"].fill_(2)
    batch1["input_ids"].fill_(3)

    m.before_task(task0, [batch0])
    m.after_task(task0, [batch0])
    m.before_task(task1, [batch1])
    slot_ids = {id(p) for p in m._slot_parameters()}
    base_before = {
        name: parameter.detach().clone()
        for name, parameter in m.model.named_parameters()
        if id(parameter) not in slot_ids
    }
    m.after_task(task1, [batch1])

    assert m._store_task_ids == [0, 0, 1, 1]
    assert torch.count_nonzero(m.head_slots.proj[:2]) > 0
    assert torch.count_nonzero(m.head_slots.proj[2:]) == 0
    assert all(
        torch.equal(base_before[name], parameter)
        for name, parameter in m.model.named_parameters()
        if name in base_before
    )
    assert "1:0" in m.diagnostic_metrics["refit"]
    m.model.id_to_label = {0: "O", 1: "B-A", 2: "I-A", 3: "B-B"}
    m.evaluate({0: [batch0], 1: [batch1]})
    assert set(m.diagnostic_metrics["base_only_by_stage"]["1"]) == {"0", "1"}


def test_colaslot_ra_anchors_acquisition_logits_after_base_drift():
    m = _colaslot_ra()
    task0 = TaskInfo(task_id=0, task_name="t0", label_set=["O", "KEY", "VALUE"])
    task1 = TaskInfo(task_id=1, task_name="t1", label_set=["O", "KEY", "VALUE"])
    batch0 = _batch(2, seed=8)
    batch1 = _batch(2, seed=9)
    batch0["input_ids"].fill_(2)
    batch1["input_ids"].fill_(3)

    m.before_task(task0, [batch0])
    m.after_task(task0, [batch0])
    assert all(doc["teacher_logits"].shape == (L, NL) for doc in m.store)
    assert m.memory_bytes() > CoLaSlotRF.memory_bytes(m)

    with torch.no_grad():
        m.model.model.classifier.bias.add_(0.5)
    m.before_task(task1, [batch1])
    m._forced_slot_owner = 0
    anchor_loss = m._positive_refit_loss(m._stack_replay(m.store[:2]), m.store[:2])
    m._forced_slot_owner = None
    assert anchor_loss > 0.1

    slot_ids = {id(p) for p in m._slot_parameters()}
    base_before = {
        name: parameter.detach().clone()
        for name, parameter in m.model.named_parameters()
        if id(parameter) not in slot_ids
    }
    m.after_task(task1, [batch1])

    assert torch.count_nonzero(m.head_slots.proj[:2]) > 0
    assert all(
        torch.equal(base_before[name], parameter)
        for name, parameter in m.model.named_parameters()
        if name in base_before
    )
    assert m.diagnostic_metrics["refit"]["1:0"]["positive_loss"] > 0


def test_colaslot_ro_online_step_changes_only_prior_owner_slots():
    m = _colaslot_ro()
    task0 = TaskInfo(task_id=0, task_name="t0", label_set=["O", "KEY", "VALUE"])
    task1 = TaskInfo(task_id=1, task_name="t1", label_set=["O", "KEY", "VALUE"])
    batch0 = _batch(2, seed=10)
    batch1 = _batch(2, seed=11)
    batch0["input_ids"].fill_(2)
    batch1["input_ids"].fill_(3)

    m.before_task(task0, [batch0])
    m.after_task(task0, [batch0])
    m.before_task(task1, [batch1])
    m._online_docs = list(m.store)
    m._online_optimizer = torch.optim.AdamW(m._slot_parameters(), lr=1e-2, weight_decay=0.1)

    slot_params = m._slot_parameters()
    inactive_before = [parameter.detach()[2:].clone() for parameter in slot_params]
    slot_ids = {id(parameter) for parameter in slot_params}
    base_before = {
        name: parameter.detach().clone()
        for name, parameter in m.model.named_parameters()
        if id(parameter) not in slot_ids
    }
    rng_before = torch.random.get_rng_state()
    m._post_optimizer_step()

    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert m.model.training
    assert torch.count_nonzero(m.head_slots.proj[:2]) > 0
    assert all(
        torch.equal(before, parameter[2:])
        for before, parameter in zip(inactive_before, slot_params, strict=True)
    )
    assert all(
        torch.equal(base_before[name], parameter)
        for name, parameter in m.model.named_parameters()
        if name in base_before
    )
    assert m.diagnostic_metrics["online"]["1:0"]["steps"] == 1


def test_colaslot_fd_tracks_function_drift_and_keeps_colar_frozen():
    m = _colaslot_fd()
    m.model.label_to_id = {"O": 0}
    task0 = TaskInfo(task_id=0, task_name="t0", label_set=["O", "KEY", "VALUE"])
    task1 = TaskInfo(task_id=1, task_name="t1", label_set=["O", "KEY", "VALUE"])
    batch0 = _batch(2, seed=12)
    batch1 = _batch(2, seed=13)
    batch0["input_ids"].fill_(2)
    batch1["input_ids"].fill_(3)

    m.before_task(task0, [batch0])
    m.after_task(task0, [batch0])
    m.before_task(task1, [batch1])
    m._online_optimizer = torch.optim.AdamW(m._slot_parameters(), lr=1e-2)
    m._slot_reads_enabled = False
    m.model.train()
    m.model(**batch1)
    rng_before = torch.random.get_rng_state()
    assert m._sample_replay() is not None
    assert torch.equal(rng_before, torch.random.get_rng_state())

    with torch.no_grad():
        m.model.model.classifier.bias[1].add_(0.5)
    slot_ids = {id(parameter) for parameter in m._slot_parameters()}
    base_before = {
        name: parameter.detach().clone()
        for name, parameter in m.model.named_parameters()
        if id(parameter) not in slot_ids
    }
    m._post_optimizer_step()

    entry = m.diagnostic_metrics["functional"]["1:0"]
    assert entry["mean_drift_loss"] > 0
    assert entry["entity_tokens"] > 0
    assert torch.count_nonzero(m.head_slots.proj[:2]) > 0
    assert all(
        torch.equal(base_before[name], parameter)
        for name, parameter in m.model.named_parameters()
        if name in base_before
    )


def test_colaslot_rejects_known_broken_composition_modes():
    for bad in ({"infer_gate": True}, {"freeze_lower": True}):
        try:
            _colaslot(**bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid CoLaSlot config to fail: {bad}")


def test_class_balanced_replay_changes_loss_without_extra_bytes():
    torch.manual_seed(0)
    m = CoLaRCB(
        _Wrapper(),
        {
            "split_layer_k": K,
            "docs_per_task": 2,
            "replay_batch_size": 2,
            "rank_r": 2,
            "replay_balance_power": 0.5,
        },
    )
    batch = _batch(2)
    batch["labels"].fill_(0)
    batch["labels"][0, 0] = 1
    m._capture_task([batch])
    replay = m._sample_replay()
    plain = CoLaR._replay_forward(m, replay).loss
    balanced = m._replay_forward(replay).loss
    assert balanced != plain
    assert m.memory_bytes() == CoLaR.memory_bytes(m)


def test_task_balanced_replay_equalizes_old_task_mass_and_sampling():
    plain = _colar()
    balanced = CoLaRCB(
        _Wrapper(),
        {
            "split_layer_k": K,
            "docs_per_task": 3,
            "replay_batch_size": 4,
            "rank_r": 2,
            "replay_task_balance": True,
        },
    )
    balanced.store = [{} for _ in range(6)]
    balanced._store_task_ids = [0, 0, 0, 1, 1, 1]
    picked = balanced._sample_indices()
    assert plain._replay_loss_scale(2) == 1.0
    assert balanced._replay_loss_scale(0) == 1.0
    assert balanced._replay_loss_scale(1) == 1.0
    assert balanced._replay_loss_scale(2) == 2.0
    assert [balanced._store_task_ids[i] for i in picked].count(0) == 2
    assert [balanced._store_task_ids[i] for i in picked].count(1) == 2


def test_kcenter_selects_for_coverage_and_random_stays_first_n():
    # random (default): banks the FIRST docs_per_task docs — prior-run reproducibility
    torch.manual_seed(0)
    lr = LatentReplay(_Wrapper(), {"split_layer_k": K, "docs_per_task": 2, "replay_batch_size": 2})
    batches = [_batch(2, seed=1), _batch(2, seed=2)]
    lr._capture_task(batches)
    assert len(lr.store) == 2
    # kcenter: pools candidates then picks a covering subset (deterministic seed=centroid-nearest)
    torch.manual_seed(0)
    kc = LatentReplay(
        _Wrapper(),
        {
            "split_layer_k": K,
            "docs_per_task": 2,
            "replay_batch_size": 2,
            "doc_selection": "kcenter",
            "selection_pool": 4,
        },
    )
    kc._capture_task([_batch(2, seed=1), _batch(2, seed=2)])
    assert len(kc.store) == 2


def test_second_task_banking_over_compressed_store():
    """Regression: banking task 1 while task 0's docs are already compressed (no "hidden"
    key) must not crash — this KeyError killed two full E2 runs before being caught."""
    m = _colar(rank=2, docs=2)
    m._capture_task([_batch(2)])  # task 0 → compressed in place
    m._capture_task([_batch(2)])  # task 1 banking iterates the mixed store for logging
    assert len(m.store) == 4
    assert all("us" in d and "v" in d and "hidden" not in d for d in m.store)


def test_kcenter_picks_the_outlier_first_n_misses():
    """Controlled pool: three near-duplicate docs + one far outlier. First-N (random path)
    banks two near-duplicates; k-center MUST cover the outlier — that is its purpose."""

    def _doc(val):
        return {
            "hidden": torch.full((L, D), float(val), dtype=torch.float16),
            "attention_mask": torch.ones(L, dtype=torch.long),
            "bbox": torch.zeros(L, 4, dtype=torch.long),
            "labels": torch.zeros(L, dtype=torch.long),
        }

    pool = [_doc(0.0), _doc(0.01), _doc(0.02), _doc(10.0)]  # outlier is index 3
    picked = LatentReplay._kcenter_select(pool, 2)
    vals = {float(d["hidden"][0, 0]) for d in picked}
    assert 10.0 in vals  # the outlier is covered
    assert len(picked) == 2


def _colar_wsvd(rank=2, docs=2, entity_weight=4.0):
    torch.manual_seed(0)
    return CoLaRWSVD(
        _Wrapper(),
        {
            "split_layer_k": K,
            "docs_per_task": docs,
            "replay_batch_size": 2,
            "rank_r": rank,
            "svd_entity_weight": entity_weight,
        },
    )


def test_wsvd_spends_rank_on_entity_rows():
    hidden = torch.tensor([[0.0, 4.0], [3.0, 0.0], [3.0, 0.0], [3.0, 0.0]])
    labels = torch.tensor([1, 0, 0, 0])  # first row is entity, others are O
    us_plain, v_plain = CoLaRWSVD._weighted_factors(hidden, labels, rank=1, entity_weight=1.0)
    us_weighted, v_weighted = CoLaRWSVD._weighted_factors(
        hidden, labels, rank=1, entity_weight=50.0
    )
    plain_err = ((us_plain.float() @ v_plain.float()) - hidden)[0].norm()
    weighted_err = ((us_weighted.float() @ v_weighted.float()) - hidden)[0].norm()
    assert weighted_err < plain_err

    hidden_with_visual = torch.cat([hidden, torch.ones(2, 2)], dim=0)
    us, v = CoLaRWSVD._weighted_factors(hidden_with_visual, labels, rank=1, entity_weight=4.0)
    assert (us.float() @ v.float()).shape == hidden_with_visual.shape


def test_wsvd_uses_colar_factor_footprint():
    m = _colar_wsvd(rank=2, docs=2, entity_weight=4.0)
    m._capture_task([_batch(2)])
    d = m.store[0]
    assert set(d) == {"us", "v", "bbox", "attention_mask", "labels"}
    assert d["us"].shape == (L, 2) and d["v"].shape == (2, D)
    assert m.memory_bytes() == sum(
        (doc["us"].numel() + doc["v"].numel()) * 2
        + (doc["bbox"].numel() + doc["attention_mask"].numel() + doc["labels"].numel()) * 8
        for doc in m.store
    )
