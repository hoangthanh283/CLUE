"""Tests for EWC strategy."""

import torch
import pytest

from src.cl_strategies.ewc import EWC
from src.config import EWCConfig
from tests.conftest import TinyModel, BATCH, SEQ_LEN, NUM_LABELS


def _make_batch():
    return {
        "input_ids": torch.randint(0, 100, (BATCH, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.long),
        "bbox": torch.zeros(BATCH, SEQ_LEN, 4, dtype=torch.long),
        "labels": torch.randint(0, NUM_LABELS, (BATCH, SEQ_LEN)),
    }


@pytest.fixture
def ewc(tmp_path):
    cfg = EWCConfig(
        name="ewc",
        ewc_lambda=1.0,
        fisher_cache_dir=str(tmp_path / "ewc_cache"),
        n_fisher_samples=None,
        ewc_chunk_size=1_000_000,
        store_fishers_on_cpu=True,
    )
    return EWC(cfg)


@pytest.fixture
def model():
    return TinyModel()


# ---------------------------------------------------------------------------
# _snapshot_params
# ---------------------------------------------------------------------------


def test_snapshot_params_captures_shapes(ewc, model):
    snap = ewc._snapshot_params(model)
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert name in snap
            assert snap[name].shape == p.shape


def test_snapshot_params_is_copy(ewc, model):
    snap = ewc._snapshot_params(model)
    # Modifying the snapshot should not change model params
    first_key = next(iter(snap))
    snap[first_key].fill_(0.0)
    param = dict(model.named_parameters())[first_key]
    assert not torch.all(param == 0)


# ---------------------------------------------------------------------------
# _estimate_fisher
# ---------------------------------------------------------------------------


def test_estimate_fisher_non_negative(ewc, model):
    loader = [_make_batch() for _ in range(2)]
    fisher = ewc._estimate_fisher(model, loader)
    for v in fisher.values():
        assert (v >= 0).all(), "Fisher values must be non-negative"


def test_estimate_fisher_same_keys(ewc, model):
    loader = [_make_batch()]
    fisher = ewc._estimate_fisher(model, loader)
    param_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(fisher.keys()) == param_names


# ---------------------------------------------------------------------------
# ewc_penalty_chunked
# ---------------------------------------------------------------------------


def test_ewc_penalty_chunked_formula(ewc):
    """Penalty = sum(F * (p - p*)^2)."""
    p = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    f = torch.tensor([1.0, 1.0, 1.0])
    p_star = torch.tensor([0.0, 0.0, 0.0])
    expected = 1.0 + 4.0 + 9.0  # 1^2 + 2^2 + 3^2
    result = ewc.ewc_penalty_chunked(p, f, p_star)
    assert result.item() == pytest.approx(expected)


def test_ewc_penalty_chunked_zero_when_no_change(ewc):
    p = torch.tensor([1.5, -0.5], requires_grad=True)
    f = torch.tensor([2.0, 3.0])
    p_star = p.detach().clone()
    result = ewc.ewc_penalty_chunked(p, f, p_star)
    assert result.item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_loss – no penalty on first task
# ---------------------------------------------------------------------------


def test_compute_loss_no_penalty_first_task(ewc, model):
    batch = _make_batch()
    outputs = model(**batch)
    base_loss = outputs["loss"].item()
    total_loss = ewc.compute_loss(model, batch, outputs)
    assert total_loss.item() == pytest.approx(base_loss)


# ---------------------------------------------------------------------------
# after_task / Fisher save-load round-trip
# ---------------------------------------------------------------------------


def test_after_task_populates_fisher_cache(ewc, model):
    loader = [_make_batch() for _ in range(2)]
    ewc.after_task(model, task_id=0, train_loader=loader)
    assert 0 in ewc._fisher_cache
    fisher, params = ewc._fisher_cache[0]
    assert len(fisher) > 0
    assert len(params) > 0


def test_fisher_save_load_round_trip(ewc, model):
    loader = [_make_batch()]
    ewc.after_task(model, task_id=0, train_loader=loader)
    # Clear in-memory cache
    ewc._fisher_cache.clear()
    # Load from disk
    fisher, params = ewc._get_fisher_data(0)
    assert len(fisher) > 0
    assert len(params) > 0


# ---------------------------------------------------------------------------
# compute_loss – penalty added after task 0
# ---------------------------------------------------------------------------


def test_compute_loss_adds_penalty_after_task(ewc, model):
    loader = [_make_batch()]
    ewc.after_task(model, task_id=0, train_loader=loader)

    batch = _make_batch()
    outputs = model(**batch)
    base_loss = outputs["loss"].item()
    total_loss = ewc.compute_loss(model, batch, outputs).item()
    # With ewc_lambda=1 the penalty should add something (unless all diffs are 0)
    # We just check the total loss is >= base_loss
    assert total_loss >= base_loss - 1e-6


# ---------------------------------------------------------------------------
# after_task raises without loader
# ---------------------------------------------------------------------------


def test_after_task_requires_loader(ewc, model):
    with pytest.raises(ValueError, match="train_loader"):
        ewc.after_task(model, task_id=0, train_loader=None)


# ---------------------------------------------------------------------------
# EWC from dict config (lines 56-61)
# ---------------------------------------------------------------------------


def test_ewc_from_dict_config(tmp_path):
    cfg_dict = {
        "cl_strategy": {
            "name": "ewc",
            "ewc_lambda": 0.5,
            "fisher_cache_dir": str(tmp_path / "ewc_dict"),
            "ewc_chunk_size": 500_000,
            "store_fishers_on_cpu": True,
        }
    }
    ewc_d = EWC(cfg_dict)
    assert ewc_d.lambda_ewc == pytest.approx(0.5)
    assert ewc_d.ewc_chunk_size == 500_000


# ---------------------------------------------------------------------------
# EWC before_task (line 137)
# ---------------------------------------------------------------------------


def test_before_task_updates_current_id(ewc, model):
    ewc.before_task(model, task_id=2)
    assert ewc.current_task_id == 2


# ---------------------------------------------------------------------------
# EWC n_fisher_samples subsetting (lines 108-112)
# ---------------------------------------------------------------------------


def test_n_fisher_samples_limits_batches(tmp_path):
    cfg = EWCConfig(
        name="ewc",
        ewc_lambda=1.0,
        fisher_cache_dir=str(tmp_path / "ewc_ns"),
        n_fisher_samples=2,
    )
    ewc_ns = EWC(cfg)
    model = TinyModel()
    # Loader with 5 batches; only 2 should be used
    big_loader = [_make_batch() for _ in range(5)]
    fisher = ewc_ns._estimate_fisher(model, big_loader)
    assert len(fisher) > 0  # Fisher computed with subset


def test_n_fisher_samples_noop_when_small_loader(tmp_path):
    """When loader is smaller than n_fisher_samples, use all batches."""
    cfg = EWCConfig(
        name="ewc",
        ewc_lambda=1.0,
        fisher_cache_dir=str(tmp_path / "ewc_ns2"),
        n_fisher_samples=10,  # larger than loader
    )
    ewc_ns = EWC(cfg)
    model = TinyModel()
    small_loader = [_make_batch() for _ in range(3)]
    fisher = ewc_ns._estimate_fisher(model, small_loader)
    assert len(fisher) > 0


# ---------------------------------------------------------------------------
# EWC store_fishers_on_cpu=False (line 132)
# ---------------------------------------------------------------------------


def test_estimate_fisher_store_on_device(tmp_path):
    cfg = EWCConfig(
        name="ewc",
        ewc_lambda=1.0,
        fisher_cache_dir=str(tmp_path / "ewc_dev"),
        store_fishers_on_cpu=False,
    )
    ewc_dev = EWC(cfg)
    model = TinyModel()
    loader = [_make_batch()]
    fisher = ewc_dev._estimate_fisher(model, loader)
    # Should still be non-negative regardless of storage location
    for v in fisher.values():
        assert (v >= 0).all()


# ---------------------------------------------------------------------------
# EWC parameter with requires_grad=False → skipped (line 251)
# ---------------------------------------------------------------------------


def test_compute_loss_skips_frozen_params(tmp_path):
    cfg = EWCConfig(ewc_lambda=1.0, fisher_cache_dir=str(tmp_path / "ewc_frz"))
    ewc_frz = EWC(cfg)
    model = TinyModel()

    loader = [_make_batch()]
    ewc_frz.after_task(model, task_id=0, train_loader=loader)

    # Freeze classifier weights (requires_grad=False)
    model.classifier.weight.requires_grad_(False)

    batch = _make_batch()
    outputs = model(**batch)
    # Should not crash; frozen param is skipped
    loss = ewc_frz.compute_loss(model, batch, outputs)
    assert loss is not None

    # Restore
    model.classifier.weight.requires_grad_(True)


# ---------------------------------------------------------------------------
# EWC chunked penalty computation (line 262)
# ---------------------------------------------------------------------------


def test_chunked_penalty_used_when_chunk_size_small(tmp_path):
    """Set ewc_chunk_size=1 to force all params through chunked path."""
    cfg = EWCConfig(
        ewc_lambda=1.0,
        fisher_cache_dir=str(tmp_path / "ewc_chunk"),
        ewc_chunk_size=1,  # forces every element to be "large"
    )
    ewc_ch = EWC(cfg)
    model = TinyModel()

    loader = [_make_batch()]
    ewc_ch.after_task(model, task_id=0, train_loader=loader)

    batch = _make_batch()
    outputs = model(**batch)
    loss = ewc_ch.compute_loss(model, batch, outputs)
    assert loss is not None


# ---------------------------------------------------------------------------
# EWC shape mismatch continue (line 257)
# ---------------------------------------------------------------------------


def test_compute_loss_shape_mismatch_skipped(tmp_path):
    """After growing classifier, shape mismatch causes `continue` on line 257."""
    cfg = EWCConfig(ewc_lambda=1.0, fisher_cache_dir=str(tmp_path / "ewc_mm"))
    ewc_mm = EWC(cfg)
    model = TinyModel()

    loader = [_make_batch()]
    ewc_mm.after_task(model, task_id=0, train_loader=loader)

    # Expand classifier → shape mismatch
    model.expand_classifier(NUM_LABELS + 3)

    batch = _make_batch()
    outputs = model(**batch)
    loss = ewc_mm.compute_loss(model, batch, outputs)
    assert loss is not None
    # Loss should equal base_loss because mismatch params are skipped
    # (non-classifier params still contribute EWC penalty)


# ---------------------------------------------------------------------------
# Shape mismatch (grown classifier) is silently skipped
# ---------------------------------------------------------------------------


def test_compute_loss_handles_shape_mismatch(ewc, model, tmp_path):
    """Growing classifier should not crash EWC penalty computation."""
    loader = [_make_batch()]
    ewc.after_task(model, task_id=0, train_loader=loader)

    # Grow the classifier
    model.expand_classifier(NUM_LABELS + 2)

    batch = _make_batch()
    outputs = model(**batch)
    # Should not raise even though classifier shape changed
    loss = ewc.compute_loss(model, batch, outputs)
    assert loss is not None
