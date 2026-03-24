"""Tests for LwF (Learning without Forgetting) strategy."""

import copy

import torch
import pytest

from src.cl_strategies.lwf import LwF
from src.config import LwFConfig
from tests.conftest import TinyModel, BATCH, SEQ_LEN, NUM_LABELS


def _make_batch():
    return {
        "input_ids": torch.randint(0, 100, (BATCH, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.long),
        "bbox": torch.zeros(BATCH, SEQ_LEN, 4, dtype=torch.long),
        "labels": torch.randint(0, NUM_LABELS, (BATCH, SEQ_LEN)),
    }


@pytest.fixture
def lwf_config():
    return LwFConfig(name="lwf", lwf_alpha=0.5, lwf_temperature=2.0, unified_label_space=True)


@pytest.fixture
def lwf(lwf_config):
    return LwF(lwf_config)


@pytest.fixture
def model():
    return TinyModel()


# ---------------------------------------------------------------------------
# LwFConfig validation
# ---------------------------------------------------------------------------


def test_lwf_config_requires_unified():
    with pytest.raises(ValueError, match="unified label space"):
        LwFConfig(unified_label_space=False)


# ---------------------------------------------------------------------------
# before_task
# ---------------------------------------------------------------------------


def test_before_task_0_no_teacher(lwf, model):
    lwf.before_task(model, task_id=0)
    assert lwf.teacher is None


def test_before_task_1_creates_teacher(lwf, model):
    lwf.before_task(model, task_id=0)
    lwf.before_task(model, task_id=1)
    assert lwf.teacher is not None


def test_teacher_is_frozen(lwf, model):
    lwf.before_task(model, task_id=0)
    lwf.before_task(model, task_id=1)
    for p in lwf.teacher.parameters():
        assert not p.requires_grad


def test_teacher_is_deepcopy(lwf, model):
    lwf.before_task(model, task_id=0)
    lwf.before_task(model, task_id=1)
    # Modifying model should not affect teacher
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.0)
    teacher_norm = sum(p.norm().item() for p in lwf.teacher.parameters())
    assert teacher_norm != 0.0, "Teacher should be a frozen copy, unaffected by model changes"


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------


def test_compute_loss_first_task_is_base_loss(lwf, model):
    lwf.before_task(model, task_id=0)
    batch = _make_batch()
    outputs = model(**batch)
    loss = lwf.compute_loss(model, batch, outputs)
    assert loss.item() == pytest.approx(outputs["loss"].item())


def test_compute_loss_second_task_differs_from_base(lwf, model):
    lwf.before_task(model, task_id=0)
    lwf.before_task(model, task_id=1)
    batch = _make_batch()
    outputs = model(**batch)
    total_loss = lwf.compute_loss(model, batch, outputs)
    base_loss = outputs["loss"].item()
    # Combined loss != base_loss (distillation term adds something)
    assert total_loss.item() != pytest.approx(base_loss)


def test_compute_loss_alpha_1_approaches_base(model):
    """With alpha=1 the distillation term weight is (1-1)*T²*KL = 0."""
    cfg = LwFConfig(name="lwf", lwf_alpha=1.0, lwf_temperature=2.0, unified_label_space=True)
    strat = LwF(cfg)
    strat.before_task(model, task_id=0)
    strat.before_task(model, task_id=1)
    batch = _make_batch()
    outputs = model(**batch)
    loss = strat.compute_loss(model, batch, outputs).item()
    base = outputs["loss"].item()
    assert loss == pytest.approx(base, abs=1e-5)


def test_compute_loss_without_mask(lwf, model):
    """Batch without attention_mask uses all tokens."""
    lwf.before_task(model, task_id=0)
    lwf.before_task(model, task_id=1)
    batch = _make_batch()
    batch.pop("attention_mask")
    outputs = model(**batch, attention_mask=None)
    # Should not raise
    loss = lwf.compute_loss(model, batch, outputs)
    assert loss is not None


# ---------------------------------------------------------------------------
# LwF from dict config
# ---------------------------------------------------------------------------


def test_lwf_from_dict_config():
    cfg_dict = {
        "cl_strategy": {"name": "lwf", "lwf_alpha": 0.3, "lwf_temperature": 3.0},
        "label_space": {"unified": True},
    }
    strat = LwF(cfg_dict)
    assert strat.alpha == pytest.approx(0.3)
    assert strat.temperature == pytest.approx(3.0)


def test_lwf_from_dict_config_raises_without_unified():
    cfg_dict = {
        "cl_strategy": {"name": "lwf"},
        "label_space": {"unified": False},
    }
    with pytest.raises(ValueError, match="unified label space"):
        LwF(cfg_dict)
