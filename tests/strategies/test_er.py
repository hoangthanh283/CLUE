"""Tests for ExperienceReplay strategy."""

from unittest.mock import MagicMock, patch

import torch
import pytest

from src.cl_strategies.er import ExperienceReplay
from src.config import ERConfig
from tests.conftest import TinyModel, BATCH, SEQ_LEN, NUM_LABELS


@pytest.fixture
def er_config(tmp_path):
    return ERConfig(name="er", memory_size=50, replay_batch_size=2, replay_weight=1.0)


@pytest.fixture
def er(er_config, tmp_path):
    strat = ExperienceReplay(er_config)
    strat.memory.storage_dir = tmp_path / "er_mem"
    strat.memory.storage_dir.mkdir(parents=True, exist_ok=True)
    return strat


@pytest.fixture
def model():
    return TinyModel()


@pytest.fixture
def batch():
    return {
        "input_ids": torch.randint(0, 100, (BATCH, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.long),
        "bbox": torch.zeros(BATCH, SEQ_LEN, 4, dtype=torch.long),
        "labels": torch.randint(0, NUM_LABELS, (BATCH, SEQ_LEN)),
    }


# ---------------------------------------------------------------------------
# compute_loss
# ---------------------------------------------------------------------------


def test_compute_loss_empty_memory_returns_base_loss(er, model, batch):
    """With empty memory there is no replay term."""
    outputs = model(**batch)
    base_loss = outputs["loss"]
    loss = er.compute_loss(model, batch, outputs)
    # Should equal base_loss (memory is empty)
    assert loss.item() == pytest.approx(base_loss.item())


def test_compute_loss_with_memory_adds_replay(er, model, batch, tmp_path):
    """After populating memory, loss should differ from base_loss."""
    # Populate memory
    er.memory.storage_dir = tmp_path / "er_mem2"
    er.memory.storage_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        er.memory.add_batch(batch, task_id=0)

    outputs = model(**batch)
    base_loss = outputs["loss"].item()
    loss = er.compute_loss(model, batch, outputs)
    # loss != base_loss (replay adds a term)
    assert loss.item() != pytest.approx(base_loss)


def test_compute_loss_zero_replay_weight_equals_base(tmp_path):
    """replay_weight=0 ⇒ loss exactly equals base_loss."""
    cfg = ERConfig(name="er", memory_size=50, replay_batch_size=2, replay_weight=0.0)
    er = ExperienceReplay(cfg)
    er.memory.storage_dir = tmp_path / "m"
    er.memory.storage_dir.mkdir(parents=True, exist_ok=True)

    model = TinyModel()
    batch = {
        "input_ids": torch.randint(0, 100, (BATCH, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.long),
        "bbox": torch.zeros(BATCH, SEQ_LEN, 4, dtype=torch.long),
        "labels": torch.randint(0, NUM_LABELS, (BATCH, SEQ_LEN)),
    }

    # Populate memory first
    for _ in range(3):
        er.memory.add_batch(batch, task_id=0)

    outputs = model(**batch)
    base_loss = outputs["loss"].item()
    total_loss = er.compute_loss(model, batch, outputs).item()
    assert total_loss == pytest.approx(base_loss)


# ---------------------------------------------------------------------------
# update_memory
# ---------------------------------------------------------------------------


def test_update_memory_calls_add_batch(er, batch):
    er.memory.add_batch = MagicMock()
    er.update_memory(batch)
    er.memory.add_batch.assert_called_once_with(batch)


def test_update_memory_increases_buffer_size(er, batch, tmp_path):
    er.memory.storage_dir = tmp_path / "er_up"
    er.memory.storage_dir.mkdir(parents=True, exist_ok=True)
    assert len(er.memory) == 0
    er.update_memory(batch)
    assert len(er.memory) == BATCH


# ---------------------------------------------------------------------------
# Config dict path
# ---------------------------------------------------------------------------


def test_er_from_dict_config(tmp_path):
    cfg_dict = {
        "cl_strategy": {
            "name": "er",
            "memory_size": 100,
            "replay_batch_size": 4,
            "replay_weight": 0.5,
        }
    }
    er = ExperienceReplay(cfg_dict)
    assert er.replay_batch_size == 4
    assert er.replay_weight == pytest.approx(0.5)
