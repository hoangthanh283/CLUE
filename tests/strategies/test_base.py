"""Tests for BaseCLStrategy lifecycle hooks."""

import torch
import pytest

from src.cl_strategies.base import BaseCLStrategy
from src.config import StrategyConfig
from tests.conftest import TinyModel, BATCH, SEQ_LEN, NUM_LABELS


@pytest.fixture
def strategy():
    cfg = StrategyConfig(name="sequential")
    return BaseCLStrategy(cfg)


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
# before_task / current_task_id
# ---------------------------------------------------------------------------


def test_before_task_sets_current_task_id(strategy, model):
    strategy.before_task(model, task_id=0)
    assert strategy.current_task_id == 0


def test_before_task_updates_task_id(strategy, model):
    strategy.before_task(model, task_id=0)
    strategy.before_task(model, task_id=2)
    assert strategy.current_task_id == 2


# ---------------------------------------------------------------------------
# after_task / seen_tasks
# ---------------------------------------------------------------------------


def test_after_task_appends_to_seen(strategy, model):
    strategy.after_task(model, task_id=0)
    assert 0 in strategy.seen_tasks


def test_after_task_no_duplicates(strategy, model):
    strategy.after_task(model, task_id=0)
    strategy.after_task(model, task_id=0)
    assert strategy.seen_tasks.count(0) == 1


def test_seen_tasks_accumulates(strategy, model):
    for i in range(3):
        strategy.after_task(model, task_id=i)
    assert strategy.seen_tasks == [0, 1, 2]


# ---------------------------------------------------------------------------
# compute_loss – returns outputs["loss"] by default
# ---------------------------------------------------------------------------


def test_compute_loss_returns_model_loss(strategy, model, batch):
    outputs = model(**batch)
    loss = strategy.compute_loss(model, batch, outputs)
    assert loss is outputs["loss"]


def test_compute_loss_raises_without_loss(strategy, model, batch):
    with pytest.raises(ValueError, match="loss"):
        strategy.compute_loss(model, batch, {})


# ---------------------------------------------------------------------------
# on_before_backward and on_after_backward are no-ops
# ---------------------------------------------------------------------------


def test_on_before_backward_noop(strategy, model):
    loss = torch.tensor(1.0, requires_grad=True)
    strategy.on_before_backward(model, loss)  # should not raise


def test_on_after_backward_noop(strategy, model):
    strategy.on_after_backward(model, is_final_accumulation_step=True)  # should not raise


# ---------------------------------------------------------------------------
# update_memory is a no-op in base
# ---------------------------------------------------------------------------


def test_update_memory_noop(strategy, batch):
    strategy.update_memory(batch)  # should not raise


# ---------------------------------------------------------------------------
# memory_strategy_mixin.py: token_type_ids branch (line 48)
# ---------------------------------------------------------------------------


def test_store_episodic_sample_with_token_type_ids(tmp_path):
    """Batch with token_type_ids exercises the conditional at mixin line 48."""
    from src.cl_strategies.memory import MemoryBuffer
    from src.cl_strategies.memory_strategy_mixin import store_episodic_sample

    buf = MemoryBuffer(capacity=10, storage_dir=str(tmp_path / "mixin_mem"))
    batch_with_tti = {
        "input_ids": torch.randint(0, 100, (BATCH, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.long),
        "bbox": torch.zeros(BATCH, SEQ_LEN, 4, dtype=torch.long),
        "labels": torch.randint(0, NUM_LABELS, (BATCH, SEQ_LEN)),
        "token_type_ids": torch.zeros(BATCH, SEQ_LEN, dtype=torch.long),
    }
    store_episodic_sample(buf, batch_with_tti, task_id=0)
    assert len(buf) == 1
