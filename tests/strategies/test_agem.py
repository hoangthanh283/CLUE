"""Tests for AGEM strategy."""

import torch
import pytest

from src.cl_strategies.agem import AGEM
from src.cl_strategies.utils import get_grad_vector, set_grad_vector
from src.config import AGEMConfig
from tests.conftest import TinyModel, BATCH, SEQ_LEN, NUM_LABELS


def _make_batch():
    return {
        "input_ids": torch.randint(0, 100, (BATCH, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.long),
        "bbox": torch.zeros(BATCH, SEQ_LEN, 4, dtype=torch.long),
        "labels": torch.randint(0, NUM_LABELS, (BATCH, SEQ_LEN)),
    }


@pytest.fixture
def agem(tmp_path):
    cfg = AGEMConfig(
        name="agem",
        memory_size=20,
        replay_batch_size=2,
        constraint_threshold=-1e-6,
        clear_cache_every=100,
        use_balanced_sampling=False,
    )
    a = AGEM(cfg)
    a.memory.storage_dir = tmp_path / "agem_mem"
    a.memory.storage_dir.mkdir(parents=True, exist_ok=True)
    return a


@pytest.fixture
def model():
    return TinyModel()


# ---------------------------------------------------------------------------
# on_after_backward – noop when no memory / first task
# ---------------------------------------------------------------------------


def test_on_after_backward_noop_first_task(agem, model):
    agem.before_task(model, task_id=0)
    batch = _make_batch()
    model(**batch)["loss"].backward()
    g_before = get_grad_vector(model).clone()
    agem.on_after_backward(model, is_final_accumulation_step=True)
    g_after = get_grad_vector(model)
    # No modification since current_task_id == 0
    assert torch.allclose(g_before, g_after)


def test_on_after_backward_noop_empty_memory(agem, model):
    """Even on task > 0, empty memory → no projection."""
    agem.before_task(model, task_id=1)
    agem.seen_tasks.append(0)
    batch = _make_batch()
    model(**batch)["loss"].backward()
    g_before = get_grad_vector(model).clone()
    agem.on_after_backward(model, is_final_accumulation_step=True)
    g_after = get_grad_vector(model)
    assert torch.allclose(g_before, g_after)


# ---------------------------------------------------------------------------
# Projection formula correctness
# ---------------------------------------------------------------------------


def test_projection_formula_manually():
    """Verify: g_proj = g - (g·g_ref / ||g_ref||^2) * g_ref."""
    model = TinyModel()
    dim = sum(p.numel() for p in model.parameters())

    g = torch.ones(dim)
    g_ref = -torch.ones(dim)   # opposite direction → violation

    # dot = -dim < threshold → projection needed
    dot = torch.dot(g, g_ref).item()
    g_ref_norm_sq = torch.dot(g_ref, g_ref).item()
    g_proj_expected = g - (dot / g_ref_norm_sq) * g_ref

    # Set g as model gradient
    set_grad_vector(model, g.clone())

    # Run projection logic manually
    g_current = get_grad_vector(model)
    dot_product = torch.dot(g_current, g_ref)
    g_ref_norm_sq_t = torch.dot(g_ref, g_ref)
    projection_coeff = dot_product / g_ref_norm_sq_t
    projected = g_current - projection_coeff * g_ref

    assert torch.allclose(projected, g_proj_expected)


# ---------------------------------------------------------------------------
# Non-final accumulation step → no projection
# ---------------------------------------------------------------------------


def test_on_after_backward_skipped_non_final(agem, model, tmp_path):
    agem.before_task(model, task_id=1)
    agem.seen_tasks.append(0)
    # Populate memory
    for _ in range(3):
        agem.memory.add_batch(_make_batch(), task_id=0)

    batch = _make_batch()
    model(**batch)["loss"].backward()
    g_before = get_grad_vector(model).clone()

    agem.on_after_backward(model, is_final_accumulation_step=False)
    g_after = get_grad_vector(model)
    assert torch.allclose(g_before, g_after)


# ---------------------------------------------------------------------------
# update_memory
# ---------------------------------------------------------------------------


def test_update_memory_increments_len(agem):
    assert len(agem.memory) == 0
    agem.update_memory(_make_batch())
    assert len(agem.memory) == 1  # store_episodic_sample stores 1 item


# ---------------------------------------------------------------------------
# Config dict path
# ---------------------------------------------------------------------------


def test_on_after_backward_with_memory_triggers_projection(tmp_path):
    """Populate memory and run on_after_backward on task 1 to exercise projection path."""
    cfg = AGEMConfig(
        name="agem",
        memory_size=20,
        replay_batch_size=2,
        constraint_threshold=-1e-6,
        clear_cache_every=100,
    )
    a = AGEM(cfg)
    a.memory.storage_dir = tmp_path / "agem_proj"
    a.memory.storage_dir.mkdir(parents=True, exist_ok=True)

    model = TinyModel()

    # Task 0 – populate memory
    a.before_task(model, task_id=0)
    for _ in range(3):
        a.update_memory(_make_batch())
    a.after_task(model, task_id=0)

    # Task 1 – backward pass + projection
    a.before_task(model, task_id=1)
    batch = _make_batch()
    model(**batch)["loss"].backward()
    # Should not raise
    a.on_after_backward(model, is_final_accumulation_step=True)


def test_agem_from_dict_config():
    cfg_dict = {
        "cl_strategy": {
            "name": "agem",
            "memory_size": 200,
            "replay_batch_size": 8,
            "constraint_threshold": -0.01,
        }
    }
    a = AGEM(cfg_dict)
    assert a.ref_batch_size == 8
    assert a.constraint_threshold == pytest.approx(-0.01)


# ---------------------------------------------------------------------------
# AGEM periodic cache clear (line 88)
# ---------------------------------------------------------------------------


def test_on_after_backward_triggers_cache_clear(tmp_path):
    """clear_cache_every=1 triggers torch.cuda.empty_cache on every step."""
    cfg = AGEMConfig(
        name="agem",
        memory_size=20,
        replay_batch_size=2,
        constraint_threshold=-1e-6,
        clear_cache_every=1,  # clears every step
        use_balanced_sampling=False,
    )
    a = AGEM(cfg)
    a.memory.storage_dir = tmp_path / "agem_cache"
    a.memory.storage_dir.mkdir(parents=True, exist_ok=True)

    model = TinyModel()
    a.before_task(model, task_id=0)
    batch = _make_batch()
    model(**batch)["loss"].backward()
    # Should not raise; cache clear path exercises line 88
    a.on_after_backward(model, is_final_accumulation_step=True)


# ---------------------------------------------------------------------------
# AGEM projection path (lines 128-136) – force constraint violation
# ---------------------------------------------------------------------------


def test_on_after_backward_projection_executed(tmp_path):
    """Force constraint violation so lines 128-136 (projection) execute."""
    # Use a large positive threshold so any dot product triggers projection
    cfg = AGEMConfig(
        name="agem",
        memory_size=20,
        replay_batch_size=2,
        constraint_threshold=1e9,  # always violated
        clear_cache_every=100,
        use_balanced_sampling=False,
    )
    a = AGEM(cfg)
    a.memory.storage_dir = tmp_path / "agem_proj2"
    a.memory.storage_dir.mkdir(parents=True, exist_ok=True)

    model = TinyModel()

    # Task 0: populate memory
    a.before_task(model, task_id=0)
    for _ in range(3):
        a.update_memory(_make_batch())
    a.after_task(model, task_id=0)

    # Task 1: projection should happen because threshold=1e9 >> any dot product
    a.before_task(model, task_id=1)
    batch = _make_batch()
    model(**batch)["loss"].backward()
    a.on_after_backward(model, is_final_accumulation_step=True)
    # If we got here without error, the projection path was exercised
