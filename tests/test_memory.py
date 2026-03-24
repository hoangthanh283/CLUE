"""Tests for src/cl_strategies/memory.py – MemoryBuffer."""

from unittest.mock import patch

import torch
import pytest

from src.cl_strategies.memory import MemoryBuffer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_batch(batch_size=2, seq_len=8) -> dict:
    return {
        "input_ids": torch.randint(0, 100, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "bbox": torch.zeros(batch_size, seq_len, 4, dtype=torch.long),
        "labels": torch.randint(0, 3, (batch_size, seq_len)),
    }


@pytest.fixture
def buf(tmp_path) -> MemoryBuffer:
    return MemoryBuffer(capacity=10, storage_dir=str(tmp_path / "mem"))


# ---------------------------------------------------------------------------
# Basic add / len
# ---------------------------------------------------------------------------


def test_empty_buffer(buf):
    assert len(buf) == 0


def test_add_batch_increases_len(buf):
    buf.add_batch(_make_batch(batch_size=3), task_id=0)
    assert len(buf) == 3


def test_len_capped_at_capacity(tmp_path):
    buf = MemoryBuffer(capacity=5, storage_dir=str(tmp_path / "m"))
    for _ in range(4):
        buf.add_batch(_make_batch(batch_size=2), task_id=0)
    assert len(buf) <= 5


# ---------------------------------------------------------------------------
# Reservoir sampling – capacity constraint
# ---------------------------------------------------------------------------


def test_reservoir_stays_at_capacity(tmp_path):
    cap = 5
    buf = MemoryBuffer(capacity=cap, storage_dir=str(tmp_path / "m"))
    # Add 20 items (much more than capacity)
    for i in range(10):
        buf.add_batch(_make_batch(batch_size=2), task_id=0)
    assert len(buf) == cap


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


def test_sample_returns_none_when_empty(buf):
    result = buf.sample(4, device=torch.device("cpu"))
    assert result is None


def test_sample_returns_dict(buf):
    buf.add_batch(_make_batch(batch_size=4), task_id=0)
    result = buf.sample(2, device=torch.device("cpu"))
    assert result is not None
    assert "input_ids" in result
    assert "labels" in result


def test_sample_batch_size_bounded(buf):
    buf.add_batch(_make_batch(batch_size=3), task_id=0)
    result = buf.sample(100, device=torch.device("cpu"))  # request more than available
    assert result is not None
    assert result["input_ids"].shape[0] <= 3


# ---------------------------------------------------------------------------
# Task-aware sampling
# ---------------------------------------------------------------------------


def test_task_aware_sampling(tmp_path):
    buf = MemoryBuffer(capacity=20, storage_dir=str(tmp_path / "m"))
    buf.add_batch(_make_batch(batch_size=3), task_id=0)
    buf.add_batch(_make_batch(batch_size=3), task_id=1)

    result = buf.sample(2, device=torch.device("cpu"), task_id=0)
    assert result is not None

    result_missing = buf.sample(2, device=torch.device("cpu"), task_id=99)
    assert result_missing is None


# ---------------------------------------------------------------------------
# get_task_counts
# ---------------------------------------------------------------------------


def test_get_task_counts(tmp_path):
    buf = MemoryBuffer(capacity=20, storage_dir=str(tmp_path / "m"))
    buf.add_batch(_make_batch(batch_size=2), task_id=0)
    buf.add_batch(_make_batch(batch_size=3), task_id=1)
    counts = buf.get_task_counts()
    assert counts.get(0, 0) == 2
    assert counts.get(1, 0) == 3


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------


def test_cleanup_empties_buffer(buf):
    buf.add_batch(_make_batch(batch_size=4), task_id=0)
    assert len(buf) > 0
    buf.cleanup()
    assert len(buf) == 0
    # Sampling after cleanup should return None
    assert buf.sample(1, device=torch.device("cpu")) is None


# ---------------------------------------------------------------------------
# inspect_sample
# ---------------------------------------------------------------------------


def test_inspect_sample(buf):
    buf.add_batch(_make_batch(batch_size=2), task_id=1)
    info = buf.inspect_sample(0)
    assert info is not None
    assert info["task_id"] == 1
    assert "input_ids_shape" in info


def test_inspect_sample_out_of_range(buf):
    result = buf.inspect_sample(999)
    assert result is None


# ---------------------------------------------------------------------------
# _load_item exception handler (lines 97-98)
# ---------------------------------------------------------------------------


def test_load_item_exception_returns_none(buf):
    """When torch.load raises, _load_item returns None (lines 97-98)."""
    buf.add_batch(_make_batch(batch_size=1), task_id=0)
    key = buf.keys[0]
    with patch("torch.load", side_effect=Exception("corrupt file")):
        result = buf._load_item(key)
    assert result is None


# ---------------------------------------------------------------------------
# _delete_item exception handler (lines 104-105)
# ---------------------------------------------------------------------------


def test_delete_item_exception_silenced(buf):
    """When Path.unlink raises, _delete_item silently passes (lines 104-105)."""
    buf.add_batch(_make_batch(batch_size=1), task_id=0)
    key = buf.keys[0]
    with patch("pathlib.Path.unlink", side_effect=PermissionError("no permission")):
        buf._delete_item(key)  # Should not raise


# ---------------------------------------------------------------------------
# sample returns None when all disk items are gone (line 213)
# ---------------------------------------------------------------------------


def test_sample_returns_none_when_items_deleted(buf):
    """If all items fail to load, sample returns None (line 213)."""
    buf.add_batch(_make_batch(batch_size=2), task_id=0)
    with patch("torch.load", side_effect=Exception("missing")):
        result = buf.sample(2, device=torch.device("cpu"))
    assert result is None


# ---------------------------------------------------------------------------
# inspect_sample when item fails to load (line 264)
# ---------------------------------------------------------------------------


def test_inspect_sample_load_failure_returns_none(buf):
    """When _load_item returns None, inspect_sample returns None (line 264)."""
    buf.add_batch(_make_batch(batch_size=1), task_id=0)
    with patch("torch.load", side_effect=Exception("missing")):
        result = buf.inspect_sample(0)
    assert result is None
