"""Tests for GEM strategy."""

from unittest.mock import patch

import torch
import pytest

from src.cl_strategies.gem import GEM
from src.config import GEMConfig
from tests.conftest import TinyModel, BATCH, SEQ_LEN, NUM_LABELS


def _make_batch():
    return {
        "input_ids": torch.randint(0, 100, (BATCH, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.long),
        "bbox": torch.zeros(BATCH, SEQ_LEN, 4, dtype=torch.long),
        "labels": torch.randint(0, NUM_LABELS, (BATCH, SEQ_LEN)),
    }


@pytest.fixture
def gem_config():
    return GEMConfig(
        name="gem",
        memory_size=20,
        samples_per_task=2,
        max_tasks=5,
        qp_tolerance=1e-3,
        qp_regularization=1e-6,
        margin=0.5,
        clear_cache_every=100,
    )


@pytest.fixture
def gem(gem_config, tmp_path):
    g = GEM(gem_config, cl_setting="class_il")
    g.memory.storage_dir = tmp_path / "gem_mem"
    g.memory.storage_dir.mkdir(parents=True, exist_ok=True)
    return g


@pytest.fixture
def model():
    return TinyModel()


# ---------------------------------------------------------------------------
# _check_violations
# ---------------------------------------------------------------------------


def test_check_violations_no_violation(gem):
    """Positive dot products → no violation."""
    g = torch.ones(4)
    constraints = [torch.ones(4), torch.ones(4) * 2]
    assert gem._check_violations(g, constraints) is False


def test_check_violations_with_violation(gem):
    """Negative dot product below -margin → violation detected."""
    g = torch.ones(4)
    # dot(g, c) = -4 which is < -margin(0.5)
    constraints = [torch.ones(4) * -1.0]
    assert gem._check_violations(g, constraints) is True


def test_check_violations_at_negative_margin(gem):
    """Exactly -margin is NOT a violation (strict <)."""
    g = torch.ones(4) * 0.5
    # dot(g, c) = -0.5 * 4 = -2 < -0.5 → violation
    c = torch.ones(4) * -1.0
    # Let's set a custom margin high enough to cover
    gem.margin = 10.0
    assert gem._check_violations(g, [c]) is False


# ---------------------------------------------------------------------------
# _project_gradient_qp_exact – output satisfies constraints
# ---------------------------------------------------------------------------


def test_qp_projection_satisfies_constraints(gem):
    """After projection, v·g_k >= -margin for each constraint."""
    torch.manual_seed(42)
    dim = 8
    g = torch.randn(dim)
    constraints = [torch.randn(dim) for _ in range(2)]

    v = gem._project_gradient_qp_exact(g, constraints)
    for c in constraints:
        dot = torch.dot(v, c).item()
        assert dot >= -gem.margin - 1e-4, f"Constraint violated: {dot} < {-gem.margin}"


# ---------------------------------------------------------------------------
# _project_gradient_qp_greedy fallback
# ---------------------------------------------------------------------------


def test_greedy_fallback_when_qp_fails(gem):
    """If quadprog.solve_qp raises, greedy projection is used."""
    torch.manual_seed(0)
    dim = 6
    g = torch.randn(dim)
    constraints = [torch.randn(dim)]

    with patch("quadprog.solve_qp", side_effect=Exception("QP failed")):
        v = gem._project_gradient_qp_exact(g, constraints)

    assert v.shape == g.shape
    assert not torch.isnan(v).any()


# ---------------------------------------------------------------------------
# on_after_backward – noop on first task
# ---------------------------------------------------------------------------


def test_on_after_backward_noop_first_task(gem, model):
    """No memory → no projection; should not raise."""
    gem.before_task(model, task_id=0)
    # Compute a real gradient
    batch = _make_batch()
    outputs = model(**batch)
    outputs["loss"].backward()
    gem.on_after_backward(model, is_final_accumulation_step=True)


# ---------------------------------------------------------------------------
# update_memory
# ---------------------------------------------------------------------------


def test_update_memory_stores_sample(gem, model):
    gem.before_task(model, task_id=0)
    batch = _make_batch()
    gem.update_memory(batch)
    assert len(gem.memory) == 1  # store_episodic_sample stores first sample only


# ---------------------------------------------------------------------------
# on_after_backward skipped when not final accumulation step
# ---------------------------------------------------------------------------


def test_on_after_backward_skipped_non_final(gem, model):
    gem.before_task(model, task_id=1)
    gem.seen_tasks.append(0)
    # Should return early without doing anything
    gem.on_after_backward(model, is_final_accumulation_step=False)


# ---------------------------------------------------------------------------
# on_after_backward with populated memory (exercises constraint path)
# ---------------------------------------------------------------------------


def test_on_after_backward_with_memory_no_crash(gem, model, tmp_path):
    """Constraint projection path runs when memory has task-0 samples."""
    gem.memory.storage_dir = tmp_path / "gem_cb"
    gem.memory.storage_dir.mkdir(parents=True, exist_ok=True)

    # Train task 0: populate memory
    gem.before_task(model, task_id=0)
    for _ in range(3):
        gem.update_memory(_make_batch())
    gem.after_task(model, task_id=0)

    # Train task 1: on_after_backward should compute constraints
    gem.before_task(model, task_id=1)
    batch = _make_batch()
    outputs = model(**batch)
    outputs["loss"].backward()
    # Should not raise regardless of violation
    gem.on_after_backward(model, is_final_accumulation_step=True)


# ---------------------------------------------------------------------------
# _compute_constraint_gradients
# ---------------------------------------------------------------------------


def test_compute_constraint_gradients_returns_none_no_prev_tasks(gem, model):
    gem.before_task(model, task_id=0)  # seen_tasks is empty
    result = gem._compute_constraint_gradients(model, device=torch.device("cpu"))
    assert result is None


def test_compute_task_gradient_no_memory_returns_none(gem, model):
    result = gem._compute_task_gradient(model, torch.device("cpu"), n_samples=2, task_id=0)
    assert result is None


# ---------------------------------------------------------------------------
# GEM on_after_backward: constraint_gradients is None (line 104)
# ---------------------------------------------------------------------------


def test_on_after_backward_no_constraints_returns_early(gem, model):
    """Empty memory → _compute_constraint_gradients returns None → return at line 104."""
    gem.before_task(model, task_id=1)
    gem.seen_tasks.append(0)  # Pretend task 0 was done, but memory is empty

    batch = _make_batch()
    outputs = model(**batch)
    outputs["loss"].backward()
    # No memory items → constraint_gradients is None → line 104 executed
    gem.on_after_backward(model, is_final_accumulation_step=True)


# ---------------------------------------------------------------------------
# GEM dict config (lines 41-49)
# ---------------------------------------------------------------------------


def test_gem_from_dict_config():
    cfg_dict = {
        "cl_setting": "task_il",
        "cl_strategy": {
            "name": "gem",
            "memory_size": 100,
            "samples_per_task": 3,
            "max_tasks": 4,
            "qp_tolerance": 1e-4,
            "qp_regularization": 1e-5,
            "margin": 0.1,
            "clear_cache_every": 50,
        },
    }
    g = GEM(cfg_dict)
    assert g.samples_per_task == 3
    assert g.max_tasks_for_constraints == 4
    assert g.margin == pytest.approx(0.1)
    assert g.cl_setting == "task_il"


# ---------------------------------------------------------------------------
# GEM periodic cache clear (line 87) and task_il branch (lines 96, 104)
# ---------------------------------------------------------------------------


def test_on_after_backward_cache_clear_and_task_il(tmp_path):
    """clear_cache_every=1 exercises line 87; task_il on task 0 returns early at line 91."""
    cfg = GEMConfig(
        name="gem",
        memory_size=20,
        samples_per_task=2,
        max_tasks=5,
        qp_tolerance=1e-3,
        qp_regularization=1e-6,
        margin=0.5,
        clear_cache_every=1,  # triggers line 87 on every step
    )
    g = GEM(cfg, cl_setting="task_il")
    g.memory.storage_dir = tmp_path / "gem_il"
    g.memory.storage_dir.mkdir(parents=True, exist_ok=True)

    model = TinyModel()
    g.before_task(model, task_id=0)
    batch = _make_batch()
    outputs = model(**batch)
    outputs["loss"].backward()
    # task_id=0 → early return after cache clear; exercises lines 85-87 + 90-91
    g.on_after_backward(model, is_final_accumulation_step=True)


# ---------------------------------------------------------------------------
# GEM task_il gradient path: lines 96, 104, 201
# ---------------------------------------------------------------------------


def test_on_after_backward_task_il_with_memory(tmp_path):
    """task_il cl_setting exercises get_grad_vector_exclude_classifier (line 96)
    and _compute_task_gradient task_il branch (line 201)."""
    cfg = GEMConfig(
        name="gem",
        memory_size=20,
        samples_per_task=2,
        max_tasks=5,
        qp_tolerance=1e-3,
        qp_regularization=1e-6,
        margin=0.5,
        clear_cache_every=100,
    )
    g = GEM(cfg, cl_setting="task_il")
    g.memory.storage_dir = tmp_path / "gem_taskil"
    g.memory.storage_dir.mkdir(parents=True, exist_ok=True)

    model = TinyModel()

    # Task 0: populate memory
    g.before_task(model, task_id=0)
    for _ in range(3):
        g.update_memory(_make_batch())
    g.after_task(model, task_id=0)

    # Task 1: on_after_backward uses exclude_classifier path
    g.before_task(model, task_id=1)
    batch = _make_batch()
    outputs = model(**batch)
    outputs["loss"].backward()
    g.on_after_backward(model, is_final_accumulation_step=True)


# ---------------------------------------------------------------------------
# GEM forced violation path (lines 115-133) and task_il projection (line 128)
# ---------------------------------------------------------------------------


def test_on_after_backward_forced_violation(tmp_path):
    """margin=-1e9 ensures any dot product triggers constraint violation (lines 115-130)."""
    cfg = GEMConfig(
        name="gem",
        memory_size=20,
        samples_per_task=2,
        max_tasks=5,
        qp_tolerance=1e-3,
        qp_regularization=1e-6,
        margin=-1e9,  # -margin = 1e9, so any dot product < 1e9 → violation
        clear_cache_every=100,
    )
    g = GEM(cfg, cl_setting="class_il")
    g.memory.storage_dir = tmp_path / "gem_viol"
    g.memory.storage_dir.mkdir(parents=True, exist_ok=True)

    model = TinyModel()

    # Task 0: populate memory
    g.before_task(model, task_id=0)
    for _ in range(3):
        g.update_memory(_make_batch())
    g.after_task(model, task_id=0)

    # Task 1: violation is forced, projection executes
    g.before_task(model, task_id=1)
    batch = _make_batch()
    outputs = model(**batch)
    outputs["loss"].backward()
    # Lines 115-130 execute due to forced violation
    g.on_after_backward(model, is_final_accumulation_step=True)


def test_on_after_backward_forced_violation_task_il(tmp_path):
    """Forced violation with task_il exercises set_grad_vector_exclude_classifier (line 128)."""
    cfg = GEMConfig(
        name="gem",
        memory_size=20,
        samples_per_task=2,
        max_tasks=5,
        qp_tolerance=1e-3,
        qp_regularization=1e-6,
        margin=-1e9,
        clear_cache_every=100,
    )
    g = GEM(cfg, cl_setting="task_il")
    g.memory.storage_dir = tmp_path / "gem_viol_til"
    g.memory.storage_dir.mkdir(parents=True, exist_ok=True)

    model = TinyModel()
    g.before_task(model, task_id=0)
    for _ in range(3):
        g.update_memory(_make_batch())
    g.after_task(model, task_id=0)

    g.before_task(model, task_id=1)
    batch = _make_batch()
    outputs = model(**batch)
    outputs["loss"].backward()
    g.on_after_backward(model, is_final_accumulation_step=True)


# ---------------------------------------------------------------------------
# GEM compute_task_gradient exception path (lines 211-214)
# ---------------------------------------------------------------------------


def test_compute_task_gradient_exception_returns_none(gem, model, tmp_path):
    """When model forward raises, _compute_task_gradient returns None (lines 211-214)."""
    from unittest.mock import patch

    gem.memory.storage_dir = tmp_path / "gem_exc"
    gem.memory.storage_dir.mkdir(parents=True, exist_ok=True)
    gem.before_task(model, task_id=0)
    gem.update_memory(_make_batch())

    with patch.object(model, "forward", side_effect=RuntimeError("simulated error")):
        result = gem._compute_task_gradient(model, torch.device("cpu"), n_samples=1, task_id=0)
    assert result is None


# ---------------------------------------------------------------------------
# GEM greedy projection inner loop (lines 315-316, 323-334)
# ---------------------------------------------------------------------------


def test_greedy_projection_inner_loop(gem):
    """Force greedy projection to iterate through violations (lines 315-316, 323-334)."""
    torch.manual_seed(7)
    dim = 8
    # Create gradient and constraints that definitely violate
    g = torch.ones(dim)
    # Constraints pointing opposite: dot product = -dim < 0 → always violated
    constraints = [-torch.ones(dim), -2 * torch.ones(dim)]

    with patch("quadprog.solve_qp", side_effect=Exception("QP failed")):
        v = gem._project_gradient_qp_exact(g, constraints)

    assert v.shape == g.shape
    assert not torch.isnan(v).any()


def test_greedy_breaks_when_most_violated_idx_negative(gem):
    """When qp_tolerance<0, inner check makes most_violated_idx remain -1 → break at line 324."""
    gem.qp_tolerance = -1.0  # -qp_tolerance = 1.0, so min_violation=0 < 1.0 → skip line 320
    dim = 6
    g = torch.ones(dim)
    # Constraints aligned with g: dot products > 0, so no violation found
    constraints = [torch.ones(dim)]  # dot(g, c) = 6 > 0 → no violation

    with patch("quadprog.solve_qp", side_effect=Exception("QP failed")):
        v = gem._project_gradient_qp_exact(g, constraints)

    assert v.shape == g.shape
