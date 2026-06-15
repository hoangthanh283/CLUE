"""Unit tests for val-F1 early stopping (doccl.methods.base.EarlyStopper).

These are pure-logic tests (no GPU, no data): they verify best-F1 tracking,
patience-based stopping, best-weight restoration, and the disabled fast-path.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from doccl.methods.base import EarlyStopper


def _toy_model() -> nn.Module:
    return nn.Linear(2, 2)


def test_stops_after_patience_bad_epochs() -> None:
    # Arrange: best F1 occurs at epoch 2, then two non-improving epochs.
    es = EarlyStopper(patience=2, enabled=True)
    model = _toy_model()
    f1_seq = [0.50, 0.60, 0.61, 0.605, 0.604]

    # Act
    stops = [es.step(f1, model, ep) for ep, f1 in enumerate(f1_seq)]

    # Assert: stop fires exactly on the 2nd consecutive bad epoch (index 4).
    assert stops == [False, False, False, False, True]
    assert es.best_f1 == 0.61
    assert es.best_epoch == 2


def test_restores_best_weights() -> None:
    # Arrange: weights mutate every epoch; best is captured at epoch 1.
    es = EarlyStopper(patience=1, enabled=True)
    model = _toy_model()
    for ep, f1 in enumerate([0.5, 0.9, 0.4]):
        with torch.no_grad():
            model.weight.add_(1.0)
        es.step(f1, model, ep)
    final_weights = model.weight.clone()

    # Act
    es.restore_best(model)

    # Assert: model no longer holds the final (degraded) weights.
    assert not torch.allclose(final_weights, model.weight)


def test_disabled_never_stops_and_keeps_no_snapshot() -> None:
    es = EarlyStopper(patience=2, enabled=False)
    model = _toy_model()
    stops = [es.step(0.1, model, ep) for ep in range(5)]
    assert stops == [False] * 5
    assert es.best_state is None


def test_monotonic_improvement_never_stops() -> None:
    es = EarlyStopper(patience=2, enabled=True)
    model = _toy_model()
    stops = [es.step(f1, model, ep) for ep, f1 in enumerate([0.1, 0.2, 0.3, 0.4, 0.5])]
    assert not any(stops)
    assert es.best_f1 == 0.5


def test_min_delta_ignores_sub_threshold_noise_on_0_to_100_scale() -> None:
    """Regression: compute_token_f1 returns F1 on a 0-100 scale, so min_delta must
    be in points (default 0.1). A noisy plateau where epoch-to-epoch wiggles are
    < min_delta must still trigger STOP after `patience` epochs — the earlier
    1e-4 default treated +0.01 noise as 'improvement' and never stopped.
    """
    # Arrange: best F1 at epoch 3, then two sub-0.1-point moves (noise).
    es = EarlyStopper(patience=2, min_delta=0.1, enabled=True)
    model = _toy_model()
    traj = [12.0, 45.3, 71.2, 84.6, 84.61, 84.55]

    # Act
    stops = [es.step(f1, model, ep) for ep, f1 in enumerate(traj)]

    # Assert: STOP on the 2nd consecutive non-improving (noisy) epoch.
    assert stops == [False, False, False, False, False, True]
    assert es.best_epoch == 3
    assert es.best_f1 == 84.6
