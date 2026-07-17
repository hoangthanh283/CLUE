"""Unit tests for the readout-marginal corrections (RCA kill-tests #1/#2) — exact synthetic
label-shift recovery. The corrections are pure numpy; test them before any GPU run."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("rca_readout_fixes", SCRIPTS / "rca_readout_fixes.py")
rrf = importlib.util.module_from_spec(spec)
sys.modules["rca_readout_fixes"] = rrf
spec.loader.exec_module(rrf)

RNG = np.random.default_rng(0)


def _posterior(likelihood: np.ndarray, prior: np.ndarray) -> np.ndarray:
    p = likelihood * prior[None, :]
    return p / p.sum(-1, keepdims=True)


def _synthetic(n=2000, prior_s=(0.7, 0.2, 0.1), prior_t=(0.1, 0.2, 0.7)):
    """Tokens drawn under target prior; model outputs posterior under SOURCE prior
    (the snap). Ground truth = posterior under target prior."""
    prior_s, prior_t = np.asarray(prior_s), np.asarray(prior_t)
    y = RNG.choice(3, size=n, p=prior_t)
    # class-conditional likelihoods: informative but overlapping
    proto = np.array([[6.0, 1, 1], [1, 6.0, 1], [1, 1, 6.0]])
    like = proto[y] * RNG.uniform(0.5, 1.5, size=(n, 3))
    snapped = _posterior(like, prior_s)  # what the snapped head outputs
    true = _posterior(like, prior_t)  # what a correct-prior head would output
    return snapped, true, y, prior_s, prior_t


def test_prior_ratio_is_exact_bayes_inverse():
    snapped, true, _, ps, pt = _synthetic()
    corrected = rrf.prior_ratio(snapped, q_old=pt, q_last=ps, alpha=0.0)
    assert np.allclose(corrected, true, atol=1e-10)  # exact algebraic identity


def test_prior_ratio_smoothing_bounds_degenerate_weights():
    # CORD-style degenerate denominator: q_last has exact-zero mass on a class
    q_old = np.array([0.35, 0.05, 0.60])
    q_last = np.array([0.002, 0.0, 0.998])
    w_raw = q_old / q_last.clip(min=1e-8)
    w_smooth = rrf.prior_ratio_weights(q_old, q_last, alpha=1e-4)
    assert w_raw.max() > 1e6  # the hazard the smoothing exists for
    assert w_smooth.max() < 1e4  # bounded after Laplace smoothing


def test_marginal_match_recovers_target_marginal():
    snapped, true, y, _, pt = _synthetic()
    corrected = rrf.marginal_match(snapped, q_target=pt)
    assert np.allclose(corrected.mean(0), pt, atol=1e-3)  # marginal pinned to target
    # and accuracy improves vs snapped predictions
    assert (corrected.argmax(-1) == y).mean() > (snapped.argmax(-1) == y).mean()


def test_per_doc_em_improves_without_task_id():
    snapped, true, y, ps, _ = _synthetic()
    docs = np.repeat(np.arange(4), 500)  # 4 "documents" of 500 tokens
    corrected = rrf.per_doc_em(snapped, docs, q_source=ps, iters=10)
    acc_c = (corrected.argmax(-1) == y).mean()
    acc_s = (snapped.argmax(-1) == y).mean()
    assert acc_c > acc_s + 0.05  # EM recovers real accuracy, no task-ID used


def test_per_doc_em_escapes_zero_prior_fixed_point():
    # CORD-style source prior: exact zero on class 1; the doc's own tokens carry strong
    # class-1 evidence. Unsmoothed EM locks class 1 at 0 forever (0/eps weight at iter 1).
    q_source = np.array([0.5, 0.0, 0.5])
    probs = np.tile(np.array([[0.2, 0.6, 0.2]]), (100, 1))
    docs = np.zeros(100, dtype=int)
    out = rrf.per_doc_em(probs, docs, q_source, iters=10)
    assert out[:, 1].min() > 0.0  # column not annihilated
    assert (out.argmax(-1) == 1).all()  # document evidence wins


def test_reweight_renormalizes():
    p = np.array([[0.5, 0.5, 0.0]])
    out = rrf.reweight(p, np.array([2.0, 1.0, 1.0]))
    assert np.allclose(out.sum(-1), 1.0)
    assert out[0, 0] > out[0, 1]
