"""Model-agnostic forgetting-localisation protocol (Localise → Test → Perturb).

The paper's anatomy result rests on two instruments and two statistical tests that were
previously spread over ``doccl.pilot`` and ``scripts/train.py``. This module is the single
entry point, written against the wrapper contract every backbone already honours
(``param_groups`` / ``param_groups_by_depth`` properties, ``cka_layers``, a ``forward`` that
returns ``.logits``), so the same call diagnoses LayoutLMv3, LiLT, BROS, BERT or a ViT.

Inputs:  a checkpoint pair (before / after training task t), a parameter partition into
         groups, and a probe loader drawn from the OLD task.
Step 1 — Localise: per-group old-task-Fisher-weighted displacement
         D_g = Σ_{φ∈g} F^{(t-1)}_φ (θ^{(t)}_φ − θ^{(t-1)}_φ)², and per-token linear CKA between the
         two checkpoints at one depth-matched probe layer per group.
Step 2 — Test: ``dominance_test`` (does one group dominate D_g across seeds?) and
         ``shared_location_test`` (do two backbones / conditions forget in the same PLACE,
         i.e. share the L1-normalised profile of D_g?).
Step 3 — Perturb: ``freeze_groups`` the located groups, retrain, re-run steps 1–2. The
         conservation statement of the paper is "the profile moves, it does not shrink".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from doccl.eval.cka import cka_per_layer, collect_activations
from doccl.eval.fisher import (
    empirical_fisher_diagonal,
    fisher_per_group,
    fisher_weighted_displacement,
    snapshot_params,
)
from doccl.pilot.analyze import _mannwhitney

Groups = dict[str, list[torch.nn.Parameter]]


def _resolve_groups(model: Any, groups: str | Callable[[Any], Groups]) -> Groups:
    if callable(groups):
        return groups(model)
    attr = {"depth": "param_groups_by_depth", "component": "param_groups"}[groups]
    g = getattr(model, attr)
    return g() if callable(g) else g


def _l1_profile(d: dict[str, float], keys: list[str]) -> np.ndarray:
    v = np.array([max(float(d.get(k, 0.0)), 0.0) for k in keys])
    s = v.sum()
    return v / s if s > 0 else v


@dataclass
class BoundaryReport:
    boundary: str
    displacement: dict[str, float]  # D_g, per-parameter mean (comparable across group sizes)
    displacement_sum: dict[str, float]  # raw quadratic form
    cka: dict[str, float]  # per probe layer, 1.0 = no drift
    fisher_old: dict[str, float]  # old-task Fisher importance per group (NOT forgetting)

    def dominant(self) -> str:
        return max(self.displacement, key=self.displacement.get)

    def profile(self) -> dict[str, float]:
        keys = sorted(self.displacement)
        return dict(zip(keys, _l1_profile(self.displacement, keys).tolist(), strict=True))

    def to_dict(self) -> dict:
        return asdict(self)


class ForgettingLocaliser:
    """Arm before training task t on the old task's probe set; measure after it."""

    def __init__(
        self,
        model: Any,
        device: str | torch.device = "cuda",
        groups: str | Callable[[Any], Groups] = "depth",
        fisher_n: int = 200,
        cka_n: int = 2000,
        cka_layers: list[str] | None = None,
        modality_mask: Any = None,
    ):
        self.model = model
        self.device = device
        self.groups = groups
        self.fisher_n = fisher_n
        self.cka_n = cka_n
        self.cka_layers = list(cka_layers or model.cka_layers)
        self.modality_mask = modality_mask
        self._armed: dict | None = None

    def _activations(self, loader) -> dict[str, torch.Tensor]:
        return collect_activations(
            self.model,
            loader,
            self.cka_layers,
            max_samples=self.cka_n,
            device=self.device,
            token_level=True,
            modality_mask=self.modality_mask,
        )

    def arm(self, old_loader) -> None:
        """Record θ^{t-1}, the old task's Fisher and its activations (call before training t)."""
        fisher = empirical_fisher_diagonal(
            self.model,
            old_loader,
            n_samples=self.fisher_n,
            device=self.device,
            modality_mask=self.modality_mask,
        )
        self._armed = {
            "fisher": {k: v.detach().cpu() for k, v in fisher.items()},
            "params": snapshot_params(self.model),
            "acts": self._activations(old_loader),
        }

    def measure(self, old_loader, boundary: str) -> BoundaryReport:
        """After training task t: localise the drift relative to the armed checkpoint."""
        if self._armed is None:
            raise RuntimeError("call arm(old_loader) before training the task you want to localise")
        a = self._armed
        groups = _resolve_groups(self.model, self.groups)
        params_after = snapshot_params(self.model)
        acts_after = self._activations(old_loader)
        report = BoundaryReport(
            boundary=boundary,
            displacement=fisher_weighted_displacement(
                a["fisher"], a["params"], params_after, groups, self.model, "mean"
            ),
            displacement_sum=fisher_weighted_displacement(
                a["fisher"], a["params"], params_after, groups, self.model, "sum"
            ),
            cka=cka_per_layer(a["acts"], acts_after),
            fisher_old=fisher_per_group(a["fisher"], groups, self.model),
        )
        self._armed = None
        return report


# --------------------------------------------------------------------------- step 2: test
def dominance_test(reports: list[BoundaryReport], alpha: float = 0.05) -> dict:
    """H0: D_g uniform across groups. One report per seed (same boundary / backbone).
    Dominant = largest mean D_g; Mann-Whitney against each rival, Bonferroni-corrected."""
    groups = sorted(reports[0].displacement)
    samples = {g: [float(r.displacement.get(g, 0.0)) for r in reports] for g in groups}
    means = {g: float(np.mean(v)) for g, v in samples.items()}
    dominant = max(means, key=means.get)
    bonf = alpha / max(len(groups) - 1, 1)
    rivals = []
    for g in groups:
        if g == dominant:
            continue
        p = _mannwhitney(samples[dominant], samples[g])
        rivals.append(
            {"group": g, "mean": means[g], "p": p, "significant": bool(p == p and p < bonf)}
        )
    return {
        "dominant": dominant,
        "dominant_mean": means[dominant],
        "share": float(_l1_profile(means, groups)[groups.index(dominant)]),
        "bonferroni": bonf,
        "rivals": rivals,
        "reject_H0": any(r["significant"] for r in rivals),
        "n_seeds": len(reports),
    }


def shared_location_test(
    reference: list[BoundaryReport],
    rival: list[BoundaryReport],
    n_perm: int = 2000,
    seed: int = 0,
) -> dict:
    """Do two sets of reports (e.g. two backbones) forget in the same PLACE?
    Cosine distance between mean L1-normalised D_g profiles; permutation p over pooled
    per-seed profiles. Failing to reject = shared location (the paper's Finding 1')."""
    keys = sorted(set(reference[0].displacement) & set(rival[0].displacement))
    ref = [_l1_profile(r.displacement, keys) for r in reference]
    riv = [_l1_profile(r.displacement, keys) for r in rival]

    def dist(a: list[np.ndarray], b: list[np.ndarray]) -> float:
        ma, mb = np.mean(a, axis=0), np.mean(b, axis=0)
        na, nb = np.linalg.norm(ma), np.linalg.norm(mb)
        return float("nan") if na < 1e-12 or nb < 1e-12 else float(1.0 - ma @ mb / (na * nb))

    observed = dist(ref, riv)
    pool = ref + riv
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if dist(pool[: len(ref)], pool[len(ref) :]) >= observed:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"groups": keys, "cosine_distance": observed, "p": p, "shared_location": bool(p >= 0.05)}


# --------------------------------------------------------------------------- step 3: perturb
def freeze_groups(
    model: Any, group_names: list[str], groups: str | Callable[[Any], Groups] = "depth"
) -> int:
    """Protect the located groups (requires_grad=False). Returns #parameters frozen.
    Re-run Localise/Test afterwards: the anatomy predicts the profile relocates."""
    resolved = _resolve_groups(model, groups)
    n = 0
    for name in group_names:
        for p in resolved[name]:
            p.requires_grad = False
            n += p.numel()
    return n
