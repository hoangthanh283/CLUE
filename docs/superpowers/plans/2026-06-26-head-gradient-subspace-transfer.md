# HGT — Head-localized Gradient-subspace Transfer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A general, buffer-free, positive-BWT continual-learning method (`hgt`) that applies gradient-subspace transfer to the classifier head, plus a faithful whole-network CUBER baseline (`cuber`), then a `dil` α-ablation that tests whether head-gradient transfer yields positive BWT.

**Architecture:** Both methods subclass `NaiveFineTune` and reuse its `train_task` loop unchanged; they hook the classifier head's gradient (HGT) or every linear layer's gradient (CUBER) to project/steer it against stored per-task feature subspaces accumulated in `after_task`. The numerics live in one tested helper (`grad_subspace.py`). Frozen-vs-trainable backbone and transfer strength α are config axes the experiment resolves.

**Tech Stack:** PyTorch, the existing CLUE Hydra/method/registry machinery, `LayoutLMv3Wrapper.token_features` (already added).

## Global Constraints

- Python ≥3.10; UV-managed; run all commands from inside `CLUE/`.
- Lint/format: `uv run ruff check .` (line-length 100, ignores E501) and `uv run black .` must pass. The torch alias `import torch.nn.functional as F` needs `# noqa: N812`; tensor-shape single-caps need `# noqa: N806`.
- Tests: `uv run pytest`. Markers `slow`/`gpu` (e2e is `slow`). New unit tests are CPU-only and fast.
- `configs/` are immutable ground truth — only ADD new option files, never edit existing ones.
- `doccl/` patches commit immediately with an `AGENT IMPL:` / `AGENT FIX:` prefix.
- The classifier head is `model.model.classifier` — a plain `nn.Linear(d, C)`, weight shape `(C, d)`, `d = model.hidden_size` (768 for base). `expand_classifier(new_labels)` grows it preserving old rows.
- Per-token features (head input): `model.token_features(batch) -> (B, L, d)`; `model.model.classifier(token_features)` reproduces token logits. Labels mask: `batch["labels"] != -100` selects scored tokens.
- Local box: RTX 2060, **VRAM < 5 GB, RAM < 14 GB** — experiments use `training.batch_size=2 training.gradient_checkpointing=true training.num_workers=0 wandb.mode=offline`; ONE dataset builder at a time. `epochs` lives at `method.epochs` (NOT `training.epochs`) — overriding `training.epochs` is a Hydra struct error.
- Metrics: `CLMetricsTracker` computes AA/BWT/FWT; **positive BWT is representable** (BWT = mean over i of R[T-1,i] − R[i,i]); NaN never fabricated.
- No CUBER reference repo exists (GitHub search = 0); implement from arXiv 2211.00789 §3.

---

### Task 1: Gradient-subspace numerics helper

**Files:**
- Create: `doccl/methods/grad_subspace.py`
- Test: `tests/methods/test_grad_subspace.py`

**Interfaces:**
- Produces:
  - `feature_moment(feats: Tensor) -> Tensor` — `feats (N, d)` → uncentered second moment `(d, d)` = `featsᵀ feats`.
  - `top_eigvecs(moment: Tensor, k: int) -> Tensor` — symmetric `(d, d)` → top-`k` eigenvectors `(d, k)` (largest eigenvalue first).
  - `project_orth(grad: Tensor, basis: Tensor) -> Tensor` — `grad (C, d)`, `basis (d, m)` (orthonormal cols) → `grad (I − basis basisᵀ)` (component acting on NO stored subspace).
  - `steer_gradient(grad: Tensor, basis: Tensor, alpha: float) -> Tensor` — returns `project_orth(grad, basis) + alpha * (grad @ basis @ basisᵀ)`. `alpha=0` → pure orthogonal projection (GPM-style protection); `alpha=1` → identity (`grad` unchanged); `0<alpha<1` → partial transfer. `basis` empty (m=0) → returns `grad` unchanged.

- [ ] **Step 1: Write the failing tests**

```python
# tests/methods/test_grad_subspace.py
from __future__ import annotations
import torch
from doccl.methods.grad_subspace import feature_moment, top_eigvecs, project_orth, steer_gradient


def test_feature_moment_is_gram():
    f = torch.randn(7, 5)
    assert torch.allclose(feature_moment(f), f.T @ f, atol=1e-5)


def test_top_eigvecs_orthonormal_and_dominant():
    torch.manual_seed(0)
    # rank-structured SPD: one strong direction.
    v = torch.randn(6, 1)
    m = v @ v.T * 10 + torch.eye(6) * 0.1
    u = top_eigvecs(m, k=2)
    assert u.shape == (6, 2)
    # columns orthonormal
    assert torch.allclose(u.T @ u, torch.eye(2), atol=1e-4)
    # leading eigvec aligns with v (up to sign)
    cos = abs(torch.nn.functional.cosine_similarity(u[:, 0], v[:, 0], dim=0).item())
    assert cos > 0.99


def test_project_orth_removes_in_subspace_component():
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(5, 2)).Q  # (5,2) orthonormal
    g = torch.randn(3, 5)
    g_orth = project_orth(g, basis)
    # g_orth has zero projection onto basis: (g_orth @ basis) ≈ 0
    assert torch.allclose(g_orth @ basis, torch.zeros(3, 2), atol=1e-5)


def test_steer_alpha0_equals_orth_alpha1_equals_identity():
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(5, 2)).Q
    g = torch.randn(3, 5)
    assert torch.allclose(steer_gradient(g, basis, 0.0), project_orth(g, basis), atol=1e-5)
    assert torch.allclose(steer_gradient(g, basis, 1.0), g, atol=1e-5)


def test_steer_empty_basis_is_identity():
    g = torch.randn(3, 5)
    empty = torch.zeros(5, 0)
    assert torch.allclose(steer_gradient(g, empty, 0.0), g, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd CLUE && uv run pytest tests/methods/test_grad_subspace.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doccl.methods.grad_subspace'`

- [ ] **Step 3: Write the implementation**

```python
# doccl/methods/grad_subspace.py
"""Gradient-subspace numerics for head-localized gradient transfer (HGT) and CUBER.

A task's "subspace" is the top-k eigenvectors of its uncentered feature second moment
M = Σ f fᵀ (f = the features feeding a linear layer over scored tokens). A weight W's
gradient G (out×in) acts on a stored input-subspace U (in×k) through G U. Removing that
component (``project_orth``) protects the stored task's outputs (GPM/Adam-NSCL); keeping a
fraction α of it (``steer_gradient``) lets new-task updates flow into the stored subspace
— the transfer knob whose α=0 vs α>0 contrast is the method's headline ablation.
"""
from __future__ import annotations

import torch


def feature_moment(feats: torch.Tensor) -> torch.Tensor:
    """Uncentered second moment featsᵀfeats. feats (N, d) -> (d, d)."""
    return feats.T @ feats


def top_eigvecs(moment: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k eigenvectors (largest eigenvalue first) of a symmetric matrix. (d, d) -> (d, k)."""
    # eigh returns ascending eigenvalues; take the last k, reverse to descending.
    evals, evecs = torch.linalg.eigh(moment.double())
    k = min(k, evecs.shape[1])
    idx = torch.arange(evecs.shape[1] - 1, evecs.shape[1] - 1 - k, -1, device=evecs.device)
    return evecs.index_select(1, idx).to(moment.dtype)


def project_orth(grad: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """grad (C, d) minus its component acting on basis (d, m) orthonormal cols -> (C, d)."""
    if basis.numel() == 0 or basis.shape[1] == 0:
        return grad
    return grad - (grad @ basis) @ basis.T


def steer_gradient(grad: torch.Tensor, basis: torch.Tensor, alpha: float) -> torch.Tensor:
    """project_orth + alpha * in-subspace component. alpha=0 -> protection; 1 -> identity."""
    if basis.numel() == 0 or basis.shape[1] == 0:
        return grad
    in_sub = (grad @ basis) @ basis.T
    return (grad - in_sub) + alpha * in_sub
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd CLUE && uv run pytest tests/methods/test_grad_subspace.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Lint + commit**

Run: `cd CLUE && uv run ruff check doccl/methods/grad_subspace.py tests/methods/test_grad_subspace.py && uv run black doccl/methods/grad_subspace.py tests/methods/test_grad_subspace.py`

```bash
cd CLUE && git add doccl/methods/grad_subspace.py tests/methods/test_grad_subspace.py
git commit -m "AGENT IMPL: grad_subspace helper (feature moment, top eigvecs, project/steer)"
```

---

### Task 2: HGT method + config + registry

**Files:**
- Create: `doccl/methods/hgt.py`
- Create: `configs/method/hgt.yaml`
- Modify: `scripts/train.py` (import; `METHOD_REGISTRY`; `_STD_FORWARD`)
- Test: `tests/methods/test_hgt.py`

**Interfaces:**
- Consumes: `feature_moment`, `top_eigvecs`, `steer_gradient` (Task 1); `model.token_features`, `model.model.classifier`, `model.hidden_size`, `model.freeze_backbone()`; `NaiveFineTune.train_task/evaluate`.
- Produces: `class HGT(NaiveFineTune)` with `name = "hgt"`; consumes config keys `backbone_trainable: bool`, `transfer_alpha: float`, `subspace_k: int`, `subspace_n_batches: int`, plus the `NaiveFineTune` keys (`lr`, `epochs`, `weight_decay`, `max_grad_norm`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/methods/test_hgt.py
"""Unit tests for HGT internals on synthetic tensors (fast, GPU-free). The full lifecycle
on a real LayoutLMv3 is covered in tests/methods/test_all_methods_e2e.py once hgt is
registered there."""
from __future__ import annotations
import torch
from doccl.methods.hgt import HGT


def test_stacked_basis_orthonormal_columns():
    """_stacked_basis concatenates per-task subspaces and re-orthonormalises -> Iᵏ."""
    m = HGT.__new__(HGT)
    m.hidden_dim = 6
    torch.manual_seed(0)
    u0 = torch.linalg.qr(torch.randn(6, 2)).Q
    u1 = torch.linalg.qr(torch.randn(6, 2)).Q
    m._task_subspaces = [u0, u1]
    basis = m._stacked_basis()
    assert basis.shape[0] == 6 and basis.shape[1] <= 4
    assert torch.allclose(basis.T @ basis, torch.eye(basis.shape[1]), atol=1e-4)


def test_stacked_basis_empty_when_no_tasks():
    m = HGT.__new__(HGT)
    m.hidden_dim = 6
    m._task_subspaces = []
    assert m._stacked_basis().shape == (6, 0)


def test_steer_hook_alpha0_protects_alpha1_passes():
    """The registered head-grad hook with alpha=0 zeroes the in-subspace gradient; alpha=1
    leaves it unchanged."""
    m = HGT.__new__(HGT)
    m.hidden_dim = 5
    m.transfer_alpha = 0.0
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(5, 2)).Q
    m._task_subspaces = [basis]
    m._steer_enabled = True
    g = torch.randn(3, 5)
    out0 = m._steer_head_grad(g)
    assert torch.allclose(out0 @ basis, torch.zeros(3, 2), atol=1e-5)  # protected
    m.transfer_alpha = 1.0
    assert torch.allclose(m._steer_head_grad(g), g, atol=1e-5)         # full pass


def test_steer_hook_noop_when_disabled():
    m = HGT.__new__(HGT)
    m.hidden_dim = 5
    m.transfer_alpha = 0.0
    m._task_subspaces = [torch.linalg.qr(torch.randn(5, 2)).Q]
    m._steer_enabled = False
    g = torch.randn(3, 5)
    assert torch.allclose(m._steer_head_grad(g), g, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd CLUE && uv run pytest tests/methods/test_hgt.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doccl.methods.hgt'`

- [ ] **Step 3: Write the HGT method**

```python
# doccl/methods/hgt.py
"""HGT — Head-localized Gradient-subspace Transfer (see the design spec
docs/superpowers/specs/2026-06-26-head-gradient-subspace-transfer-design.md).

After each task, store the top-k eigenvectors of that task's head-input (token feature)
second moment. During later tasks, a gradient hook on the head rewrites its gradient to
``orthogonal-to-old-subspaces + transfer_alpha * in-old-subspace`` — alpha=0 protects old
tasks (GPM-style, BWT≈0 control), alpha>0 lets new-task gradients flow into old subspaces
(the positive-transfer hypothesis). Backbone frozen or trainable per ``backbone_trainable``.
Standard-forward method: inference is the inherited NaiveFineTune.evaluate.
"""
from __future__ import annotations

import logging

import torch

from doccl.methods.grad_subspace import feature_moment, steer_gradient, top_eigvecs
from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo

log = logging.getLogger(__name__)

__all__ = ["HGT"]


class HGT(NaiveFineTune):
    name = "hgt"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.backbone_trainable = bool(config.get("backbone_trainable", False))
        self.transfer_alpha = float(config.get("transfer_alpha", 1.0))
        self.subspace_k = int(config.get("subspace_k", 32))
        self.subspace_n_batches = int(config.get("subspace_n_batches", 16))
        self.hidden_dim = model.hidden_size
        if not self.backbone_trainable:
            self.model.freeze_backbone()
            for p in self.model.model.classifier.parameters():
                p.requires_grad = True
        self._task_subspaces: list[torch.Tensor] = []  # each (d, k) orthonormal cols
        self._steer_enabled = False
        # One persistent hook on the head weight; it consults _steer_enabled / subspaces.
        self.model.model.classifier.weight.register_hook(self._steer_head_grad)

    # ─── subspace bookkeeping ────────────────────────────────────────────────────
    def _stacked_basis(self) -> torch.Tensor:
        """Concatenate stored per-task subspaces, re-orthonormalise -> (d, m)."""
        if not self._task_subspaces:
            return torch.zeros(self.hidden_dim, 0, device=self.model.model.classifier.weight.device)
        stacked = torch.cat([u.to(self.model.model.classifier.weight.device)
                             for u in self._task_subspaces], dim=1)
        q, _ = torch.linalg.qr(stacked)
        return q

    def _steer_head_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Head-weight grad hook: steer against stored old subspaces when enabled."""
        if not self._steer_enabled or not self._task_subspaces:
            return grad
        basis = self._stacked_basis().to(grad.dtype)
        return steer_gradient(grad, basis, self.transfer_alpha)

    # ─── lifecycle ───────────────────────────────────────────────────────────────
    def before_task(self, task: TaskInfo, train_loader) -> None:
        # Steering only applies from task 1 on (no old subspaces before then).
        self._steer_enabled = task.task_id > 0 and len(self._task_subspaces) > 0

    def after_task(self, task: TaskInfo, train_loader) -> None:
        self._accumulate_subspace(train_loader)
        log.info("hgt: task %d done; stored %d head subspaces (k=%d, alpha=%.2f, bb_trainable=%s)",
                 task.task_id, len(self._task_subspaces), self.subspace_k,
                 self.transfer_alpha, self.backbone_trainable)

    @torch.no_grad()
    def _accumulate_subspace(self, loader) -> None:
        """Top-k eigenvectors of this task's scored-token feature second moment."""
        self.model.eval()
        moment = torch.zeros(self.hidden_dim, self.hidden_dim, device=self.device)
        for bi, batch in enumerate(loader):
            if bi >= self.subspace_n_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            feats = self.model.token_features(batch)  # (B, L, d)
            labels = batch["labels"][:, : feats.shape[1]]
            f = feats[labels != -100]  # (n_tok, d)
            if f.numel():
                moment += feature_moment(f)
        self._task_subspaces.append(top_eigvecs(moment.cpu(), self.subspace_k))
```

- [ ] **Step 4: Write the config**

```yaml
# configs/method/hgt.yaml
name: hgt
lr: 5.0e-5
weight_decay: 0.01
epochs: 10
max_grad_norm: 1.0
# Backbone regime — the experiment resolves frozen vs trainable.
backbone_trainable: false
# Transfer strength on the in-old-subspace head-gradient component.
# 0.0 = GPM-style protection (BWT~0 control); >0 = positive-transfer hypothesis; 1.0 = no projection.
transfer_alpha: 1.0
# Per-task head-input subspace: top-k eigenvectors of the token-feature second moment.
subspace_k: 32
subspace_n_batches: 16
```

- [ ] **Step 5: Wire into train.py**

Add the import next to the other method imports (alphabetical, after `from doccl.methods.ewc import EWC` / near `hybrid_routed_prompt`):

```python
from doccl.methods.hgt import HGT
```

Add to `METHOD_REGISTRY` (next to `"hrp"`):

```python
    "hgt": HGT,
```

Add `"hgt"` to the `_STD_FORWARD` set (it uses standard forward):

```python
    _STD_FORWARD = {"naive", "joint", "ewc", "lwf", "er", "der_pp", "er_cflat", "doccl", "lca", "hgt"}
```

- [ ] **Step 6: Run unit tests + config resolve**

Run: `cd CLUE && uv run pytest tests/methods/test_hgt.py -v`
Expected: PASS (4 passed)

Run: `cd CLUE && uv run python -c "from scripts.train import METHOD_REGISTRY; print(METHOD_REGISTRY['hgt'].__name__)"`
Expected: prints `HGT`

Run: `cd CLUE && uv run python scripts/train.py method=hgt scenario=dil seed=42 method.transfer_alpha=0.0 method.epochs=3 wandb.mode=offline --cfg job 2>&1 | grep -E "transfer_alpha:|backbone_trainable:"`
Expected: shows `transfer_alpha: 0.0` and `backbone_trainable: false` (no Hydra error)

- [ ] **Step 7: Lint + commit**

Run: `cd CLUE && uv run ruff check doccl/methods/hgt.py tests/methods/test_hgt.py scripts/train.py && uv run black doccl/methods/hgt.py tests/methods/test_hgt.py`
(If ruff flags N806 on single-cap tensor names, add `# noqa: N806`.)

```bash
cd CLUE && git add doccl/methods/hgt.py configs/method/hgt.yaml scripts/train.py tests/methods/test_hgt.py
git commit -m "AGENT IMPL: HGT — head-localized gradient-subspace transfer method"
```

---

### Task 3: CUBER whole-network baseline + config + registry

**Files:**
- Create: `doccl/methods/cuber.py`
- Create: `configs/method/cuber.yaml`
- Modify: `scripts/train.py` (import; `METHOD_REGISTRY`; `_STD_FORWARD`)
- Test: `tests/methods/test_cuber.py`

**Interfaces:**
- Consumes: `feature_moment`, `top_eigvecs`, `steer_gradient` (Task 1); `NaiveFineTune`; per-linear-layer activation capture via forward hooks.
- Produces: `class CUBER(NaiveFineTune)` with `name = "cuber"`; config keys `transfer_alpha: float`, `subspace_k: int`, `subspace_n_batches: int`. CUBER trains the WHOLE network (no `freeze_backbone`) and steers EVERY tracked `nn.Linear`'s weight gradient against that layer's stored input subspace.

- [ ] **Step 1: Write the failing tests**

```python
# tests/methods/test_cuber.py
"""Unit tests for CUBER internals on synthetic tensors (fast, GPU-free)."""
from __future__ import annotations
import torch
from doccl.methods.cuber import CUBER


def test_layer_basis_steers_per_layer():
    """_steer_for_layer steers a layer's grad against ITS stored subspace only."""
    m = CUBER.__new__(CUBER)
    m.transfer_alpha = 0.0
    m._steer_enabled = True
    torch.manual_seed(0)
    basis = torch.linalg.qr(torch.randn(5, 2)).Q
    m._layer_subspaces = {"layerA": [basis]}
    g = torch.randn(4, 5)
    out = m._steer_for_layer("layerA", g)
    assert torch.allclose(out @ basis, torch.zeros(4, 2), atol=1e-5)  # protected
    # a layer with no stored subspace is unchanged
    assert torch.allclose(m._steer_for_layer("unknown", g), g, atol=1e-6)


def test_steer_disabled_is_identity():
    m = CUBER.__new__(CUBER)
    m.transfer_alpha = 0.0
    m._steer_enabled = False
    m._layer_subspaces = {"layerA": [torch.linalg.qr(torch.randn(5, 2)).Q]}
    g = torch.randn(4, 5)
    assert torch.allclose(m._steer_for_layer("layerA", g), g, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd CLUE && uv run pytest tests/methods/test_cuber.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doccl.methods.cuber'`

- [ ] **Step 3: Write the CUBER method**

```python
# doccl/methods/cuber.py
"""CUBER — whole-network gradient-subspace transfer (Lin et al., NeurIPS 2022, arXiv
2211.00789), the mandatory baseline for HGT. Same steering mechanism as HGT
(orthogonal-to-old-subspace + alpha * in-subspace) but applied to EVERY tracked nn.Linear's
weight gradient against that layer's stored INPUT subspace — i.e. whole-network, not
head-only. The HGT-vs-CUBER comparison isolates the value of localizing to the head
(cheaper, frozen-backbone-compatible, backbone-depth-independent).

Faithful to CUBER's mechanism at the level this study compares: per-layer input-subspace
projection with a positive-transfer component (alpha). We track every nn.Linear inside the
backbone + head, capture each one's input activations on scored tokens to build its second
moment, and hook each weight's gradient.
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

from doccl.methods.grad_subspace import feature_moment, steer_gradient, top_eigvecs
from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo

log = logging.getLogger(__name__)

__all__ = ["CUBER"]


class CUBER(NaiveFineTune):
    name = "cuber"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.transfer_alpha = float(config.get("transfer_alpha", 1.0))
        self.subspace_k = int(config.get("subspace_k", 32))
        self.subspace_n_batches = int(config.get("subspace_n_batches", 16))
        # name -> list of per-task input subspaces (each (in_features, k)).
        self._layer_subspaces: dict[str, list[torch.Tensor]] = {}
        self._steer_enabled = False
        self._linear_names = self._collect_linears()
        self._register_grad_hooks()

    def _collect_linears(self) -> list[str]:
        return [n for n, mod in self.model.model.named_modules() if isinstance(mod, nn.Linear)]

    def _register_grad_hooks(self) -> None:
        named = dict(self.model.model.named_modules())
        for name in self._linear_names:
            w = named[name].weight
            w.register_hook(lambda g, nm=name: self._steer_for_layer(nm, g))

    def _stacked_basis(self, name: str, device, dtype) -> torch.Tensor:
        subs = self._layer_subspaces.get(name)
        if not subs:
            return torch.zeros(0, 0, device=device, dtype=dtype)
        stacked = torch.cat([u.to(device=device, dtype=dtype) for u in subs], dim=1)
        q, _ = torch.linalg.qr(stacked)
        return q

    def _steer_for_layer(self, name: str, grad: torch.Tensor) -> torch.Tensor:
        if not self._steer_enabled or name not in self._layer_subspaces:
            return grad
        basis = self._stacked_basis(name, grad.device, grad.dtype)
        return steer_gradient(grad, basis, self.transfer_alpha)

    def before_task(self, task: TaskInfo, train_loader) -> None:
        self._steer_enabled = task.task_id > 0 and bool(self._layer_subspaces)

    def after_task(self, task: TaskInfo, train_loader) -> None:
        self._accumulate_layer_subspaces(train_loader)
        log.info("cuber: task %d done; %d layers tracked (k=%d, alpha=%.2f)",
                 task.task_id, len(self._layer_subspaces), self.subspace_k, self.transfer_alpha)

    @torch.no_grad()
    def _accumulate_layer_subspaces(self, loader) -> None:
        """Capture each tracked Linear's input activations -> per-layer second moment -> eigvecs."""
        named = dict(self.model.model.named_modules())
        moments: dict[str, torch.Tensor] = {}
        handles = []

        def make_hook(nm):
            def hook(_mod, inp, _out):
                x = inp[0].detach()
                x = x.reshape(-1, x.shape[-1])  # (tokens, in_features)
                m = feature_moment(x)
                moments[nm] = moments.get(nm, torch.zeros_like(m)) + m
            return hook

        for name in self._linear_names:
            handles.append(named[name].register_forward_hook(make_hook(name)))
        self.model.eval()
        for bi, batch in enumerate(loader):
            if bi >= self.subspace_n_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            self.model(**{k: v for k, v in batch.items() if k != "labels"})
        for h in handles:
            h.remove()
        for name, m in moments.items():
            self._layer_subspaces.setdefault(name, []).append(top_eigvecs(m.cpu(), self.subspace_k))
```

- [ ] **Step 4: Write the config**

```yaml
# configs/method/cuber.yaml
name: cuber
lr: 5.0e-5
weight_decay: 0.01
epochs: 10
max_grad_norm: 1.0
# Whole-network gradient-subspace transfer (CUBER baseline). transfer_alpha as in HGT:
# 0 = protection (whole-network GPM); >0 = positive transfer; 1 = no projection.
transfer_alpha: 1.0
subspace_k: 32
subspace_n_batches: 16
```

- [ ] **Step 5: Wire into train.py**

Add import (near the others):

```python
from doccl.methods.cuber import CUBER
```

Add to `METHOD_REGISTRY`:

```python
    "cuber": CUBER,
```

Add `"cuber"` to `_STD_FORWARD`:

```python
    _STD_FORWARD = {"naive", "joint", "ewc", "lwf", "er", "der_pp", "er_cflat", "doccl", "lca", "hgt", "cuber"}
```

- [ ] **Step 6: Run unit tests + config resolve**

Run: `cd CLUE && uv run pytest tests/methods/test_cuber.py -v`
Expected: PASS (2 passed)

Run: `cd CLUE && uv run python scripts/train.py method=cuber scenario=dil seed=42 method.transfer_alpha=0.5 method.epochs=3 wandb.mode=offline --cfg job 2>&1 | grep -E "name: cuber|transfer_alpha:"`
Expected: shows `name: cuber` and `transfer_alpha: 0.5` (no Hydra error)

- [ ] **Step 7: Lint + commit**

Run: `cd CLUE && uv run ruff check doccl/methods/cuber.py tests/methods/test_cuber.py && uv run black doccl/methods/cuber.py tests/methods/test_cuber.py`

```bash
cd CLUE && git add doccl/methods/cuber.py configs/method/cuber.yaml scripts/train.py tests/methods/test_cuber.py
git commit -m "AGENT IMPL: CUBER whole-network gradient-subspace baseline"
```

---

### Task 4: E2E lifecycle registration for hgt + cuber

**Files:**
- Modify: `tests/methods/test_all_methods_e2e.py` (imports; `_METHOD_CFG`; `_METHOD_CLS`)

**Interfaces:**
- Consumes: `HGT`, `CUBER` classes; the existing `_run_one_task` / CIL+DIL lifecycle harness.
- Produces: nothing new; extends the parametrized e2e coverage so the merge/steer/subspace lifecycle runs on a real LayoutLMv3 (CIL head growth + DIL).

- [ ] **Step 1: Add imports**

After `from doccl.methods.lca import LCA`:

```python
from doccl.methods.hgt import HGT
from doccl.methods.cuber import CUBER
```

- [ ] **Step 2: Add configs to `_METHOD_CFG`** (tiny subspace/batches for speed; include both α regimes for hgt)

After the `"lca": {...}` entry:

```python
    # HGT: head-only gradient-subspace transfer. alpha>0 exercises the steer/transfer path;
    # tiny subspace + 2 feature batches for speed. Frozen backbone (default).
    "hgt": {"transfer_alpha": 0.5, "subspace_k": 4, "subspace_n_batches": 2,
            "backbone_trainable": False},
    # CUBER: whole-network, exercises per-layer hooks. tiny subspace for speed.
    "cuber": {"transfer_alpha": 0.5, "subspace_k": 4, "subspace_n_batches": 2},
```

- [ ] **Step 3: Add classes to `_METHOD_CLS`**

After `"lca": LCA,`:

```python
    "hgt": HGT, "cuber": CUBER,
```

- [ ] **Step 4: Run the e2e for hgt + cuber**

Run: `cd CLUE && uv run pytest "tests/methods/test_all_methods_e2e.py" -k "hgt or cuber" -v`
Expected: PASS (4 passed — CIL + DIL × {hgt, cuber}); finite losses, CIL head growth survived, evaluate F1 in [0,100].

- [ ] **Step 5: Commit**

```bash
cd CLUE && git add tests/methods/test_all_methods_e2e.py
git commit -m "AGENT IMPL: register hgt + cuber in e2e lifecycle tests"
```

---

### Task 5: The α-ablation experiment runner (the go/no-go)

**Files:**
- Create: `scripts/run_hgt_ablation.sh`

**Interfaces:**
- Consumes: the `hgt`, `cuber`, `er`, `naive` methods; `scripts/train.py`; `analyze_results.py`.
- Produces: per-variant result dirs under `results/hgt_ablation/<tag>/` + a printed AA/BWT table. The headline contrast: `hgt α=0` (protection, expect BWT≈0) vs `hgt α>0` (transfer, expect BWT>0), vs `cuber`, `er`, `naive`.

- [ ] **Step 1: Write the runner**

```bash
#!/usr/bin/env bash
# HGT alpha-ablation on dil — the go/no-go: does head-gradient TRANSFER (alpha>0) give
# positive BWT vs head-gradient PROTECTION (alpha=0, BWT~0)? Plus CUBER (whole-network),
# ER (buffer reference), naive (floor). Resume-safe; local-2060 VRAM-safe.
# Knobs: SEED, EPOCHS, BATCH_SIZE, SUBSPACE_K, BACKBONE_TRAINABLE.
set -euo pipefail
cd "$(dirname "$0")/.."

SEED="${SEED:-42}"; EPOCHS="${EPOCHS:-5}"; BATCH_SIZE="${BATCH_SIZE:-2}"
SUBSPACE_K="${SUBSPACE_K:-32}"; BB="${BACKBONE_TRAINABLE:-false}"
ROOT="${ROOT:-results/hgt_ablation}"; SCEN=dil
PY() { uv run python "$@"; }

common=(scenario="$SCEN" seed="$SEED" method.epochs="$EPOCHS"
        training.batch_size="$BATCH_SIZE" training.gradient_checkpointing=true
        training.num_workers=0 wandb.mode=offline)

run() {  # $1 method, $2 tag, rest overrides
  local method="$1" tag="$2"; shift 2
  local odir="$ROOT/$tag" rundir
  rundir="$odir/${SCEN}_${method}_seed${SEED}"
  if [[ -f "$rundir/metrics.json" ]]; then echo "SKIP $tag (done)"; return 0; fi
  echo "=== RUN $tag :: method=$method $* ==="
  if ! uv run python scripts/train.py method="$method" "${common[@]}" output_dir="$odir" "$@"; then
    echo "!!! RUN $tag FAILED — continuing"
  fi
}

echo "############ HGT alpha-ablation ($SCEN, seed $SEED, ${EPOCHS}ep, bb_trainable=$BB) ############"
run naive  naive
run er     er
run hgt    hgt_a0   method.transfer_alpha=0.0 method.subspace_k="$SUBSPACE_K" method.backbone_trainable="$BB"
run hgt    hgt_a05  method.transfer_alpha=0.5 method.subspace_k="$SUBSPACE_K" method.backbone_trainable="$BB"
run hgt    hgt_a1   method.transfer_alpha=1.0 method.subspace_k="$SUBSPACE_K" method.backbone_trainable="$BB"
run cuber  cuber    method.transfer_alpha=0.5 method.subspace_k="$SUBSPACE_K"

echo "############ summary (AA / BWT) ############"
PY - "$ROOT" "$SCEN" "$SEED" <<'PYEOF'
import json, sys, glob, os
root, scen, seed = sys.argv[1:4]
print(f"{'variant':<12}{'method':<8}{'AA':>8}{'BWT':>8}")
for rundir in sorted(glob.glob(f"{root}/*/{scen}_*_seed{seed}")):
    tag = rundir.split("/")[-2]; meth = rundir.split("/")[-1].split("_seed")[0].replace(f"{scen}_", "")
    mp = os.path.join(rundir, "metrics.json"); aa = bwt = None
    if os.path.exists(mp):
        d = json.load(open(mp)); aa, bwt = d.get("AA"), d.get("BWT")
    f = lambda x: (f"{x:.2f}" if isinstance(x, (int, float)) else "-")
    print(f"{tag:<12}{meth:<8}{f(aa):>8}{f(bwt):>8}")
print("\nGO/NO-GO: hgt_a05/a1 BWT > hgt_a0 BWT (transfer beats protection) AND ideally BWT>0.")
print("Compare AA to er (buffer reference) and cuber. Flat alpha -> head-transfer falsified.")
PYEOF
echo "DONE."
```

- [ ] **Step 2: Make executable + dry-run the plan**

Run: `cd CLUE && chmod +x scripts/run_hgt_ablation.sh`
Run: `cd CLUE && grep -c "run " scripts/run_hgt_ablation.sh` (sanity: the 6 run lines present)
Expected: prints `6` or more.

- [ ] **Step 3: Commit**

```bash
cd CLUE && git add scripts/run_hgt_ablation.sh
git commit -m "AGENT IMPL: HGT alpha-ablation runner (dil go/no-go vs cuber/er/naive)"
```

- [ ] **Step 4: Run the experiment (GPU; ONE dataset builder at a time; ~hours)**

> Only when the GPU is free (no other train.py / dataset builder). Background it and monitor the log; do NOT block.

Run: `cd CLUE && SEED=42 EPOCHS=5 BATCH_SIZE=2 bash scripts/run_hgt_ablation.sh 2>&1 | tee results/hgt_ablation/run.log`

- [ ] **Step 5: Read the verdict**

Run: `cd CLUE && uv run python scripts/analyze_results.py --source local || true`
Then inspect the printed summary table.

**Decision:**
- **GO** if `hgt_a05`/`hgt_a1` BWT clearly exceeds `hgt_a0` BWT (transfer beats protection), ideally BWT > 0, while AA is competitive with `er`/`cuber`. → proceed to multi-backbone (BERT/LiLT/BROS) generality runs (separate plan; heavy → rented GPU).
- **NO-GO** if α has no effect on BWT (head-gradients orthogonal across tasks → no transfer to exploit). → record the negative; pivot to the analysis-paper framing (the spec's documented fallback). Either outcome is a publishable result.

---

## Self-Review

**Spec coverage:** HGT method (Task 2) ✓; CUBER baseline (Task 3) ✓; grad-subspace numerics incl. α=0 protection control (Task 1) ✓; `token_features` reuse (Task 2) ✓; frozen-vs-trainable + α + k config axes (Tasks 2/5) ✓; e2e CIL+DIL (Task 4) ✓; the `dil` α-ablation go/no-go vs CUBER/ER/naive (Task 5) ✓; analysis-paper fallback on flat α (Task 5 Step 5) ✓; multi-backbone generality flagged as a follow-up plan (Task 5) ✓ (heavy → rented GPU, out of this plan's scope per the spec).

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every command has expected output. ✓

**Type consistency:** `feature_moment`/`top_eigvecs`/`project_orth`/`steer_gradient` signatures identical across Tasks 1→2→3; `_stacked_basis`/`_steer_head_grad` (HGT) and `_steer_for_layer`/`_stacked_basis(name,…)` (CUBER) named consistently with their tests; `transfer_alpha`/`subspace_k`/`subspace_n_batches`/`backbone_trainable` config keys identical across method, config, runner. ✓

**Known caveat (carried from the spec, not a plan defect):** the positive-BWT-on-the-head hypothesis is unproven; Task 5's α-contrast is the test, with the analysis-paper fallback documented.
