"""Frozen pre-trained-model baselines for image CIL: SimpleCIL and RanPAC.

No gradient training: the ViT trunk stays at its pre-trained weights; each task only adds
statistics of the frozen CLS feature (the classifier input).

SimpleCIL (Zhou et al., 2023): cosine nearest-class-mean on class prototypes.
RanPAC (McDonnell et al., NeurIPS 2023), without its optional first-session adaptation:
fixed random projection φ(f) = ReLU(f W), W ∈ R^{d×M}; ridge classifier from the
accumulated Gram matrix G = Σ φφᵀ and targets Q = Σ φ yᵀ; λ chosen on a held-out split.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from doccl.eval.metrics import compute_eval_metrics
from doccl.methods.base import ContinualMethod
from doccl.types import EvalMetrics, TaskInfo, TrainMetrics

log = logging.getLogger(__name__)

LAMBDAS = (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5)


def ridge_solve(gram: torch.Tensor, q: torch.Tensor, lam: float) -> torch.Tensor:
    """β = (G + λI)^-1 Q, solved in float64 for stability."""
    eye = torch.eye(gram.shape[0], dtype=torch.float64, device=gram.device)
    return torch.linalg.solve(gram.double() + lam * eye, q.double()).float()


@torch.no_grad()
def cls_features(model, loader: DataLoader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Frozen classifier-input features (layer-normed CLS) and labels for a loader."""
    model.eval()
    feats, labels = [], []
    for batch in loader:
        x = batch["pixel_values"].to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            f = model.encode_query({"pixel_values": x})
        feats.append(f.float())
        labels.append(batch["labels"].to(device))
    return torch.cat(feats), torch.cat(labels)


class _FrozenPTM(ContinualMethod):
    def trainable_parameters(self):
        return []

    def train_task(self, task: TaskInfo, train_loader: DataLoader, val_loader=None) -> TrainMetrics:
        f, y = cls_features(self.model, train_loader, self.device)
        self._absorb(f, y)
        return TrainMetrics(task_id=task.task_id, loss=0.0, n_steps=0)

    def evaluate(self, eval_loaders: dict[int, DataLoader]) -> dict[int, EvalMetrics]:
        out = {}
        for tid, loader in eval_loaders.items():
            f, y = cls_features(self.model, loader, self.device)
            if not self._fitted():  # zero-shot reading before the first task is absorbed
                preds = torch.zeros_like(y)
            else:
                preds = self._predict(f).argmax(dim=-1)
            m = compute_eval_metrics(preds.tolist(), y.tolist(), {}, "image")
            out[tid] = EvalMetrics(task_id=tid, f1=m["f1"], n_samples=len(y))
        return out


class SimpleCIL(_FrozenPTM):
    name = "simplecil"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.protos: dict[int, torch.Tensor] = {}

    def _fitted(self) -> bool:
        return bool(self.protos)

    def _absorb(self, f, y):
        for c in y.unique().tolist():
            self.protos[c] = f[y == c].mean(0)

    def _predict(self, f):
        n = max(self.protos) + 1
        p = torch.zeros(n, f.shape[1], device=f.device)
        for c, v in self.protos.items():
            p[c] = v
        return F.normalize(f, dim=-1) @ F.normalize(p, dim=-1).T


class RanPAC(_FrozenPTM):
    name = "ranpac"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.m = int(config.get("rp_dim", 10000))
        self.val_frac = float(config.get("val_frac", 0.2))
        g = torch.Generator().manual_seed(int(config.get("rp_seed", 0)))
        self.w = torch.randn(model.hidden_size, self.m, generator=g)
        self.feats: list[torch.Tensor] = []  # kept only for the λ split (CPU)
        self.labels: list[torch.Tensor] = []
        self.beta: torch.Tensor | None = None
        self.lam: float | None = None

    def _fitted(self) -> bool:
        return self.beta is not None

    def _phi(self, f: torch.Tensor) -> torch.Tensor:
        return F.relu(f @ self.w.to(f.device))

    def _stats(self, f, y, n_cls, w=None, chunk: int = 4096):
        """G = Σ wφφᵀ, Q = Σ wφyᵀ, accumulated in chunks (φ for 50k×10k would be 2 GB)."""
        g = torch.zeros(self.m, self.m, device=f.device)
        q = torch.zeros(self.m, n_cls, device=f.device)
        for i in range(0, len(f), chunk):
            phi = self._phi(f[i : i + chunk])
            target = F.one_hot(y[i : i + chunk], n_cls).float()
            if w is not None:  # weighted ridge: rows scaled by sqrt(weight)
                s = w[i : i + chunk].sqrt()[:, None]
                phi, target = phi * s, target * s
            g += phi.T @ phi
            q += phi.T @ target
        return g, q

    def _absorb(self, f, y):
        self.feats.append(f.cpu())
        self.labels.append(y.cpu())
        self.fit(torch.cat(self.feats).to(self.device), torch.cat(self.labels).to(self.device))

    def fit(self, x: torch.Tensor, t: torch.Tensor, w: torch.Tensor | None = None) -> None:
        """Choose λ on a held-out split, then solve the (optionally weighted) ridge on all."""
        n_cls = int(t.max()) + 1
        gen = torch.Generator(device="cpu").manual_seed(0)
        perm = torch.randperm(len(t), generator=gen).to(self.device)
        n_val = int(len(t) * self.val_frac)
        val, tr = perm[:n_val], perm[n_val:]
        g_tr, q_tr = self._stats(x[tr], t[tr], n_cls, None if w is None else w[tr])
        best = max(
            LAMBDAS,
            key=lambda lam: (self._predict_with(x[val], ridge_solve(g_tr, q_tr, lam)) == t[val])
            .float()
            .mean()
            .item(),
        )
        g, q = self._stats(x, t, n_cls, w)
        self.beta, self.lam = ridge_solve(g, q, best), best
        log.info("ranpac: %d samples, %d classes, lambda=%g", len(t), n_cls, best)

    def _predict_with(self, f, beta, chunk: int = 4096):
        return torch.cat(
            [(self._phi(f[i : i + chunk]) @ beta).argmax(-1) for i in range(0, len(f), chunk)]
        )

    def _predict(self, f, chunk: int = 4096):
        beta = self.beta.to(f.device)
        return torch.cat([self._phi(f[i : i + chunk]) @ beta for i in range(0, len(f), chunk)])
