"""Read-side utilities over a LatentReplay/CoLaR store — the datastore VIEW.

Everything below ``split_layer_k`` is frozen after task 0, so the banked layer-k
activations are drift-free by construction and directly comparable to a test
document's layer-k features at any later time. That makes the replay store a stable
token-level labeled datastore usable at INFERENCE — a memory that is read, not
gradient-replayed. This module holds the shared machinery: datastore build from
per-doc SVD factors (or raw hiddens), chunked cosine-kNN label readout, and doc-level
retrieval keys. Used by CoLaRKNN (kNN readout head) and CoLaRMbPA (episodic retrieval).

Adds zero buffer bytes: features are reconstructed from the factors the store already
holds, and the GPU copy lives only for the duration of one ``evaluate()`` call.
"""

from __future__ import annotations

import torch

__all__ = ["LatentDatastore"]


def _doc_features(d: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct a banked doc's supervised-token features + labels.

    Banked hiddens cover the full encoder sequence (text + visual patch positions);
    labels cover only the text slice — slice before masking, as _kcenter_select does.
    """
    h = (d["us"].float() @ d["v"].float()) if "us" in d else d["hidden"].float()
    labels = d["labels"]
    mask = d["attention_mask"].bool() & (labels != -100)
    return h[: labels.shape[0]][mask], labels[mask]


class LatentDatastore:
    """Lazy token-level (feature, label) view over a replay store, cosine-kNN readout."""

    def __init__(self) -> None:
        self._x: torch.Tensor | None = None  # (N, d) fp16 cpu, L2-normalized rows
        self._y: torch.Tensor | None = None  # (N,) int64 cpu
        self._dev: tuple[torch.Tensor, torch.Tensor] | None = None  # device copies
        self._dirty = True

    def __len__(self) -> int:
        return 0 if self._x is None else int(self._x.shape[0])

    def invalidate(self) -> None:
        """Mark stale (call after the store grows); next build() re-reads the store."""
        self._dirty = True
        self._dev = None

    def build(self, store: list[dict[str, torch.Tensor]]) -> None:
        if not self._dirty:
            return
        feats, labs = [], []
        for d in store:
            h, y = _doc_features(d)
            feats.append(torch.nn.functional.normalize(h, dim=1).half())
            labs.append(y)
        self._x = torch.cat(feats) if feats else None
        self._y = torch.cat(labs) if labs else None
        self._dev = None
        self._dirty = False

    def free_device(self) -> None:
        """Drop the GPU copy (keep the CPU datastore) — call at the end of evaluate()."""
        self._dev = None

    def _device_tensors(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._dev is None:
            self._dev = (self._x.to(device), self._y.to(device))
        return self._dev

    @torch.no_grad()
    def knn_label_dist(
        self,
        query: torch.Tensor,
        top_k: int,
        tau: float | None,
        n_labels: int,
        class_balance: bool = False,
        chunk: int = 512,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        """(..., d) query tokens -> (..., n_labels) probability simplex per token.

        softmax(cos_sim / tau) vote over the top_k neighbors (tau=None: uniform vote).
        class_balance divides the vote mass per label by that label's datastore count
        before renormalizing — counters "O"-domination in sparse-entity tasks.
        Chunked over query rows; peak VRAM ~ chunk x len(self) fp16 sims.
        """
        x, y = self._device_tensors(device)
        counts = torch.bincount(y, minlength=n_labels).clamp(min=1).float()
        q = query.reshape(-1, query.shape[-1])
        out = torch.empty(q.shape[0], n_labels, device=device)
        k = min(top_k, x.shape[0])
        for i in range(0, q.shape[0], chunk):
            qc = torch.nn.functional.normalize(q[i : i + chunk].to(device).half(), dim=1)
            vals, idx = (qc @ x.T).float().topk(k, dim=1)
            w = torch.full_like(vals, 1.0 / k) if tau is None else torch.softmax(vals / tau, 1)
            dist = torch.zeros(vals.shape[0], n_labels, device=device)
            dist.scatter_add_(1, y[idx], w)
            if class_balance:
                dist = dist / counts
            out[i : i + chunk] = dist / dist.sum(-1, keepdim=True).clamp(min=1e-8)
        return out.reshape(*query.shape[:-1], n_labels)

    @staticmethod
    def doc_keys_mean_feature(store: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        """(N_docs, d) mean supervised-token feature per banked doc — R3 retrieval key."""
        return torch.stack([_doc_features(d)[0].mean(0) for d in store])
