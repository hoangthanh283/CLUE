"""LexMem memory head: flat key-value slot memory with sparse TF-IDF-selected updates.

The design follows "Continual Learning via Sparse Memory Finetuning" (Lin et al.,
arXiv 2510.15103), adapted from decoder-LM memory layers to an encoder token-
classification head:

  - **Keys** (N, d) are data-anchored (k-means / sampled from task-0 token hidden
    states) and FROZEN — the lexical/feature address space never drifts.
  - **Values** (N, C) live directly in logit space and are zero-init, so an
    untrained slot is an EXACT no-op on the base head's logits.
  - **Retrieval** is per-token top-k over cosine scores — routing is the forward
    pass itself (train == eval), replacing LexSlot's external soft gate.
  - **Update selection** is TF-IDF over slot *access counts* against a background
    corpus (task-0 + previously learned tasks): slots specific to the new data are
    trainable, slots carrying existing knowledge are frozen. This is LexSlot's
    lexical-signature idea at per-slot granularity, enforced by hard masking.

Gradient sparsity uses the same row-scaling grad-hook pattern as
``doccl.methods.lexslot_memory._MaskedSlots``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812 — canonical torch alias (repo-wide)

__all__ = ["LexicalMemoryHead"]

# Token-row chunk for the (tokens x N) score matmul — bounds transient memory on
# small-VRAM GPUs (2048 x 65536 fp32 ~= 512 MB) while staying one-shot on CPU tests.
_SCORE_CHUNK_CUDA = 2048
_SCORE_CHUNK_CPU = 8192


class LexicalMemoryHead(nn.Module):
    """Flat key-value memory producing an additive logit delta per token."""

    def __init__(
        self,
        n_slots: int,
        hidden_dim: int,
        n_labels: int,
        top_k: int = 32,
        temp: float = 0.05,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.hidden_dim = hidden_dim
        self.top_k = min(top_k, n_slots)
        self.temp = temp
        # Random unit keys until init_keys anchors them in task-0 feature space.
        self.register_buffer("keys", F.normalize(torch.randn(n_slots, hidden_dim), dim=-1))
        # Logit-space values, zero-init -> exact no-op until a slot is trained.
        self.values = nn.Parameter(torch.zeros(n_slots, n_labels))
        # Per-task binary trainability mask (set by select_topt).
        self.register_buffer("grad_mask", torch.zeros(n_slots))
        # First task to train each slot (-1 = unclaimed) — analysis artifact.
        self.register_buffer("slot_owner", torch.full((n_slots,), -1, dtype=torch.long))
        # Background usage statistics for the IDF term: #batches touching each slot.
        self.register_buffer("bg_hits", torch.zeros(n_slots))
        self.n_bg_batches = 0
        # Access-counting state (non-persistent scratch).
        self.register_buffer("_counts", torch.zeros(n_slots), persistent=False)
        self.counting = False
        # (B, L) attention mask set externally before a counting forward so padding
        # positions are excluded from access counts (SMF masks pad-accessed indices).
        self._count_mask: torch.Tensor | None = None
        self.values.register_hook(self._scale_rows)

    def _scale_rows(self, grad: torch.Tensor) -> torch.Tensor:
        return grad * self.grad_mask.to(device=grad.device, dtype=grad.dtype).unsqueeze(-1)

    # ── retrieval ────────────────────────────────────────────────────────────

    def _topk_scores(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunked cosine top-k. q (M, d) unit rows -> (M, k) scores, (M, k) indices."""
        chunk = _SCORE_CHUNK_CUDA if q.is_cuda else _SCORE_CHUNK_CPU
        keys = self.keys.to(q.dtype)
        vals, idxs = [], []
        for i in range(0, q.shape[0], chunk):
            s = q[i : i + chunk] @ keys.T  # (chunk, N)
            v, ix = s.topk(self.top_k, dim=-1)
            vals.append(v)
            idxs.append(ix)
        return torch.cat(vals), torch.cat(idxs)

    def delta(self, feats: torch.Tensor) -> torch.Tensor:
        """Additive logit delta for (B, L, d) classifier-input features -> (B, L, C).

        The query side is detached: keys are frozen and the backbone is frozen
        during memory training, so gradient flows into ``values`` only.
        """
        B, L, d = feats.shape  # noqa: N806
        q = F.normalize(feats.detach().reshape(-1, d).float(), dim=-1)
        with torch.no_grad():
            topv, topi = self._topk_scores(q)  # (B*L, k)
            w = torch.softmax(topv / self.temp, dim=-1)
            if self.counting:
                self._accumulate_counts(topi)
        v = self.values[topi]  # (B*L, k, C) — gather carries grad to selected rows
        out = (w.unsqueeze(-1) * v).sum(dim=1)
        return out.reshape(B, L, -1)

    # ── access counting / background stats ───────────────────────────────────

    def _accumulate_counts(self, topi: torch.Tensor) -> None:
        ki = topi
        if self._count_mask is not None:
            m = self._count_mask.reshape(-1).bool().to(ki.device)
            ki = ki[m]
        self._counts += torch.bincount(ki.reshape(-1), minlength=self.n_slots).to(
            self._counts.dtype
        )

    def start_counting(self) -> None:
        self._counts.zero_()
        self.counting = True

    def stop_counting(self) -> torch.Tensor:
        self.counting = False
        return self._counts.clone()

    def note_background_batch(self, batch_counts: torch.Tensor) -> None:
        """Fold one batch's access counts into the IDF background statistics."""
        self.bg_hits += (batch_counts.to(self.bg_hits.device) > 0).float()
        self.n_bg_batches += 1

    # ── slot selection ───────────────────────────────────────────────────────

    def tfidf_scores(self, task_counts: torch.Tensor) -> torch.Tensor:
        """SMF ranking: TF = task access share; IDF vs background batch hits."""
        c = task_counts.to(self.bg_hits.device)
        tf = c / c.sum().clamp(min=1.0)
        idf = torch.log((self.n_bg_batches + 1.0) / (self.bg_hits + 1.0))
        return tf * idf

    def select_topt(
        self, task_counts: torch.Tensor, t: int, task_id: int, mode: str = "tfidf"
    ) -> torch.Tensor:
        """Pick the top-t slots for this task and make ONLY them trainable.

        Returns the selected indices. Ownership is recorded on first claim only;
        the grad mask is per-task (reset on every call).
        """
        c = task_counts.to(self.bg_hits.device)
        t = min(t, self.n_slots)
        if mode == "random":
            accessed = (c > 0).nonzero().flatten()
            perm = accessed[torch.randperm(len(accessed), device=accessed.device)]
            idx = perm[:t]
        else:
            scores = self.tfidf_scores(c) if mode == "tfidf" else c
            # Never select a slot the task did not access at all.
            scores = torch.where(c > 0, scores, torch.full_like(scores, float("-inf")))
            idx = scores.topk(t).indices
            idx = idx[c[idx] > 0]  # drop -inf fillers when #accessed < t
        self.grad_mask.zero_()
        self.grad_mask[idx] = 1.0
        fresh = self.slot_owner[idx] == -1
        self.slot_owner[idx[fresh]] = task_id
        return idx

    # ── key initialization ───────────────────────────────────────────────────

    @torch.no_grad()
    def init_keys(self, feats: torch.Tensor, mode: str = "kmeans", iters: int = 8) -> None:
        """Anchor keys in data: (M, d) token features -> N unit-norm keys.

        ``sample``: random subsample (all rows + random-init padding when M < N —
        random unit keys are near-orthogonal to real features so they stay cold).
        ``kmeans``: chunked Lloyd iterations from a sampled init (only meaningfully
        different from sampling when M >> N). ``random``: leave the random init.
        """
        if mode == "random":
            return
        feats = F.normalize(feats.float().to(self.keys.device), dim=-1)
        M = feats.shape[0]  # noqa: N806
        if M == 0:
            return
        if self.n_slots >= M:
            self.keys[:M] = feats
            return
        pick = torch.randperm(M, device=feats.device)[: self.n_slots]
        centroids = feats[pick].clone()
        if mode == "kmeans" and 2 * self.n_slots < M:
            centroids = self._lloyd(feats, centroids, iters)
        self.keys.copy_(F.normalize(centroids, dim=-1))

    @staticmethod
    def _lloyd(feats: torch.Tensor, centroids: torch.Tensor, iters: int) -> torch.Tensor:
        """A few chunked Lloyd iterations (cosine assignment, mean update)."""
        n = centroids.shape[0]
        chunk = _SCORE_CHUNK_CUDA if feats.is_cuda else _SCORE_CHUNK_CPU
        for _ in range(iters):
            assign = torch.empty(feats.shape[0], dtype=torch.long, device=feats.device)
            for i in range(0, feats.shape[0], chunk):
                assign[i : i + chunk] = (feats[i : i + chunk] @ centroids.T).argmax(dim=-1)
            sums = torch.zeros_like(centroids)
            sums.index_add_(0, assign, feats)
            cnt = torch.bincount(assign, minlength=n).clamp(min=1).unsqueeze(-1)
            new = sums / cnt
            # Empty clusters keep their previous centroid.
            empty = torch.bincount(assign, minlength=n) == 0
            new[empty] = centroids[empty]
            centroids = F.normalize(new, dim=-1)
        return centroids

    # ── CIL label growth ─────────────────────────────────────────────────────

    @torch.no_grad()
    def expand_labels(self, new_n_labels: int) -> None:
        old = self.values
        if new_n_labels <= old.shape[1]:
            return
        new = torch.zeros(self.n_slots, new_n_labels, device=old.device, dtype=old.dtype)
        new[:, : old.shape[1]] = old
        self.values = nn.Parameter(new)
        self.values.register_hook(self._scale_rows)
