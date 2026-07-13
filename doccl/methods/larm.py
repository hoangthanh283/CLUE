"""LARM — Lexical-Associative Rewrite Memory (fuse latent replay + lexical routing).

One mechanism, two roles. A lexically-keyed associative memory ``M`` rewrites each
document's frozen-boundary (layer-``k``) features with an **additive low-rank correction**,
and that memory's values are kept correct by **compressed latent replay** (inherited from
CoLaR). Latent replay is the *writer* (keeps the correction live as the plastic weights move);
lexical routing is the *reader* (applies the correction, keyed by the doc's OCR signature).

Why this fuses instead of stacking — each component repairs the OTHER's named weakness:
- LexSlot's slots are frozen after their task (stale) → LARM's values are replay-refit (live).
- CoLaR's replay is global/untargeted → LARM's correction is lexically attributed (per cell).

Why it survives the consistency law that killed SLR/AGLR/coreset: LARM never REPLACES features
with a marginal summary. Every token's base feature is the real, frozen, whole-document
activation of the actual current document; LARM only ADDS a routed low-rank Δ on top. The
(feature, position, label) binding is intact by construction — the correction is a nudge, not
a substitute. ``add_not_replace=False`` flips this to the falsified replace-mode (the Gate-0
ablation: add must hold retention, replace must collapse).

The read path (``h → h + Δh``) runs identically at train and eval; the key is the OCR bag of
the current doc's input_ids, available at test time (no task-ID leakage). The routed retrieval
IS the forward-transfer lever: a new doc reuses corrections learned for lexically-similar past
docs — the only mechanism in the codebase that can move FWT (measurable on xlingual, where
vocabularies overlap; on dil the router degenerates to per-task and LARM ≈ CoLaR — fine for BWT).
"""

from __future__ import annotations

import logging
import random

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.utils.data import DataLoader

from doccl.methods.colar import CoLaR
from doccl.methods.hybrid_routed_prompt import sparse_doc_vectors
from doccl.types import TaskInfo

log = logging.getLogger(__name__)

__all__ = ["LARM", "RewriteMemory"]


class RewriteMemory(nn.Module):
    """Lexically-keyed low-rank feature-correction memory.

    Cells: L2-normalised OCR key ``K[c]`` (V,), low-rank factors ``down[c]`` (d, r) and
    ``up[c]`` (r, d) with ``up`` zero-init so a fresh/untrained cell is an exact no-op.
    Read: attention ``α = softmax(q·Kᵀ/τ)`` over cells → additive low-rank rewrite of ``h``.
    """

    def __init__(self, vocab_size: int, d: int, rank: int, tau: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.rank = rank
        self.tau = tau
        self.keys: list[torch.Tensor] = []  # each (V,), L2-normalised; kept on CPU, moved on read
        self.down = nn.ParameterList()
        self.up = nn.ParameterList()

    def n_cells(self) -> int:
        return len(self.keys)

    @torch.no_grad()
    def add_cell(self, key: torch.Tensor, device) -> None:
        """Add one memory cell for a stored doc: frozen lexical key + zero-init correction."""
        self.keys.append(key.detach().float().cpu())
        down = nn.Parameter(torch.randn(self.d, self.rank, device=device) * 0.02)
        up = nn.Parameter(torch.zeros(self.rank, self.d, device=device))  # no-op until trained
        self.down.append(down)
        self.up.append(up)

    def read(self, h: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Routed additive correction. h: (B, L, d); q: (B, V) L2-normalised. Returns Δh (B,L,d).

        Δh = Σ_c α[:,c] · (h @ down_c) @ up_c, with α = softmax(q @ Kᵀ / τ). Zero when empty.
        """
        if not self.keys:
            return torch.zeros_like(h)
        keymat = torch.stack(self.keys).to(h.device)  # (C, V)
        alpha = F.softmax((q @ keymat.T) / self.tau, dim=-1)  # (B, C)
        # Vectorised over cells (was a C-iteration Python loop — matters at the 150-cell d50
        # headline). Contract alpha into the r-space projection BEFORE the up-projection so we
        # never materialise the (B,C,L,d) per-cell tensor: Δ = ((Σ_c α·(h@down_c)) )@up  — but
        # up differs per cell, so weight in r-space then sum-project per cell via einsum.
        down = torch.stack(list(self.down))  # (C, d, r)
        up = torch.stack(list(self.up))  # (C, r, d)
        hd = torch.einsum("bld,cdr->bclr", h, down)  # (B, C, L, r)
        hd = hd * alpha.view(alpha.shape[0], alpha.shape[1], 1, 1)  # weight each cell in r-space
        delta = torch.einsum("bclr,crd->bld", hd, up)  # (B, L, d), summed over cells
        return delta


class LARM(CoLaR):
    """CoLaR replay + a lexically-routed, replay-maintained additive feature rewrite."""

    name = "larm"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.mem_rank = int(config.get("mem_rank", 16))
        self.mem_tau = float(config.get("mem_tau", 0.1))
        self.route_replay = bool(config.get("route_replay", True))
        self.mem_warmup_steps = int(config.get("mem_warmup_steps", 0))
        self.add_not_replace = bool(config.get("add_not_replace", True))
        self._vocab_size = int(self.model.model.config.vocab_size)
        self.mem = RewriteMemory(
            self._vocab_size, self.model.hidden_size, self.mem_rank, self.mem_tau
        )
        self.mem.to(self.device)
        # query for the CURRENT forward (batch OCR signature), stashed before each model call;
        # and a per-replay-forward query override (routes by the STORED sig, not dummy_ids).
        self._cur_query: torch.Tensor | None = None
        self._replay_query: torch.Tensor | None = None
        self._cur_task_sig: torch.Tensor | None = None
        self._step = 0

    # ── the additive-rewrite branch on the layer-k pre-hook ───────────────────

    def _pre_hook(self, module, args, kwargs):
        # Capture branch: unchanged (bank the layer-k input, no rewrite).
        if self._capture is not None:
            return super()._pre_hook(module, args, kwargs)
        hidden = args[0] if args else kwargs["hidden_states"]
        # Inject branch (replay): the base hidden becomes the reconstructed activation; THEN
        # the routed memory delta is added on top, so replay CE trains M (keeps it live).
        if self._inject is not None:
            hidden = self._inject.to(device=hidden.device, dtype=hidden.dtype)
        # Route the memory: replay forward uses the STORED sig; current batch uses _cur_query.
        q = self._replay_query if self._replay_query is not None else self._cur_query
        if self.mem.n_cells() > 0 and q is not None:
            delta = self.mem.read(hidden.float(), q.to(hidden.device)).to(hidden.dtype)
            warm = (
                self.mem_warmup_steps > 0
                and self._step < self.mem_warmup_steps
                and self.model.training
            )
            if warm:  # early: train M from replay only, don't chase the moving current loss
                delta = delta.detach()
            hidden = delta if not self.add_not_replace else hidden + delta
        elif self._inject is None:
            return None  # nothing to do (no memory yet, normal forward) → inherit base no-op
        if args:
            return (hidden, *args[1:]), kwargs
        kwargs = dict(kwargs, hidden_states=hidden)
        return args, kwargs

    def _query_of(self, input_ids: torch.Tensor) -> torch.Tensor:
        return sparse_doc_vectors(input_ids, self._vocab_size)

    # ── train: stash the current-batch query so the pre-hook can route ────────

    def train_task(self, task, train_loader, val_loader=None):
        # Wrap the model so every forward stashes its query first. Simplest robust hook:
        # monkeypatch is avoided — instead we set _cur_query in a forward pre-hook on the wrapper.
        handle = self.model.register_forward_pre_hook(self._stash_query, with_kwargs=True)
        try:
            return super().train_task(task, train_loader, val_loader)
        finally:
            handle.remove()

    def _stash_query(self, _module, _args, kwargs):
        ids = kwargs.get("input_ids")
        # replay forward uses dummy_ids and sets _replay_query explicitly; skip stashing then
        if self._replay_query is None and torch.is_tensor(ids):
            self._cur_query = self._query_of(ids)
        if self.model.training:
            self._step += 1
        return None

    def evaluate(self, eval_loaders):
        # eval forwards also need the query stashed (read path is train/eval-identical)
        handle = self.model.register_forward_pre_hook(self._stash_query, with_kwargs=True)
        try:
            return super().evaluate(eval_loaders)
        finally:
            handle.remove()
            self._cur_query = None

    # ── write: bank the doc's lexical signature + add a memory cell ───────────

    def _capture_task(self, train_loader: DataLoader) -> None:
        n_before = len(self.store)
        super()._capture_task(train_loader)  # CoLaR: banks us/v/bbox/mask/labels + compresses
        # ponytail: the parent banks the FIRST docs_per_task docs of the loader in order; we
        # mirror that draw to align each stored doc with its OCR signature + a memory cell.
        # Holds because _capture_task does not reshuffle within a single call. If the loader
        # were re-shuffled between passes this would misalign — assert-safe: same first-N order.
        self._attach_sigs_and_cells(train_loader, n_before)

    @torch.no_grad()
    def _attach_sigs_and_cells(self, train_loader: DataLoader, n_before: int) -> None:
        """Give each doc banked in [n_before:] its OCR signature and a fresh memory cell.
        Mirrors the parent's first-N draw so signatures align with the stored docs."""
        need = len(self.store) - n_before
        got = 0
        for batch in train_loader:
            ids = batch["input_ids"]
            for i in range(ids.shape[0]):
                if got >= need:
                    break
                sig = self._query_of(ids[i : i + 1].to(self.device))[0].cpu()
                self.store[n_before + got]["sig"] = sig
                self.mem.add_cell(sig, self.device)
                got += 1
            if got >= need:
                break
        log.info("larm: banked %d sigs + memory cells (total cells=%d)", got, self.mem.n_cells())

    # ── replay: route by the STORED sig, weight the draw lexically ────────────

    def _sample_replay(self):
        if not self.store:
            return None
        k = min(self.replay_batch_size, len(self.store))
        if self.route_replay and self._cur_task_sig is not None:
            # weight the draw by cosine(current-task sig, stored sig): replay the contested
            # (lexically-near) region hard, far regions lightly.
            sims = torch.tensor(
                [
                    float(self._cur_task_sig @ d["sig"]) if d.get("sig") is not None else 1.0
                    for d in self.store
                ]
            )
            w = torch.softmax(sims / self.mem_tau, dim=0)
            idx = list(torch.utils.data.WeightedRandomSampler(w, k, replacement=False))
            docs = [self.store[i] for i in idx]
        else:
            docs = random.sample(self.store, k)
        # reconstruct hidden from CoLaR factors (parent does this in its own _sample_replay);
        # carry the stored sig so _replay_forward routes the memory read by the right cell.
        return {
            "hidden": torch.stack(
                [(d["us"].float() @ d["v"].float()).to(torch.float16) for d in docs]
            ),
            "bbox": torch.stack([d["bbox"] for d in docs]),
            "attention_mask": torch.stack([d["attention_mask"] for d in docs]),
            "labels": torch.stack([d["labels"] for d in docs]),
            "sig": torch.stack([d["sig"] for d in docs]),
        }

    def _replay_forward(self, replay: dict[str, torch.Tensor]):
        # route the memory read by the stored sig (the cell each doc originally activated),
        # not by the dummy_ids the parent injects.
        self._replay_query = replay["sig"].to(self.device)
        try:
            return super()._replay_forward({k: v for k, v in replay.items() if k != "sig"})
        finally:
            self._replay_query = None

    # ── track the current task's signature (for lexical replay weighting) ─────

    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        super().before_task(task, train_loader)
        self._cur_task_sig = self._accumulate_task_sig(train_loader)

    @torch.no_grad()
    def _accumulate_task_sig(self, train_loader: DataLoader) -> torch.Tensor:
        acc = torch.zeros(self._vocab_size)
        for n, batch in enumerate(train_loader):
            acc = acc + self._query_of(batch["input_ids"].to(self.device)).sum(0).cpu()
            if n >= 19:  # a few batches suffice for the task-level bag
                break
        return F.normalize(acc, dim=0)

    def trainable_parameters(self):
        return super().trainable_parameters() + list(self.mem.parameters())

    def memory_bytes(self) -> int:
        total = super().memory_bytes()
        for c in range(self.mem.n_cells()):
            total += (self.mem.down[c].numel() + self.mem.up[c].numel()) * 4
            total += self.mem.keys[c].numel() * 4  # int-sparse in practice; fp32 upper bound
        return total
