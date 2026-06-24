"""Hybrid-Routed Prompt pool (HRP) — dense + sparse task routing.

A side-memory CL method in the L2P family: a frozen backbone with a learnable
prompt pool whose slots are **partitioned into per-task blocks**. The novelty is the
router. Instead of routing on the dense CLS query alone (L2P/CODA), HRP fuses two
signals:

    * **dense** — cosine(CLS query, learnable slot key)        — paraphrase / layout
    * **sparse** — cosine(OCR-token bag-of-words, slot signature) — exact vocabulary

fused by **Reciprocal Rank Fusion** (rank-based, so the two heterogeneous scores need
no calibration). The sparse signal is BM25/TF-IDF over the document OCR token stream
(``batch["input_ids"]``) — the ingredient no doc-CL baseline uses. It fixes the L2P
failure mode where two layout-similar forms (a receipt vs a form) collide in CLS space
but differ sharply in vocabulary ("TOTAL/TAX" vs "Name/Date").

``router ∈ {dense, sparse, hybrid}`` (config) selects which engine(s) feed the ranking,
so the controlled ablation (does sparse beat dense routing?) is ONE method toggled by
config, not three separate methods.

Because slots are task-pinned, the method can measure **routing accuracy**: at eval it
knows each loader's true task id, so it records whether the top-1 routed block is the
one trained on that task and writes ``results/<run>/routing.json``. That routing
hit-rate — not "no forgetting" and not speed — is the claim this method exists to test.

The train/eval loop, query extraction, prompt injection, label truncation and
prompt-aware early stopping are all inherited from ``PromptBasedMethod``; HRP only
defines the pool, the router, the task→block pinning, and the routing log.
"""

from __future__ import annotations

import json
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from doccl.eval.metrics import compute_token_f1
from doccl.methods.prompt_base import PromptBasedMethod
from doccl.types import EvalMetrics, TaskInfo

log = logging.getLogger(__name__)

__all__ = ["HybridRoutedPrompt", "HybridPromptPool", "sparse_doc_vectors"]


def sparse_doc_vectors(
    input_ids: torch.Tensor, vocab_size: int, idf: torch.Tensor | None = None
) -> torch.Tensor:
    """Bag-of-token-ids vector per document, optionally IDF-weighted, L2-normalised.

    Args:
        input_ids: (B, L) long token ids.
        vocab_size: width V of the output vectors (the backbone vocab size).
        idf: optional (V,) inverse-document-frequency weights; ones if None.

    Returns:
        (B, V) float — each row is the (idf-weighted) term-frequency over the
        document's OCR tokens, L2-normalised so a dot product is a cosine.
    """
    B = input_ids.shape[0]
    counts = torch.zeros(B, vocab_size, device=input_ids.device)
    # scatter_add term frequencies; clamp (out-of-place — never mutate the caller's
    # batch tensor) guards against any id ≥ vocab_size.
    ids = input_ids.clamp(0, vocab_size - 1)
    counts.scatter_add_(1, ids, torch.ones_like(ids, dtype=counts.dtype))
    if idf is not None:
        counts = counts * idf.unsqueeze(0)
    return F.normalize(counts, dim=-1)


class HybridPromptPool(nn.Module):
    """Task-pinned prompt pool with a dense key and a sparse signature per slot.

    Slots ``[0, N)`` are split into ``n_tasks`` contiguous blocks of ``slots_per_task``
    (``N = n_tasks * slots_per_task``). Each slot carries:

        prompts:    (N, L_p, D) — learnable prompt vectors
        keys:       (N, D)      — learnable dense keys (cosine vs CLS query)
        signatures: (N, V)      — sparse OCR-token signatures (buffer, accumulated
                                  from the documents that trained the slot's block)

    ``route`` returns top-k slot indices under the chosen engine plus the dense
    key-pull auxiliary loss (kept identical to L2P so dense-only HRP == L2P).
    """

    def __init__(
        self,
        n_tasks: int,
        slots_per_task: int = 2,
        prompt_length: int = 5,
        hidden_dim: int = 768,
        vocab_size: int = 50265,
        rrf_k: int = 60,
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.slots_per_task = slots_per_task
        self.n_prompts = n_tasks * slots_per_task
        self.prompt_length = prompt_length
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.rrf_k = rrf_k
        self.prompts = nn.Parameter(torch.randn(self.n_prompts, prompt_length, hidden_dim) * 0.02)
        self.keys = nn.Parameter(torch.randn(self.n_prompts, hidden_dim) * 0.02)
        # Sparse signatures + document-frequency accumulator are state, not learned.
        self.register_buffer("signatures", torch.zeros(self.n_prompts, vocab_size))
        self.register_buffer("doc_freq", torch.zeros(vocab_size))
        self.register_buffer("n_docs", torch.zeros(()))

    # ─── block <-> slot helpers ─────────────────────────────────────────────────
    def block_slots(self, task_id: int) -> slice:
        """Slot index slice owned by ``task_id``."""
        start = task_id * self.slots_per_task
        return slice(start, start + self.slots_per_task)

    def slot_to_block(self, slot_idx: torch.Tensor) -> torch.Tensor:
        """Map slot indices → owning block (task) ids."""
        return slot_idx // self.slots_per_task

    @property
    def idf(self) -> torch.Tensor | None:
        """Smoothed IDF from accumulated document frequencies (None before any task)."""
        if float(self.n_docs) == 0:
            return None
        return torch.log((1.0 + self.n_docs) / (1.0 + self.doc_freq)) + 1.0

    # ─── routing ────────────────────────────────────────────────────────────────
    def _dense_scores(self, dense_q: torch.Tensor) -> torch.Tensor:
        q = F.normalize(dense_q, dim=-1)
        k = F.normalize(self.keys, dim=-1)
        return q @ k.T  # (B, N)

    def _sparse_scores(self, sparse_q: torch.Tensor) -> torch.Tensor:
        sig = F.normalize(self.signatures, dim=-1)  # rows of all-zero signatures stay zero
        return sparse_q @ sig.T  # (B, N)

    @staticmethod
    def _rrf(*score_mats: torch.Tensor, k: int = 60) -> torch.Tensor:
        """Reciprocal Rank Fusion of several (B, N) score matrices → fused (B, N).

        Each engine ranks the N slots; a slot's fused score is Σ 1/(k + rank). Rank is
        0 for the engine's best slot. Rank-based → no cross-engine score calibration.
        """
        fused = torch.zeros_like(score_mats[0])
        for scores in score_mats:
            # rank 0 = highest score. argsort(descending) gives slot ids by rank;
            # invert to get each slot's rank.
            order = scores.argsort(dim=-1, descending=True)
            ranks = torch.empty_like(order)
            ar = torch.arange(scores.shape[-1], device=scores.device).expand_as(order)
            ranks.scatter_(1, order, ar)
            fused = fused + 1.0 / (k + ranks.to(fused.dtype))
        return fused

    def route(
        self, dense_q: torch.Tensor, sparse_q: torch.Tensor, top_k: int, router: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (selected slot indices (B, top_k), dense key-pull aux loss).

        ``router``: 'dense' | 'sparse' | 'hybrid'. The key-pull loss always uses the
        dense keys of the selected slots (it trains the dense keys regardless of the
        routing engine, so the dense path stays comparable to L2P).
        """
        top_k = min(top_k, self.n_prompts)
        dense = self._dense_scores(dense_q)
        if router == "dense":
            fused = dense
        elif router == "sparse":
            fused = self._sparse_scores(sparse_q)
        elif router == "hybrid":
            fused = self._rrf(dense, self._sparse_scores(sparse_q), k=self.rrf_k)
        else:
            raise ValueError(f"unknown router '{router}' (dense|sparse|hybrid)")
        topk_idx = fused.topk(top_k, dim=-1).indices  # (B, top_k)
        # Pull the dense keys of the selected slots toward the query (L2P aux loss).
        sel_keys = F.normalize(self.keys[topk_idx], dim=-1)  # (B, top_k, D)
        q = F.normalize(dense_q, dim=-1).unsqueeze(1)  # (B, 1, D)
        key_pull = (1.0 - (sel_keys * q).sum(-1)).mean()
        return topk_idx, key_pull

    def gather_prompts(self, slot_idx: torch.Tensor) -> torch.Tensor:
        """(B, top_k) slot ids → (B, top_k * L_p, D) prompt embeddings."""
        selected = self.prompts[slot_idx]  # (B, top_k, L_p, D)
        B, K, Lp, D = selected.shape
        return selected.reshape(B, K * Lp, D)

    # ─── signature accumulation (called in after_task) ──────────────────────────
    @torch.no_grad()
    def accumulate_signature(self, task_id: int, input_ids: torch.Tensor) -> None:
        """Fold one batch of OCR tokens into this task's block signature + doc-freq."""
        bow = sparse_doc_vectors(input_ids, self.vocab_size)  # (B, V) L2-normed TF
        sig = bow.sum(0).to(self.signatures.device)  # (V,)
        sl = self.block_slots(task_id)
        # Every slot in the block shares the task's signature.
        self.signatures[sl] += sig.unsqueeze(0)
        # Document frequency: count docs containing each token (presence, not count).
        present = input_ids.clamp(0, self.vocab_size - 1).new_zeros(
            input_ids.shape[0], self.vocab_size
        )
        present.scatter_(1, input_ids.clamp(0, self.vocab_size - 1), 1)
        self.doc_freq += present.sum(0).to(self.doc_freq.device).float()
        self.n_docs += float(input_ids.shape[0])


class HybridRoutedPrompt(PromptBasedMethod):
    """L2P-family method with a hybrid dense+sparse task router (see module docstring)."""

    name = "hrp"

    def _build_prompt_modules(self, config: dict, hidden_dim: int) -> None:
        self.router_mode = config.get("router", "hybrid")
        self.top_k = config.get("top_k", 2)
        self.lambda_key = config.get("lambda_key", 0.5)
        n_tasks = config.get("n_tasks", config.get("n_prompts", 10))
        slots_per_task = config.get("slots_per_task", 2)
        prompt_length = config.get("prompt_length", 5)
        vocab_size = int(self.model.model.config.vocab_size)
        self._register_prompt_module(
            "prompt_pool",
            HybridPromptPool(
                n_tasks=n_tasks,
                slots_per_task=slots_per_task,
                prompt_length=prompt_length,
                hidden_dim=hidden_dim,
                vocab_size=vocab_size,
                rrf_k=config.get("rrf_k", 60),
            ),
        )
        self._active_task: int = 0
        self._frozen_until: int = 0  # slots [0, _frozen_until) are owned by finished tasks
        # task_id → owning block id (identity here, but explicit for the routing metric).
        self._task_to_block: dict[int, int] = {}
        # set by train.py (results/<run> dir); None in unit tests → routing log skipped.
        self.out_dir = config.get("output_dir")

        # Block freezing via grad hooks: zero the gradient rows of already-finished
        # blocks so the inherited optimizer step leaves old tasks' prompts/keys intact.
        def _freeze_rows(grad: torch.Tensor) -> torch.Tensor:
            if self._frozen_until > 0:
                grad = grad.clone()
                grad[: self._frozen_until] = 0
            return grad

        self.prompt_pool.prompts.register_hook(_freeze_rows)
        self.prompt_pool.keys.register_hook(_freeze_rows)

    # ─── lifecycle ──────────────────────────────────────────────────────────────
    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        self._active_task = task.task_id
        self._task_to_block[task.task_id] = task.task_id

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Accumulate this task's OCR signature, then freeze its block's prompts+keys.

        Freezing the block enforces "each task owns its region": later tasks train
        only their own slots, so an old task's prompts can never be overwritten. The
        freeze is realised by a grad hook (registered in ``_build_prompt_modules``)
        that zeroes the gradient rows of all finished blocks — this works with the
        inherited ``PromptBasedMethod.train_task`` without overriding the loop.
        """
        for batch in train_loader:
            ids = batch["input_ids"]
            if torch.is_tensor(ids):
                self.prompt_pool.accumulate_signature(task.task_id, ids.to(self.device))
        self._frozen_until = (task.task_id + 1) * self.prompt_pool.slots_per_task
        sl = self.prompt_pool.block_slots(task.task_id)
        log.info(
            "hrp: task %d signature accumulated; slots [%d,%d) frozen",
            task.task_id,
            sl.start,
            sl.stop,
        )

    # ─── routing ────────────────────────────────────────────────────────────────
    def _sparse_query(self, batch: dict) -> torch.Tensor:
        idf = self.prompt_pool.idf
        return sparse_doc_vectors(
            batch["input_ids"].to(self.device), self.prompt_pool.vocab_size, idf
        )

    def _select_prompts(
        self, query: torch.Tensor, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Task-pinned at TRAIN time, freely routed at EVAL time.

        During training we pin every document to the *active* task's block (DualPrompt
        recipe): this is what actually trains the router — the block's prompts learn the
        task, and its dense key is pulled toward the task's queries so that at eval the
        router can recognise the task. If we routed freely while the keys are still
        random, the active task's prompts would rarely be selected and never learn
        (the cold-start collapse that gave AA~18). At eval (``model.eval()``) we use the
        learned hybrid router and *measure* whether it routes each doc to the right
        block — that routing hit-rate is the method's claim.
        """
        sparse_q = self._sparse_query(batch)
        if self.model.training:
            slot_idx = self._active_block_slots(query.shape[0])
            # Still pull the active block's dense keys toward the query (router training).
            sel_keys = F.normalize(self.prompt_pool.keys[slot_idx], dim=-1)  # (B, S, D)
            q = F.normalize(query, dim=-1).unsqueeze(1)  # (B, 1, D)
            key_pull = (1.0 - (sel_keys * q).sum(-1)).mean()
        else:
            slot_idx, key_pull = self.prompt_pool.route(
                query, sparse_q, self.top_k, self.router_mode
            )
        prompt_embeds = self.prompt_pool.gather_prompts(slot_idx)
        return prompt_embeds, self.lambda_key * key_pull

    def _active_block_slots(self, batch_size: int) -> torch.Tensor:
        """(B, slots_per_task) slot ids for the active task's block, one row per doc."""
        sl = self.prompt_pool.block_slots(self._active_task)
        row = torch.arange(sl.start, sl.stop, device=self.device)
        return row.unsqueeze(0).expand(batch_size, -1)

    # ─── evaluate (inherited prediction + routing-accuracy log) ──────────────────
    def evaluate(self, eval_loaders: dict[int, DataLoader]) -> dict[int, EvalMetrics]:
        self.model.eval()
        results: dict[int, EvalMetrics] = {}
        id_to_label = getattr(self.model, "id_to_label", None) or {
            i: str(i) for i in range(self.model.model.config.num_labels)
        }
        n_blocks = self.prompt_pool.n_tasks
        confusion: dict[int, list[int]] = {}  # true_task → [routed-block counts]
        hits: dict[int, int] = {}
        totals: dict[int, int] = {}
        with torch.no_grad():
            for tid, loader in eval_loaders.items():
                confusion[tid] = [0] * n_blocks
                hits[tid] = 0
                totals[tid] = 0
                all_preds, all_labels = [], []
                for batch in loader:
                    batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                    query = self._query(batch)
                    sparse_q = self._sparse_query(batch)
                    slot_idx, _ = self.prompt_pool.route(
                        query, sparse_q, self.top_k, self.router_mode
                    )
                    # Routing accuracy: top-1 routed block vs the true task's block.
                    top1_block = self.prompt_pool.slot_to_block(slot_idx[:, 0])
                    true_block = self._task_to_block.get(tid, tid)
                    for b in top1_block.tolist():
                        if 0 <= b < n_blocks:
                            confusion[tid][b] += 1
                    hits[tid] += int((top1_block == true_block).sum().item())
                    totals[tid] += top1_block.shape[0]

                    prompt_embeds = self.prompt_pool.gather_prompts(slot_idx)
                    logits = self.model.forward_with_prompts(
                        input_ids=batch["input_ids"],
                        bbox=batch["bbox"],
                        pixel_values=batch.get("pixel_values"),
                        prompt_embeds=prompt_embeds,
                        attention_mask=batch.get("attention_mask"),
                    )
                    preds = logits.argmax(dim=-1)
                    labels = batch["labels"][:, : logits.shape[1]]
                    mask = labels != -100
                    all_preds.extend(preds[mask].cpu().tolist())
                    all_labels.extend(labels[mask].cpu().tolist())
                metrics = compute_token_f1(all_preds, all_labels, id_to_label)
                results[tid] = EvalMetrics(
                    task_id=tid,
                    f1=metrics["f1"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    n_samples=len(all_labels),
                )
        self._write_routing_log(hits, totals, confusion)
        return results

    def _write_routing_log(
        self, hits: dict[int, int], totals: dict[int, int], confusion: dict[int, list[int]]
    ) -> None:
        """Write results/<run>/routing.json. Never fails a run if out_dir is unset."""
        per_task = {t: (hits[t] / totals[t] if totals[t] else float("nan")) for t in totals}
        tot = sum(totals.values())
        overall = sum(hits.values()) / tot if tot else float("nan")
        payload = {
            "router": self.router_mode,
            "overall_hit_rate": overall,
            "per_task_hit_rate": per_task,
            "confusion": confusion,
            "slots_per_task": self.prompt_pool.slots_per_task,
        }
        log.info("hrp routing[%s]: overall hit-rate=%.3f", self.router_mode, overall)
        if not self.out_dir:
            return
        try:
            from pathlib import Path

            out = Path(self.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "routing.json", "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:  # never fail a run over a diagnostic write
            log.warning("hrp: routing.json write skipped: %s", e)
