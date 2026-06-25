"""DocMERGE — diagnosis-guided head-merging + drift-immune lexical memory.

A continual-learning method for document information extraction that spends its entire
anti-forgetting budget on the one place the thesis localizes forgetting: the classifier
**head** (Fisher-weighted displacement ~1e-4 in the head vs ~1e-8 in late backbone
layers, ≈4 orders of magnitude; corroborated across BERT and cross-lingual XFUND). The
backbone is frozen. Two head-targeted mechanisms, exposed as a single ``consolidate``
ablation axis:

    * **head merge** (the positive-BWT engine). Each task's head is fine-tuned from a
      shared base in the same frozen-backbone loss basin (Linear Mode Connectivity), so
      the per-task head deltas can be MERGED into one head jointly good for all tasks —
      the model-soup effect. ``merge_rule ∈ {plain, ties, fisher}`` (see
      ``doccl.methods.head_merge``). This is what makes old tasks *improve* (positive
      BWT), which frozen memory slots alone cannot.

    * **drift-immune lexical memory** (the sharpness engine). A task-partitioned,
      write-once-frozen prompt pool (reused verbatim from HRP, ``HybridPromptPool``)
      addressed by a sparse BM25/TF-IDF signature over the OCR token stream
      (``batch["input_ids"]``) — a signal that never passes through the backbone, so it
      cannot drift (HRP measured 0.92 vs 0.29 routing hit-rate for lexical vs a
      backbone-hidden-state query). At eval the routed slot's prompts sharpen the
      (merged) head's prediction for that task.

``consolidate ∈ {merge, memory, both}`` selects which mechanism(s) are active, so the
"does each piece earn its keep?" ablation is ONE method toggled by config:

    * ``merge``  — head merge only; no memory read (prompts are zero-length / unused).
    * ``memory`` — lexical memory only; head trains naively (no merge) — this is HRP
                   without head-replay, i.e. the routing-only baseline.
    * ``both``   — the full method.

The method also re-validates its own premise every run: after each task it measures the
head-vs-backbone Fisher-weighted displacement and writes ``results/<run>/diag.json``, so
the head-locus claim is confirmed *on this method*, not inherited from the pilot.

Inherits the frozen-backbone train/eval loop, query extraction, prompt injection, label
truncation and prompt-aware early stopping from ``PromptBasedMethod`` (the memory path)
and the routing log + pool from HRP.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812 — canonical torch alias (repo-wide)
from torch.utils.data import DataLoader

from doccl.eval.fisher import (
    empirical_fisher_diagonal,
    fisher_weighted_displacement,
    snapshot_params,
)
from doccl.eval.metrics import compute_token_f1
from doccl.methods.head_merge import merge_head_deltas
from doccl.methods.hybrid_routed_prompt import HybridPromptPool, sparse_doc_vectors
from doccl.methods.prompt_base import PromptBasedMethod
from doccl.types import EvalMetrics, TaskInfo

log = logging.getLogger(__name__)

__all__ = ["DocMerge"]


class DocMerge(PromptBasedMethod):
    """Head-merging + drift-immune lexical memory (see module docstring)."""

    name = "doc_merge"

    # ─── construction ────────────────────────────────────────────────────────────
    def _build_prompt_modules(self, config: dict, hidden_dim: int) -> None:
        self.consolidate = config.get("consolidate", "both")  # merge | memory | both
        if self.consolidate not in ("merge", "memory", "both"):
            raise ValueError(f"consolidate must be merge|memory|both, got {self.consolidate!r}")
        self.merge_rule = config.get("merge_rule", "fisher")
        self.ties_density = float(config.get("ties_density", 0.2))
        self.headspace_constraint = bool(config.get("headspace_constraint", False))
        self.router_mode = config.get("router", "sparse")  # lexical address by default
        self.top_k = config.get("top_k", 2)
        self.lambda_key = config.get("lambda_key", 0.5)
        self.fisher_n_samples = int(config.get("fisher_n_samples", 64))
        self.out_dir = config.get("output_dir")

        self._use_memory = self.consolidate in ("memory", "both")
        self._use_merge = self.consolidate in ("merge", "both")

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
        self._frozen_until: int = 0
        self._task_to_block: dict[int, int] = {}

        # Head-merge state: the shared base θ_base (snapshot at before_task of the FIRST
        # task) and one delta Δ_t = θ_t − θ_base per finished task, plus a per-task head
        # Fisher mass for the fisher merge rule. Heads grow across CIL boundaries, so
        # deltas are stored padded-on-read by the merge helper.
        self._head_base: dict[str, torch.Tensor] | None = None
        self._head_deltas: list[dict[str, torch.Tensor]] = []
        self._head_fisher_mass: list[float] = []
        # Head-locus diagnostic per task: {task_id: {group: displacement}}.
        self._diag: dict[int, dict[str, float]] = {}
        # Paired t-1 snapshot for the head-locus diagnostic (same moment → same width).
        self._fisher_old: dict[str, torch.Tensor] | None = None
        self._params_old: dict[str, torch.Tensor] | None = None

        # Each task owns its slots: zero the gradient rows of finished blocks so the
        # inherited optimizer leaves old tasks' prompts/keys intact (HRP recipe).
        def _freeze_rows(grad: torch.Tensor) -> torch.Tensor:
            if self._frozen_until > 0:
                grad = grad.clone()
                grad[: self._frozen_until] = 0
            return grad

        self.prompt_pool.prompts.register_hook(_freeze_rows)
        self.prompt_pool.keys.register_hook(_freeze_rows)

    # ─── head tensor access ───────────────────────────────────────────────────────
    @property
    def _head(self) -> torch.nn.Linear:
        return self.model.model.classifier

    def _head_state(self) -> dict[str, torch.Tensor]:
        """Detached CPU copy of the head's weight+bias (the merge unit)."""
        return {
            "weight": self._head.weight.detach().clone().cpu(),
            "bias": self._head.bias.detach().clone().cpu(),
        }

    @torch.no_grad()
    def _load_head_state(self, state: dict[str, torch.Tensor]) -> None:
        """Write a (possibly grown) weight+bias back into the live head in place."""
        dev = self._head.weight.device
        n = self._head.weight.shape[0]
        # Merged tensors are at least as wide as the current head (deltas were padded to
        # the widest); crop to the current head width to be safe.
        self._head.weight.copy_(state["weight"][:n].to(dev))
        self._head.bias.copy_(state["bias"][:n].to(dev))

    # ─── lifecycle ────────────────────────────────────────────────────────────────
    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        self._active_task = task.task_id
        self._task_to_block[task.task_id] = task.task_id
        if self._use_merge and self._head_base is None:
            # The shared base all task-heads are merged around (LMC anchor). Captured at
            # the first task, after any head expansion the loop performed for this task.
            self._head_base = self._head_state()

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        # 1. Freeze this task's lexical signature + its prompt block (memory write-once).
        if self._use_memory:
            for batch in train_loader:
                ids = batch["input_ids"]
                if torch.is_tensor(ids):
                    self.prompt_pool.accumulate_signature(task.task_id, ids.to(self.device))
            self._frozen_until = (task.task_id + 1) * self.prompt_pool.slots_per_task

        # 2. Head merge. Compute the head Fisher on THIS task (drives the fisher rule and
        # the head-locus diagnostic), store the task's delta, re-merge, load it back.
        if self._use_merge:
            fisher = self._head_task_fisher(train_loader)
            self._store_head_delta(fisher)
            self._merge_and_load_head()

        # 3. Head-locus diagnostic: how much did the OLD task's important head params move
        # vs backbone? Written to diag.json (re-validates the premise on this method).
        self._update_diagnostic(task, train_loader)

        log.info(
            "doc_merge: task %d done [consolidate=%s rule=%s]; head-deltas=%d frozen_until=%d",
            task.task_id,
            self.consolidate,
            self.merge_rule,
            len(self._head_deltas),
            self._frozen_until,
        )

    # ─── head merge internals ─────────────────────────────────────────────────────
    @torch.no_grad()
    def _store_head_delta(self, fisher_mass: float) -> None:
        """Append Δ_t = θ_t − θ_base for this task (padded-on-read by the merge helper)."""
        cur = self._head_state()
        base = self._head_base
        delta = {}
        for k in ("weight", "bias"):
            b = base[k]
            c = cur[k]
            if c.shape != b.shape:  # head grew since base; pad base with zeros to width
                pad = torch.zeros_like(c)
                crop = tuple(slice(0, s) for s in b.shape)
                pad[crop] = b
                b = pad
            delta[k] = c - b
        self._head_deltas.append(delta)
        self._head_fisher_mass.append(fisher_mass)

    @torch.no_grad()
    def _merge_and_load_head(self) -> None:
        """Merge all stored head deltas onto the base and load into the live head."""
        weights = self._head_fisher_mass if self.merge_rule == "fisher" else None
        merged = {}
        for k in ("weight", "bias"):
            base = self._head_base[k]
            merged_delta = merge_head_deltas(
                [d[k] for d in self._head_deltas],
                rule=self.merge_rule,
                weights=weights,
                density=self.ties_density,
            )
            # Pad base to the merged-delta width (CIL growth) before adding.
            if base.shape != merged_delta.shape:
                pad = torch.zeros_like(merged_delta)
                crop = tuple(slice(0, s) for s in base.shape)
                pad[crop] = base
                base = pad
            merged[k] = base + merged_delta
        self._load_head_state(merged)

    def _head_task_fisher(self, loader: DataLoader) -> float:
        """Total head Fisher mass on this task (for the fisher merge weight).

        Reuses ``empirical_fisher_diagonal`` (one backward per doc — conservative head
        numbers) and sums the diagonal over classifier params. Cheap: a few dozen docs.
        """
        fisher = empirical_fisher_diagonal(
            self.model, loader, n_samples=self.fisher_n_samples, device=self.device
        )
        mass = 0.0
        for name, t in fisher.items():
            if "classifier" in name:
                mass += float(t.sum())
        return mass

    # ─── head-locus diagnostic (diag.json) ─────────────────────────────────────────
    def _update_diagnostic(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Per-task head-vs-backbone Fisher-weighted displacement → diag.json.

        At each task boundary t we hold a PAIRED snapshot of the previous task t-1
        captured at the *same* moment — its Fisher (``_fisher_old``) and its parameters
        (``_params_old``), so both share the t-1 head width (critical: the CIL head grows
        between t-1 and t, so a Fisher and a param snapshot taken at different points
        would mismatch in width). Displacement of t-1's important params over training t
        localizes forgetting; a head ≫ backbone ratio re-confirms the thesis premise on
        this method. Defined from task 1 on. After computing, we re-pair on task t.
        """
        if task.task_id > 0 and self._fisher_old is not None and self._params_old is not None:
            params_after = snapshot_params(self.model)
            disp = fisher_weighted_displacement(
                self._fisher_old,
                self._params_old,
                params_after,
                self.model.param_groups_by_depth,
                self.model,
                reduction="mean",
            )
            self._diag[task.task_id] = disp
            self._write_diag_log()
        # (Re)pair the snapshot for THIS task — taken together so widths agree.
        self._params_old = snapshot_params(self.model)
        self._fisher_old = empirical_fisher_diagonal(
            self.model, train_loader, n_samples=self.fisher_n_samples, device=self.device
        )

    def _write_diag_log(self) -> None:
        """Write results/<run>/diag.json. Never fails a run if out_dir is unset."""
        # Summarise the head:backbone displacement ratio per measured boundary.
        summary = {}
        for tid, groups in self._diag.items():
            head = groups.get("head", float("nan"))
            backbone = max(
                (groups.get(g, 0.0) for g in ("input", "early", "mid", "late")), default=0.0
            )
            ratio = head / backbone if backbone > 0 else float("inf")
            summary[str(tid)] = {"groups": groups, "head_over_backbone": ratio}
        if not self.out_dir:
            return
        try:
            out = Path(self.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "diag.json", "w") as f:
                json.dump({"per_boundary": summary, "consolidate": self.consolidate}, f, indent=2)
        except Exception as e:  # never fail a run over a diagnostic write
            log.warning("doc_merge: diag.json write skipped: %s", e)

    # ─── prompt selection (memory read) ────────────────────────────────────────────
    def _sparse_query(self, batch: dict) -> torch.Tensor:
        idf = self.prompt_pool.idf
        return sparse_doc_vectors(
            batch["input_ids"].to(self.device), self.prompt_pool.vocab_size, idf
        )

    def _active_block_slots(self, batch_size: int) -> torch.Tensor:
        sl = self.prompt_pool.block_slots(self._active_task)
        row = torch.arange(sl.start, sl.stop, device=self.device)
        return row.unsqueeze(0).expand(batch_size, -1)

    def _empty_prompts(self, batch_size: int) -> torch.Tensor:
        """A zero-width prompt tensor — used when memory is off (merge-only ablation)."""
        return torch.zeros(batch_size, 0, self.hidden_dim, device=self.device)

    def _select_prompts(
        self, query: torch.Tensor, batch: dict
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Memory read: task-pinned at TRAIN, freely routed at EVAL (HRP recipe).

        With ``consolidate == merge`` the memory is off → an empty prompt set and a zero
        aux loss, so the inherited loop trains/evaluates the head with no prompt
        modulation (a clean merge-only ablation).
        """
        if not self._use_memory:
            return self._empty_prompts(query.shape[0]), torch.zeros((), device=self.device)
        sparse_q = self._sparse_query(batch)
        if self.model.training:
            slot_idx = self._active_block_slots(query.shape[0])
            sel_keys = F.normalize(self.prompt_pool.keys[slot_idx], dim=-1)
            q = F.normalize(query, dim=-1).unsqueeze(1)
            key_pull = (1.0 - (sel_keys * q).sum(-1)).mean()
        else:
            slot_idx, key_pull = self.prompt_pool.route(
                query, sparse_q, self.top_k, self.router_mode
            )
        prompt_embeds = self.prompt_pool.gather_prompts(slot_idx)
        return prompt_embeds, self.lambda_key * key_pull

    # ─── evaluate (inherited prediction + routing log when memory is on) ────────────
    def evaluate(self, eval_loaders: dict[int, DataLoader]) -> dict[int, EvalMetrics]:
        if not self._use_memory:
            # Merge-only: no routing to measure; the inherited prompt loop with empty
            # prompts evaluates the merged head directly.
            return super().evaluate(eval_loaders)

        self.model.eval()
        results: dict[int, EvalMetrics] = {}
        id_to_label = getattr(self.model, "id_to_label", None) or {
            i: str(i) for i in range(self.model.model.config.num_labels)
        }
        n_blocks = self.prompt_pool.n_tasks
        confusion: dict[int, list[int]] = {}
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
        """Write results/<run>/routing.json (mirrors HRP). Never fails a run."""
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
        log.info("doc_merge routing[%s]: overall hit-rate=%.3f", self.router_mode, overall)
        if not self.out_dir:
            return
        try:
            out = Path(self.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            with open(out / "routing.json", "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            log.warning("doc_merge: routing.json write skipped: %s", e)
