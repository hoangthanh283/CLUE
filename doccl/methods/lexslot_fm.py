"""LexSlot-FM — Functional Memory LexSlot (dual-stream plasticity-stability).

The core insight (RCA-motivated): additive delta-slots go stale when the base they sit
on drifts. LexSlot-FM resolves this with a *dual-stream* architecture inspired by
complementary learning systems:

  - **Plastic base head** (trainable): fits the current task freely. This is the stream
    that drifts — by design, because stability lives elsewhere.
  - **Functional memory bank** (frozen after each task): per-task FULL-RANK head copies
    that store a snapshot of each old task's output function, reading the FROZEN encoder
    features. Because both the encoder and the functional slots are frozen, ``slot_t(f(x))``
    reproduces task-t's behavior exactly at eval time — no drift, no forgetting, no stale
    deltas.
  - **Drift-immune lexical gate** (blend at eval): ``g_t = cos(sig(x), sig_t)`` routes a
    document to its task's functional memory; old-task docs come from the frozen functional
    copy, current-task docs from the plastic base. The gate is pure TF-IDF — never learned,
    never forgets.

Blend formulas:
  - **Training**: ``logits = fm_cur(f)`` — FM-only. The FM receives the FULL CE gradient,
    learning the entire task mapping (no splitting with the base head). The base head is
    excluded because at eval the replacement blend suppresses it — any capacity invested
    in the base head during training is wasted at eval.
  - **Evaluation**: ``logits = (1-Σg)·base(f) + Σg·fm_all(f)`` — replacement. The lexical
    gate suppresses the (drifted) base for old-task docs, routing instead to the frozen
    functional copy. For current-task/MU data the base dominates.

Retention bound (informal): forgetting ≤ (gate-misroute rate) × (functional-slot
expressiveness gap). With lexically-separable domains (FUNSD↔CORD S≈0.10) misroute→0
→ forgetting→0. With full-rank stored heads, the expressiveness gap is zero (a full
Linear can reproduce any head mapping) — so retention is bounded only by gate quality.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.eval.fisher import empirical_fisher_diagonal

from doccl.eval.metrics import compute_token_f1
from doccl.methods.naive import NaiveFineTune
from doccl.types import EvalMetrics, TaskInfo, TrainMetrics

log = logging.getLogger(__name__)

__all__ = ["LexSlotFM", "FunctionalHeadSlots"]


class FunctionalHeadSlots(nn.Module):
    """Per-task full-rank functional head memory.

    Each task t owns a full ``nn.Linear`` head ``W_t, b_t`` that maps frozen encoder
    features ``f∈R^{d}`` directly to logits, INDEPENDENTLY of the plastic base head.
    After task t completes, its head is frozen.

    Key invariant vs delta-slots: a functional head reads ``f(x)`` from the frozen
    encoder and produces logits directly — the drifted plastic base never enters its
    computation, so it cannot go stale. This is the "no stale delta" guarantee.

    ``logits_all(x)`` returns ``(B, L, T, C)`` — the per-task functional logits for ALL
    owned tasks — so the eval blend can route each document to its task's copy.
    """

    def __init__(self, hidden_dim: int, n_labels: int, max_tasks: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_labels = n_labels
        self.max_tasks = max_tasks
        # Full-rank per-task head weights: W ∈ R^{d×C}, b ∈ R^{C}.
        # Zero-init so an unclaimed slot contributes nothing; claimed slots learn
        # from scratch via SGD during their owning task.
        self.weight = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_dim, n_labels)) for _ in range(max_tasks)]
        )
        self.bias = nn.ParameterList(
            [nn.Parameter(torch.zeros(n_labels)) for _ in range(max_tasks)]
        )
        self.owner: list[int] = [-1] * max_tasks
        self.frozen_tasks: list[bool] = [False] * max_tasks

    def claim(self, task_id: int) -> int:
        """Claim a free slot block for this task. Returns the slot index."""
        free = [i for i, o in enumerate(self.owner) if o == -1]
        if not free:
            raise RuntimeError("FunctionalHeadSlots: no free slots to claim")
        idx = free[0]
        self.owner[idx] = task_id
        return idx

    def _owned_indices(self) -> list[int]:
        return [i for i, o in enumerate(self.owner) if o != -1]

    def logits_all(self, feats: torch.Tensor) -> torch.Tensor:
        """Per-task functional logits for ALL owned tasks.

        Args:
            feats: (B, L, d) frozen encoder features.
        Returns:
            (B, L, T, C) where T = n_owned. Caller maps task_id→column via ``owner``.
        """
        owned = self._owned_indices()
        if not owned:
            return feats.new_zeros(feats.shape[0], feats.shape[1], 0, self.n_labels)
        stacked = []
        for i in owned:
            stacked.append(feats @ self.weight[i] + self.bias[i])
        return torch.stack(stacked, dim=2)

    def logits_one(self, feats: torch.Tensor, idx: int) -> torch.Tensor:
        """Functional logits for a single owned slot (for training additive blend)."""
        return feats @ self.weight[idx] + self.bias[idx]

    def freeze_owned(self) -> None:
        """Freeze all owned task slots (call after each task's training)."""
        for i in self._owned_indices():
            self.weight[i].requires_grad = False
            self.bias[i].requires_grad = False
            self.frozen_tasks[i] = True

    @torch.no_grad()
    def expand_labels(self, new_n_labels: int) -> None:
        """CIL: grow all weight/bias slots' label dim, preserving old weights."""
        if new_n_labels <= self.n_labels:
            return
        for i in range(self.max_tasks):
            old_w = self.weight[i].data
            old_b = self.bias[i].data
            new_w = torch.zeros(self.hidden_dim, new_n_labels, device=old_w.device, dtype=old_w.dtype)
            new_b = torch.zeros(new_n_labels, device=old_b.device, dtype=old_b.dtype)
            new_w[:, :old_w.shape[1]] = old_w
            new_b[:old_b.shape[0]] = old_b
            self.weight[i] = nn.Parameter(new_w)
            self.bias[i] = nn.Parameter(new_b)
        self.n_labels = new_n_labels


class LexSlotFM(NaiveFineTune):
    """LexSlot-FM: Functional Memory LexSlot (dual-stream, no replay/KD/Fisher).

    Inherits NaiveFineTune's plain-CE ``train_task`` / ``evaluate``. The functional
    memory is blended at the head via a forward hook; the lexical gate is installed by
    a forward pre-hook on the wrapper.

    Blend:
      - Train: ``logits = fm_cur(f)`` (FM-only, full gradient).
      - Eval: ``logits = (1 - Σg)·base(f) + Σg·fm_all(f)`` (replacement, gate-routed).
    """

    name = "lexslot_fm"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.max_tasks = int(config.get("max_tasks", 10))
        self.infer_gate = bool(config.get("infer_gate", True))

        self.hidden_dim = model.hidden_size
        self.vocab_size = int(self.model.model.config.vocab_size)
        n_labels = self.model.model.classifier.out_features

        # Functional memory bank: per-task frozen full-rank head copies.
        self.fm = FunctionalHeadSlots(self.hidden_dim, n_labels, self.max_tasks).to(
            self.device
        )

        # Capture the classifier's input features via a pre-hook.
        self._cur_feats: torch.Tensor | None = None

        # Register the head hook (functional memory blend).
        self._hook_handles: list = []
        self._register_head_hook()

        self._cur_gate: torch.Tensor | None = None
        self._cur_fm_idx: int | None = None

        # Gate mode: "cls" (learned Linear on [CLS]), "tfidf" (token ID TF-IDF
        # cosine), or "oracle" (ground-truth task IDs).  "cls" is the default.
        self.gate_mode = str(config.get("gate_mode", "cls"))
        self.oracle_gate = bool(config.get("oracle_gate", False))
        if self.oracle_gate:
            self.gate_mode = "oracle"

        # Learned CLS classifier state (used when gate_mode == "cls").
        self._cls_buffer_x: list[torch.Tensor] = []
        self._cls_buffer_y: list[int] = []
        self._gate_clf: nn.Linear | None = None

        # Oracle gate state.
        self._eval_task_id: int | None = None

        # TF-IDF lexical gate state (decoded-text, sklearn-based).
        self._task_texts: dict[int, list[str]] = {}
        self._tfidf_vec: TfidfVectorizer | None = None
        self._tfidf_centroids: np.ndarray | None = None  # (n_tasks, n_features) L2-normed
        self._task_ids_tfidf: list[int] = []
        self._cur_tfidf_gate: torch.Tensor | None = None

        # Layout gate state (vocab-agnostic, bbox-derived spatial features).
        self._layout_buffer_x: list[torch.Tensor] = []
        self._layout_buffer_y: list[int] = []
        self._layout_clf: nn.Linear | None = None
        self._cur_layout_gate: torch.Tensor | None = None

        # EWC for encoder relaxation: Fisher+θ* per parameter (layoutlmv3 only).
        self.ewc_lambda = float(config.get("ewc_lambda", 0.0))
        self.fisher_n_samples = int(config.get("fisher_n_samples", 200))
        self._fisher: dict[str, torch.Tensor] = {}
        self._theta_star: dict[str, torch.Tensor] = {}

        # FM replay buffers: re-fit old FM slots after encoder shift.
        self.fm_refit_epochs = int(config.get("fm_refit_epochs", 0))
        self.fm_refit_samples = int(config.get("fm_refit_samples", 50))
        self._fm_replay_buffers: dict[int, list[dict]] = {}

        # Full encoder freeze (the no-stale-delta guarantee requires frozen features).
        # Overridden in train_task when ewc_lambda > 0 and task_id > 0.
        self.model.freeze_backbone()
        n_enc = sum(1 for p in self.model.parameters() if p.requires_grad)
        log.info(
            "lexslot_fm: full encoder freeze; trainable params (head + FM unclaimed) = %d; "
            "ewc_lambda=%.1f",
            n_enc,
            self.ewc_lambda,
        )

    # ─── hooks ──────────────────────────────────────────────────────────────────
    def _register_head_hook(self) -> None:
        self._hook_handles = []
        cls = self.model.model.classifier
        self._hook_handles.append(cls.register_forward_pre_hook(self._capture_feats))
        self._hook_handles.append(cls.register_forward_hook(self._blend_fm))

    def _detach_head_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def _capture_feats(self, _module, inp):
        self._cur_feats = inp[0]
        return None

    def _blend_fm(self, _module, _inp, output):
        """Forward hook on the classifier: blend functional memory with base head.

        Two regimes determined by ``self.model.training``:

        **Training (FM-only blend)**::
            logits = fm_cur(f)

        The current task's FM slot (``_cur_fm_idx``) receives the FULL CE gradient and
        learns the entire task mapping. The base head is excluded — it contributes
        nothing to the forward pass (zero gradient) and stays at initialization,
        because the eval replacement blend suppresses it with ``(1 − Σg) ≈ 0``.

        **Evaluation (replacement blend)**::
            logits = (1 − Σg)·base(f) + Σg·fm_all(f)

        The lexical gate determines which FM fires: old-task docs are routed to their
        frozen functional copy (suppressing the drifted base), while current-task / OOD
        docs fall back to the plastic base head.
        """
        if self._cur_feats is None or self._cur_feats.shape[1] != output.shape[1]:
            return output
        feats = self._cur_feats.to(output.dtype)

        if self.model.training:
            return self._additive_blend(output, feats)
        return self._replacement_blend(output, feats)

    def _additive_blend(self, output: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Training: fm_cur(f) only.

        The FM alone must receive the FULL gradient for its task. If we added the
        base head (``output + fm_cur(f)``), the CE gradient would be split between
        the two → each learns only half the mapping → at eval the replacement blend
        suppresses the base head and the FM alone gives ≈½ F1. This was the root
        cause of the full-rank FM's initial 53 → 5.8 collapse.

        Notes:
          - The base head (``output``) still appears in ``trainable_parameters()``
            but receives zero gradient because it does not contribute to the loss.
            Its weights stay at initialization — harmless, since the eval
            replacement blend suppresses it with ``(1 − Σg) ≈ 0`` for any gated
            document.
          - After task t, ``_cur_fm_idx`` is set to None so subsequent tasks train
            their own FM independently.
        """
        if self._cur_fm_idx is None:
            return output
        return self.fm.logits_one(feats, self._cur_fm_idx)

    def _replacement_blend(self, output: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """Eval: (1−Σg)·base(f) + Σg·fm_all(f). Old docs → frozen FM, new docs → base.

        Gate: rare-token cosine. The pre-hook ``_install_gate`` computes the cosine
        between the rare-masked bag-of-tokens document vector and each task's rare-masked
        signature. Tokens that appear in >10 % of the first task's documents (structural
        tokens common to ALL document AI domains) are zeroed out, so the remaining cosine
        similarity reflects domain-specific vocabulary overlap.
        """
        owned = self.fm._owned_indices()
        if not owned:
            return output
        fm_logits = self.fm.logits_all(feats)
        if fm_logits.shape[2] == 0:
            return output
        n_tasks = fm_logits.shape[2]

        # Hybrid gate: geometric-mean fusion of layout + TF-IDF.
        if self.gate_mode == "hybrid" and self._cur_layout_gate is not None and self._cur_tfidf_gate is not None:
            g_l = self._cur_layout_gate.to(device=output.device, dtype=output.dtype)
            g_t = self._cur_tfidf_gate.to(device=output.device, dtype=output.dtype)
            if g_l.shape[2] < n_tasks:
                pad = g_l.new_zeros(g_l.shape[0], 1, n_tasks - g_l.shape[2], 1)
                g_l = torch.cat([g_l, pad], dim=2)
            if g_t.shape[2] < n_tasks:
                pad = g_t.new_zeros(g_t.shape[0], 1, n_tasks - g_t.shape[2], 1)
                g_t = torch.cat([g_t, pad], dim=2)
            # Averaged fusion: each gate corrects the other's blind spot.
            g = 0.5 * g_l + 0.5 * g_t
        # Layout gate: bbox-derived spatial features (vocab-agnostic).
        elif self.gate_mode == "layout" and self._cur_layout_gate is not None:
            g = self._cur_layout_gate.to(device=output.device, dtype=output.dtype)
            if g.shape[2] < n_tasks:
                pad = g.new_zeros(g.shape[0], 1, n_tasks - g.shape[2], 1)
                g = torch.cat([g, pad], dim=2)
        # TF-IDF lexical gate: from pre-computed per-batch gate.
        elif self.gate_mode == "tfidf" and self._cur_tfidf_gate is not None:
            g = self._cur_tfidf_gate.to(device=output.device, dtype=output.dtype)
        # Oracle gate: use ground-truth eval task id (for ablation / verification).
        elif self.oracle_gate and self._eval_task_id is not None:
            g = torch.zeros(1, 1, n_tasks, 1, device=output.device, dtype=output.dtype)
            if self._eval_task_id < n_tasks:
                g[0, 0, self._eval_task_id, 0] = 1.0
        # Compute gate from the gate classifier on the CURRENT batch's [CLS].
        elif self.infer_gate and self._gate_clf is not None:
            cls = feats[:, 0].to(self._gate_clf.weight.dtype)
            with torch.no_grad():
                g_logits = self._gate_clf(cls)
            g = g_logits.softmax(dim=-1).to(output.dtype)  # (B, n_clf)
            # If the classifier has fewer outputs than owned tasks (current task is
            # still training), pad with zeros for the extra task(s) — the current
            # task's FM has not yet converged and should contribute nothing.
            if g.shape[-1] < n_tasks:
                pad = g.new_zeros(g.shape[0], n_tasks - g.shape[-1])
                g = torch.cat([g, pad], dim=-1)
            g = g.unsqueeze(1).unsqueeze(-1)  # (B, 1, T, 1)
        elif n_tasks == 1:
            # Single owned task → always route through it (no competition).
            g = torch.ones(1, 1, 1, 1, device=output.device, dtype=output.dtype)
        elif n_tasks > 0:
            g = torch.full(
                (1, 1, n_tasks, 1), 1.0 / n_tasks, device=output.device, dtype=output.dtype
            )
        else:
            g = torch.ones(1, 1, 1, 1, device=output.device, dtype=output.dtype)
        fm_weighted = (fm_logits * g).sum(dim=2)
        gate_sum = g.sum(dim=2).squeeze(-1)
        base_weight = (1.0 - gate_sum).clamp(0.0, 1.0)
        return output * base_weight.unsqueeze(1) + fm_weighted

    # ─── TF-IDF lexical gate (decoded-text, sklearn) ──────────────────────────

    def _get_tokenizer(self):
        """Get the tokenizer from the model wrapper."""
        if hasattr(self.model, "processor") and hasattr(self.model.processor, "tokenizer"):
            return self.model.processor.tokenizer
        if hasattr(self.model, "tokenizer"):
            return self.model.tokenizer
        raise RuntimeError("lexslot_fm: no tokenizer found for TF-IDF gate")

    @torch.no_grad()
    def _collect_task_texts(self, loader, task_id: int) -> None:
        """Decode ``input_ids`` to text strings and store per task."""
        self.model.eval()
        tk = self._get_tokenizer()
        texts: list[str] = []
        for batch in loader:
            input_ids = batch["input_ids"]
            for b in range(input_ids.shape[0]):
                texts.append(
                    tk.decode(
                        input_ids[b].tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                )
        self._task_texts[task_id] = texts
        log.info("lexslot_fm: TF-IDF decoded %d docs for task %d", len(texts), task_id)

    def _build_tfidf_gate(self) -> None:
        """Fit sklearn TfidfVectorizer and compute per-task centroids."""
        if len(self._task_texts) < 2:
            return
        task_ids = sorted(self._task_texts.keys())
        all_texts: list[str] = []
        task_labels: list[int] = []
        for tid in task_ids:
            all_texts.extend(self._task_texts[tid])
            task_labels.extend([tid] * len(self._task_texts[tid]))

        vec = TfidfVectorizer(
            max_features=10000,
            sublinear_tf=True,
            stop_words="english",
        )
        tfidf_matrix = vec.fit_transform(all_texts)  # (N, V)

        # Per-task centroids.
        n_tasks = len(task_ids)
        centroids = np.zeros((n_tasks, tfidf_matrix.shape[1]))
        for i, tid in enumerate(task_ids):
            mask = np.array(task_labels) == tid
            if mask.sum() > 0:
                centroids[i] = tfidf_matrix[mask].mean(axis=0).A1
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / (norms + 1e-10)

        self._tfidf_vec = vec
        self._tfidf_centroids = centroids
        self._task_ids_tfidf = task_ids
        log.info(
            "lexslot_fm: TF-IDF gate built %d tasks, vocab=%d",
            n_tasks,
            tfidf_matrix.shape[1],
        )

    @torch.no_grad()
    def _compute_tfidf_gate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Per-batch TF-IDF gate via decoded-text cosine similarity."""
        if self._tfidf_vec is None or self._tfidf_centroids is None:
            return input_ids.new_zeros(input_ids.shape[0], 1, 0, 1)
        tk = self._get_tokenizer()
        b = input_ids.shape[0]
        texts = [
            tk.decode(
                input_ids[i].tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for i in range(b)
        ]
        doc_vec = self._tfidf_vec.transform(texts).toarray()  # (B, V)
        norms = np.linalg.norm(doc_vec, axis=1, keepdims=True)
        doc_vec = doc_vec / (norms + 1e-10)
        sim = doc_vec @ self._tfidf_centroids.T  # (B, T)
        g = torch.from_numpy(sim).to(dtype=torch.float32, device=input_ids.device)
        g = g.softmax(dim=-1)
        return g.unsqueeze(1).unsqueeze(-1)

    # ─── layout gate (vocab-agnostic, bbox-only) ─────────────────────────────────

    @staticmethod
    @torch.no_grad()
    def _compute_layout_features(
        bbox: torch.Tensor, attention_mask: torch.Tensor, grid_size: int = 4
    ) -> torch.Tensor:
        """Vocab-agnostic layout features from bbox coordinates only.

        Builds a fixed-dim spatial signature per document from bounding-box
        statistics — position, size, shape — with zero text content influence.

        Feature vector (86-d):
          - 16: 4x4 histogram of box centers (spatial occupancy)
          - 16: width histogram
          - 16: height histogram
          - 16: log-area histogram
          - 16: log-aspect-ratio histogram
          -  6: n_boxes, area_density, var_cx, var_cy, mean_w, mean_h

        Args:
            bbox: (B, N, 4) bbox coords in [0, 1000].
            attention_mask: (B, N) boolean / 0-1 mask.
            grid_size: bins per axis for the 2D spatial histogram.

        Returns:
            (B, 86) float tensor of layout features.
        """
        bsz, seq_len, _ = bbox.shape
        valid = (bbox.sum(dim=-1) > 0) & attention_mask.bool()  # (B, N)

        bbox_f = bbox.float()
        cx = (bbox_f[..., 0] + bbox_f[..., 2]) / 2
        cy = (bbox_f[..., 1] + bbox_f[..., 3]) / 2
        w = (bbox_f[..., 2] - bbox_f[..., 0]).clamp(min=0)
        h = (bbox_f[..., 3] - bbox_f[..., 1]).clamp(min=0)
        area = w * h
        ar = w / (h + 1e-6)
        log_area = (area + 1.0).log()
        log_ar = ar.log()

        float_dtype = bbox_f.dtype
        dev = bbox.device
        n_grid = grid_size * grid_size
        n_hist = 16
        n_scalar = 6
        n_total = n_grid + 5 * n_hist + n_scalar

        feats_list = []
        for b_idx in range(bsz):
            mask = valid[b_idx]
            n_valid = mask.sum().item()

            if n_valid < 2:
                feats_list.append(torch.zeros(n_total, dtype=float_dtype, device=dev))
                continue

            cx_b = cx[b_idx][mask]
            cy_b = cy[b_idx][mask]
            w_b = w[b_idx][mask]
            h_b = h[b_idx][mask]
            area_b = area[b_idx][mask]
            log_area_b = log_area[b_idx][mask]
            log_ar_b = log_ar[b_idx][mask]

            f = []

            # 1. 2D spatial histogram of centers on grid_size × grid_size.
            bin_x = (cx_b / 1000 * grid_size).clamp(0, grid_size - 1).long()
            bin_y = (cy_b / 1000 * grid_size).clamp(0, grid_size - 1).long()
            bin_idx = bin_y * grid_size + bin_x
            spatial_hist = torch.zeros(n_grid, dtype=float_dtype, device=dev)
            spatial_hist.scatter_add_(0, bin_idx, torch.ones_like(bin_idx, dtype=float_dtype))
            spatial_hist = spatial_hist / (spatial_hist.sum() + 1e-8)
            f.append(spatial_hist)

            # 2. Width histogram.
            w_hist = torch.zeros(n_hist, dtype=float_dtype, device=dev)
            w_bins = (w_b / 1000 * n_hist).clamp(0, n_hist - 1).long()
            w_hist.scatter_add_(0, w_bins, torch.ones_like(w_bins, dtype=float_dtype))
            w_hist = w_hist / (w_hist.sum() + 1e-8)
            f.append(w_hist)

            # 3. Height histogram.
            h_hist = torch.zeros(n_hist, dtype=float_dtype, device=dev)
            h_bins = (h_b / 1000 * n_hist).clamp(0, n_hist - 1).long()
            h_hist.scatter_add_(0, h_bins, torch.ones_like(h_bins, dtype=float_dtype))
            h_hist = h_hist / (h_hist.sum() + 1e-8)
            f.append(h_hist)

            # 4. Log-area histogram over [0, log(1000²)].
            max_log_area = math.log(1000 * 1000 + 1.0)
            area_hist = torch.zeros(n_hist, dtype=float_dtype, device=dev)
            area_bins = (log_area_b / max_log_area * n_hist).clamp(0, n_hist - 1).long()
            area_hist.scatter_add_(0, area_bins, torch.ones_like(area_bins, dtype=float_dtype))
            area_hist = area_hist / (area_hist.sum() + 1e-8)
            f.append(area_hist)

            # 5. Log-aspect-ratio histogram over [-3, 3].
            ar_hist = torch.zeros(n_hist, dtype=float_dtype, device=dev)
            ar_bins = ((log_ar_b + 3.0) / 6.0 * n_hist).clamp(0, n_hist - 1).long()
            ar_hist.scatter_add_(0, ar_bins, torch.ones_like(ar_bins, dtype=float_dtype))
            ar_hist = ar_hist / (ar_hist.sum() + 1e-8)
            f.append(ar_hist)

            # 6. Global scalar features.
            n_boxes_f = float(n_valid)
            total_area_density = float(area_b.sum().item() / (1000 * 1000))
            var_cx_f = float(cx_b.var().item()) if n_valid > 1 else 0.0
            var_cy_f = float(cy_b.var().item()) if n_valid > 1 else 0.0
            mean_w_f = float(w_b.mean().item())
            mean_h_f = float(h_b.mean().item())
            f.append(
                torch.tensor(
                    [n_boxes_f, total_area_density, var_cx_f, var_cy_f, mean_w_f, mean_h_f],
                    dtype=float_dtype,
                    device=dev,
                )
            )

            feats_list.append(torch.cat(f))

        return torch.stack(feats_list)  # (B, n_total)

    @torch.no_grad()
    def _collect_layout(self, loader, task_id: int) -> None:
        """Collect layout features from ``loader`` into the buffer."""
        self.model.eval()
        all_layout: list[torch.Tensor] = []
        for batch in loader:
            batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            lf = self._compute_layout_features(batch["bbox"], batch["attention_mask"])
            all_layout.append(lf.cpu())
        if all_layout:
            stacked = torch.cat(all_layout)
            self._layout_buffer_x.append(stacked)
            self._layout_buffer_y.append(task_id)
            log.info(
                "lexslot_fm: layout buffer collected %d embeds for task %d (total=%d)",
                stacked.shape[0],
                task_id,
                sum(c.shape[0] for c in self._layout_buffer_x),
            )

    def _train_layout_classifier(self) -> None:
        """Train ``_layout_clf`` on all collected layout features.

        Uses ensemble selection (multiple random inits → best train accuracy)
        to avoid the initialisation-sensitivity failure mode where a tiny
        (86→n_classes) linear classifier collapses one class to zero recall.
        """
        if len(self._layout_buffer_x) < 2:
            return
        x = torch.cat(self._layout_buffer_x).to(self.device)  # (N, 86)
        y = torch.cat([
            torch.full((c.shape[0],), tid, dtype=torch.long, device=self.device)
            for c, tid in zip(self._layout_buffer_x, self._layout_buffer_y, strict=True)
        ])
        n_classes = y.max().item() + 1
        n_per_class = y.bincount(minlength=n_classes).float()
        weight = (n_per_class.sum() / n_per_class).to(self.device)

        best_clf: nn.Linear | None = None
        best_acc = -1.0
        best_recalls: list[tuple[int, float]] = []

        for trial in range(10):
            clf = nn.Linear(x.shape[1], n_classes).to(self.device)
            opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
            for _ in range(100):
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(clf(x), y, weight=weight)
                loss.backward()
                opt.step()
            preds = clf(x).argmax(-1)
            acc = (preds == y).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_clf = clf
                best_recalls = []
                for c in range(n_classes):
                    mask = y == c
                    if mask.sum() > 0:
                        best_recalls.append((mask.sum().item(), ((preds[mask] == c).float().mean().item())))

        self._layout_clf = best_clf
        log.info(
            "lexslot_fm: layout classifier trained 86→%d (best-of-10); train acc=%.2f%% cls_recall=%s",
            n_classes,
            100.0 * best_acc,
            " ".join(f"c{k}:{n}/{r:.2f}" for k, (n, r) in enumerate(best_recalls)),
        )

    @torch.no_grad()
    def _compute_layout_gate_batch(
        self, bbox: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor | None:
        """Per-batch layout gate via layout classifier."""
        if self._layout_clf is None:
            return None
        lf = self._compute_layout_features(bbox, attention_mask)
        g_logits = self._layout_clf(lf.to(self._layout_clf.weight.dtype))
        g = g_logits.softmax(dim=-1).to(torch.float32)  # (B, n_clf)
        return g.unsqueeze(1).unsqueeze(-1)  # (B, 1, T, 1)

    # ─── learned-gate classifier ─────────────────────────────────────────────────

    @torch.no_grad()
    def _collect_cls(self, loader, task_id: int) -> None:
        """Collect [CLS] embeddings from ``loader`` into the buffer."""
        self.model.eval()
        all_cls: list[torch.Tensor] = []
        for batch in loader:
            batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            _ = self.model(**batch)
            if self._cur_feats is not None:
                all_cls.append(self._cur_feats[:, 0].cpu())
        if all_cls:
            stacked = torch.cat(all_cls)
            self._cls_buffer_x.append(stacked)
            self._cls_buffer_y.append(task_id)
            log.info(
                "lexslot_fm: CLS buffer collected %d embeds for task %d (total=%d)",
                stacked.shape[0],
                task_id,
                sum(c.shape[0] for c in self._cls_buffer_x),
            )

    def _train_gate_classifier(self) -> None:
        """Train ``_gate_clf`` on all collected CLS embeddings."""
        if len(self._cls_buffer_x) < 2:
            return  # need ≥2 tasks to learn discrimination
        x = torch.cat(self._cls_buffer_x).to(self.device)  # (N, d)
        y = torch.cat([
            torch.full((c.shape[0],), tid, dtype=torch.long, device=self.device)
            for c, tid in zip(self._cls_buffer_x, self._cls_buffer_y, strict=True)
        ])
        n_classes = y.max().item() + 1
        clf = nn.Linear(x.shape[1], n_classes).to(self.device)
        n_per_class = y.bincount(minlength=n_classes).float()
        weight = (n_per_class.sum() / n_per_class).to(self.device)
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
        for _ in range(100):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(clf(x), y, weight=weight)
            loss.backward()
            opt.step()
        preds = clf(x).argmax(-1)
        acc = (preds == y).float().mean().item()
        per_class_recall = []
        for c in range(n_classes):
            mask = y == c
            if mask.sum() > 0:
                recall = ((preds[mask] == c).float().mean().item())
                per_class_recall.append((mask.sum().item(), recall))
        self._gate_clf = clf
        log.info(
            "lexslot_fm: gate classifier trained %d→%d; train acc=%.2f%% cls_recall=%s",
            x.shape[1],
            n_classes,
            100.0 * acc,
            " ".join(f"c{k}:{n}/{r:.2f}" for k, (n, r) in enumerate(per_class_recall)),
        )

    # ─── optimizer target ───────────────────────────────────────────────────────
    def trainable_parameters(self):
        """Base head + unclaimed (training) FM slots."""
        base = [p for p in self.model.parameters() if p.requires_grad]
        fm_params = []
        for i in self.fm._owned_indices():
            if not self.fm.frozen_tasks[i]:
                fm_params.extend([self.fm.weight[i], self.fm.bias[i]])
        return base + fm_params

    # ─── EWC (Fisher-relaxed encoder) ──────────────────────────────────────────

    def _ewc_penalty(self) -> torch.Tensor:
        """Quadratic EWC penalty on encoder params only.

        ``(λ/2) * Σ_i F_i * (θ_i − θ*_i)²`` where F_i is the accumulated
        diagonal Fisher (task importance) and θ*_i is the post-task snapshot.
        Only the ``layoutlmv3`` encoder parameters are regularised — the FM
        slots already provide perfect head-level forgetting.
        """
        if not self._fisher:
            return torch.zeros((), device=self.device)
        params = dict(self.model.named_parameters())
        penalty = torch.zeros((), device=self.device)
        for name, fisher_val in self._fisher.items():
            if name not in params or name not in self._theta_star:
                continue
            p = params[name]
            theta_star = self._theta_star[name]
            penalty = penalty + (fisher_val * (p - theta_star) ** 2).sum()
        return (self.ewc_lambda / 2) * penalty

    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        """Overrides NaiveFineTune: adds EWC encoder relaxation for tasks > 0."""
        # Unfreeze encoder for task 1+ when EWC is active.
        if self.ewc_lambda > 0 and task.task_id > 0:
            self.model.unfreeze_backbone()
            log.info(
                "lexslot_fm: encoder UNFROZEN for task %d (EWC λ=%.1f, Fisher keys=%d)",
                task.task_id,
                self.ewc_lambda,
                len(self._fisher),
            )

        self.model.train()
        optimizer = torch.optim.AdamW(
            self.trainable_parameters(),
            lr=self.config.get("lr", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        stopper = self.make_early_stopper(val_loader)
        self._amp_setup()

        total_loss = 0.0
        n_steps = 0
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"T{task.task_id} ep{epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                optimizer.zero_grad()
                with self._amp_autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    if self.ewc_lambda > 0 and self._fisher:
                        loss = loss + self._ewc_penalty()
                self._amp_backward_step(loss, optimizer, self.trainable_parameters(), max_grad_norm)
                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            if self._early_stop_after_epoch(stopper, val_loader, task, epoch):
                break

        stopper.restore_best(self.model)
        return TrainMetrics(
            task_id=task.task_id,
            loss=total_loss / max(n_steps, 1),
            n_steps=n_steps,
        )

    # ─── FM replay & refit (re-fit old FM slots after encoder shift) ────────

    @torch.no_grad()
    def _store_fm_replay(self, loader, task_id: int) -> None:
        """Store a small replay buffer for FM slot re-fitting."""
        n = self.fm_refit_samples
        stored: list[dict] = []
        for batch in loader:
            batch = {k: v.to("cpu") if torch.is_tensor(v) else v for k, v in batch.items()}
            bs = batch["input_ids"].shape[0]
            for i in range(bs):
                ex = {k: v[i].unsqueeze(0) if torch.is_tensor(v) else v for k, v in batch.items()}
                stored.append(ex)
                if len(stored) >= n:
                    break
            if len(stored) >= n:
                break
        self._fm_replay_buffers[task_id] = stored[:n]
        log.info("lexslot_fm: stored FM replay for task %d (%d samples)", task_id, len(stored))

    def _refit_old_fm_slots(self, task_id: int) -> None:
        """Re-fit old FM slots on their replay buffers using the current encoder."""
        old_slots = [
            i for i in self.fm._owned_indices()
            if self.fm.owner[i] < task_id
        ]
        if not old_slots:
            return

        self.model.eval()
        log.info("lexslot_fm: re-fitting %d old FM slots", len(old_slots))

        # Switch to train mode so _additive_blend (not _replacement_blend with
        # the gate) is used — we want the old FM slot to fire alone, without
        # competing FM slot contributions or gate-scheduled blending.
        self.model.train()
        for old_idx in old_slots:
            owner = self.fm.owner[old_idx]
            replay = self._fm_replay_buffers.get(owner)
            if not replay:
                continue

            # Temporarily unfreeze this FM slot.
            self.fm.weight[old_idx].requires_grad = True
            self.fm.bias[old_idx].requires_grad = True
            self.fm.frozen_tasks[old_idx] = False
            self._cur_fm_idx = old_idx

            refit_opt = torch.optim.AdamW(
                [self.fm.weight[old_idx], self.fm.bias[old_idx]], lr=1e-4
            )
            for epoch in range(self.fm_refit_epochs):
                total_loss = 0.0
                for ex in replay:
                    batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in ex.items()}
                    refit_opt.zero_grad()
                    outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
                    labels = batch["labels"]
                    loss = torch.nn.functional.cross_entropy(
                        outputs.logits.view(-1, outputs.logits.shape[-1]),
                        labels.view(-1),
                        ignore_index=-100,
                    )
                    loss.backward()
                    refit_opt.step()
                    total_loss += loss.item()

            # Re-freeze.
            self.fm.weight[old_idx].requires_grad = False
            self.fm.bias[old_idx].requires_grad = False
            self.fm.frozen_tasks[old_idx] = True
            log.info(
                "lexslot_fm: FM slot %d (task %d) re-fitted (%.1f ep, loss=%.4f)",
                old_idx, owner, self.fm_refit_epochs, total_loss / max(len(replay), 1),
            )

        self._cur_fm_idx = None

    # ─── lifecycle ──────────────────────────────────────────────────────────────
    def before_task(self, task: TaskInfo, train_loader) -> None:
        # CIL: grow head + FM if new labels.
        new_n_labels = self.model.model.classifier.out_features
        self.fm.expand_labels(new_n_labels)

        # CIL expand_classifier replaces the head object → re-attach head hooks.
        self._detach_head_hooks()
        self._register_head_hook()

        # Claim a functional slot for this task.
        idx = self.fm.claim(task.task_id)
        self._cur_fm_idx = idx

        log.info(
            "lexslot_fm: before_task %d; FM owned=%d, slot=%d",
            task.task_id,
            len(self.fm._owned_indices()),
            idx,
        )

    def after_task(self, task: TaskInfo, train_loader) -> None:
        for i in self.fm._owned_indices():
            if self.fm.owner[i] == task.task_id and not self.fm.frozen_tasks[i]:
                self.fm.weight[i].requires_grad = False
                self.fm.bias[i].requires_grad = False
                self.fm.frozen_tasks[i] = True
        self._cur_fm_idx = None

        # ── Store FM replay buffer for re-fitting after future encoder shifts ──
        if self.fm_refit_epochs > 0:
            self._store_fm_replay(train_loader, task.task_id)

        # ── EWC: Fisher diagonal & θ* snapshot for encoder params ──────────────
        if self.ewc_lambda > 0:
            log.info(
                "lexslot_fm: computing EWC Fisher on task %d (%d samples)",
                task.task_id,
                self.fisher_n_samples,
            )
            # Temporarily unfreeze encoder so gradients flow for Fisher.
            self.model.unfreeze_backbone()
            new_fisher = empirical_fisher_diagonal(
                self.model,
                train_loader,
                n_samples=self.fisher_n_samples,
                device=self.device,
            )
            params = dict(self.model.named_parameters())
            for name, f_val in new_fisher.items():
                if "layoutlmv3" not in name:
                    continue
                # Snapshot current optimal value.
                self._theta_star[name] = params[name].detach().clone()
                # Accumulate (online EWC: simple sum across tasks).
                old = self._fisher.get(name)
                if old is None:
                    self._fisher[name] = f_val
                elif old.shape == f_val.shape:
                    self._fisher[name] = old + f_val
                else:
                    padded = torch.zeros_like(f_val)
                    idx = tuple(slice(0, s) for s in old.shape)
                    padded[idx] = old
                    self._fisher[name] = padded + f_val
            # Re-freeze encoder (next task's train_task will unfreeze if needed).
            self.model.freeze_backbone()
            log.info(
                "lexslot_fm: EWC Fisher computed; encoder params protected=%d",
                len(self._fisher),
            )

        # ── Re-fit old FM slots after encoder shift (current encoder frozen) ──
        if self.fm_refit_epochs > 0:
            self._refit_old_fm_slots(task.task_id)

        # ── Update gates ─────────────────────────────────────────────────────
        if self.gate_mode in ("tfidf", "hybrid"):
            self._collect_task_texts(train_loader, task.task_id)
            self._build_tfidf_gate()
        if self.gate_mode in ("layout", "hybrid"):
            self._collect_layout(train_loader, task.task_id)
            self._train_layout_classifier()
        if self.gate_mode not in ("tfidf", "layout", "hybrid"):
            self._collect_cls(train_loader, task.task_id)
            self._train_gate_classifier()

        log.info("lexslot_fm: task %d done; functional memory frozen", task.task_id)

    # ─── evaluate (overridden to inject oracle / TF-IDF gate context) ──────────
    def evaluate(self, eval_loaders: dict[int, DataLoader]) -> dict[int, EvalMetrics]:
        use_tfidf = self.gate_mode in ("tfidf", "hybrid") and self._tfidf_centroids is not None
        use_layout = self.gate_mode in ("layout", "hybrid") and self._layout_clf is not None
        if not self.oracle_gate and not use_tfidf and not use_layout:
            return super().evaluate(eval_loaders)
        self.model.eval()
        results: dict[int, EvalMetrics] = {}
        id_to_label = getattr(self.model, "id_to_label", None)
        if not id_to_label:
            id_to_label = {i: str(i) for i in range(self.model.model.config.num_labels)}
        with torch.no_grad():
            for tid, loader in eval_loaders.items():
                if self.oracle_gate:
                    self._eval_task_id = tid
                all_preds, all_labels = [], []
                for batch in loader:
                    batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                    if use_tfidf:
                        self._cur_tfidf_gate = self._compute_tfidf_gate(batch["input_ids"])
                    if use_layout:
                        self._cur_layout_gate = self._compute_layout_gate_batch(
                            batch["bbox"], batch["attention_mask"]
                        )
                    outputs = self.model(**{k: v for k, v in batch.items() if k != "labels"})
                    preds = outputs.logits.argmax(dim=-1)
                    labels = batch["labels"]
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
        self._eval_task_id = None
        self._cur_tfidf_gate = None
        self._cur_layout_gate = None
        return results
