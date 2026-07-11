"""Latent Replay — the missing cell of the replay 2x2 (falsification-chain level 6).

Reference: Pellegrini et al., "Latent Replay for Real-Time Continual Learning",
IROS 2020, adapted to multimodal document IE.

Stores NO raw documents. After each task, the hidden state entering encoder layer
``split_layer_k`` is captured for ``docs_per_task`` training documents (together with
``bbox``/``attention_mask``/``labels``, which are needed to recompute the relative-
position biases and the token-classification loss — not to re-embed the input).
After task 0 everything below the split point (text/patch embeddings and
``encoder.layer[:k]``) is frozen, so the stored activations are valid *by
construction* — unlike every feature-space memory in the falsification chain
(LexSlot, LexMem v1-v5, feature Gaussians), they cannot go stale under drift.

Pre-registered collision with the migration law (STATE.md experiment #4):

- the migration law says freezing layers < k reroutes forgetting into the plastic
  upper layers, conserving its total;
- latent replay delivers REAL past-task gradients to exactly those plastic layers.

This is the untested cell of the 2x2 (real gradients x feature validity): ER has
both (DIL AA 82.1 at 5 docs/task), the buffer-free memories have surrogate
gradients on stale features (63-66). Success bounds the law ("conserved *unless*
the plastic remainder receives real past-task gradients"); failure closes the
chain at level 6.

``docs_per_task: 0`` is the control arm: identical early-freeze map, no replay.

Mechanics: replay batches are pushed through the full wrapper forward with dummy
``input_ids``/``pixel_values``; a forward pre-hook on ``encoder.layer[k]`` swaps in
the stored hidden state, so the computation below k is discarded and the autograd
graph for the replay pass starts at layer k. The stored ``bbox``/``attention_mask``
drive the correct rel_pos/rel_2d_pos biases and masking for the replayed document.
"""

from __future__ import annotations

import logging
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo, TrainMetrics

log = logging.getLogger(__name__)


class LatentReplay(NaiveFineTune):
    """Frozen-trunk activation replay at encoder layer ``split_layer_k``."""

    name = "latent_replay"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.split_layer_k = int(config.get("split_layer_k", 8))
        self.docs_per_task = int(config.get("docs_per_task", 5))
        self.replay_batch_size = int(config.get("replay_batch_size", 4))
        self.store: list[dict[str, torch.Tensor]] = []
        self._inject: torch.Tensor | None = None
        self._capture: list[torch.Tensor] | None = None
        self._pixel_shape: tuple[int, ...] | None = None
        n_layers = len(self._encoder_layers())
        if not 0 < self.split_layer_k < n_layers:
            raise ValueError(f"split_layer_k must be in (0, {n_layers}), got {self.split_layer_k}")
        self._encoder_layers()[self.split_layer_k].register_forward_pre_hook(
            self._pre_hook, with_kwargs=True
        )

    # ------------------------------------------------------------------ hooks
    def _encoder_layers(self):
        # Backbone-agnostic (same resolution as lexmem._inner_encoder): secondaries
        # (LiLT/BROS/BERT) expose the inner encoder as ``_inner``; LayoutLMv3Wrapper
        # deliberately does not inherit the shared base, so fall back to its path.
        inner = getattr(self.model, "_inner", None)
        if inner is None:
            inner = getattr(self.model.model, "layoutlmv3", None)
        if inner is None:
            raise RuntimeError("latent_replay: cannot locate the inner encoder module")
        return inner.encoder.layer

    def _pre_hook(self, module, args, kwargs):
        """Capture or replace the hidden state entering layer k.

        Fires on module ``__call__`` and therefore also under the wrapper's
        per-layer gradient-checkpointing wrap (which only replaces ``forward``).

        Captures/injects ``args[0]`` (the text-stream hidden state). Single-stream
        backbones (LayoutLMv3 — unified sequence — BERT, BROS) are exact; LiLT's
        parallel layout stream is NOT stored, so LiLT latent replay is approximate
        (its layout stream re-derives from the stored bbox below k). The thesis
        experiment runs on LayoutLMv3.
        """
        if self._capture is not None:
            hidden = args[0] if args else kwargs["hidden_states"]
            self._capture.append(hidden.detach().to("cpu", torch.float16))
            return None
        if self._inject is not None:
            hidden = args[0] if args else kwargs["hidden_states"]
            injected = self._inject.to(device=hidden.device, dtype=hidden.dtype)
            if args:
                return (injected, *args[1:]), kwargs
            kwargs = dict(kwargs, hidden_states=injected)
            return args, kwargs
        return None

    # ------------------------------------------------------------------ train
    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.trainable_parameters(),
            lr=self.config.get("lr", 5e-5),
            weight_decay=self.config.get("weight_decay", 0.01),
        )
        epochs = self.config.get("epochs", 10)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        stopper = self.make_early_stopper(val_loader)

        total_loss = 0.0
        n_steps = 0
        for epoch in range(epochs):
            pbar = tqdm(train_loader, desc=f"LR T{task.task_id} ep{epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}

                optimizer.zero_grad()
                cur_out = self.model(**batch)
                ce_loss = cur_out.loss
                replay_loss = torch.zeros((), device=self.device)

                replay = self._sample_replay()
                if replay is not None:
                    replay_loss = self._replay_forward(replay).loss

                loss = ce_loss + replay_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.trainable_parameters(), max_grad_norm)
                optimizer.step()

                total_loss += float(loss.item())
                n_steps += 1
                pbar.set_postfix(
                    {"ce": f"{ce_loss.item():.3f}", "replay": f"{replay_loss.item():.3f}"}
                )
            if self._early_stop_after_epoch(stopper, val_loader, task, epoch):
                break

        stopper.restore_best(self.model)
        return TrainMetrics(
            task_id=task.task_id,
            loss=total_loss / max(n_steps, 1),
            n_steps=n_steps,
        )

    def _replay_forward(self, replay: dict[str, torch.Tensor]):
        """Full wrapper forward with the stored hidden injected at layer k."""
        bbox = replay["bbox"].to(self.device)
        labels = replay["labels"].to(self.device)
        attention_mask = replay["attention_mask"].to(self.device)
        b = bbox.shape[0]
        # pad_id value is irrelevant to correctness (everything below layer k is
        # discarded by the injection); vision-free wrappers have no processor.
        processor = getattr(self.model, "processor", None)
        pad_id = processor.tokenizer.pad_token_id if processor is not None else 0
        dummy_ids = torch.full((b, bbox.shape[1]), pad_id, dtype=torch.long, device=self.device)
        dummy_pixels = (
            torch.zeros((b, *self._pixel_shape), device=self.device)
            if self._pixel_shape is not None
            else None
        )
        self._inject = replay["hidden"]
        try:
            return self.model(
                input_ids=dummy_ids,
                bbox=bbox,
                pixel_values=dummy_pixels,
                attention_mask=attention_mask,
                labels=labels,
            )
        finally:
            self._inject = None

    def _sample_replay(self) -> dict[str, torch.Tensor] | None:
        if not self.store:
            return None
        docs = random.sample(self.store, min(self.replay_batch_size, len(self.store)))
        return {k: torch.stack([d[k] for d in docs]) for k in docs[0]}

    # ------------------------------------------------------------- after task
    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        """Freeze below the split after task 0, then bank this task's activations.

        Order matters: task-0 activations are captured AFTER the freeze point, i.e.
        from the exact encoder state that layers < k keep forever.
        """
        if task.task_id == 0:
            self._apply_freeze_map()
        if self.docs_per_task > 0:
            self._capture_task(train_loader)

    def _apply_freeze_map(self) -> None:
        """Early-freeze mirror of lexmem's map: layers < k + embeddings frozen,
        layers >= k and the classifier head plastic (they receive the replay
        gradients — the migration-law test lives exactly there)."""
        self.model.freeze_backbone()  # backbone only; classifier stays trainable
        for layer in self._encoder_layers()[self.split_layer_k :]:
            for p in layer.parameters():
                p.requires_grad = True
        log.info(
            "latent_replay: freeze map — layers <%d + embeddings frozen; trainable params=%d",
            self.split_layer_k,
            self.model.trainable_param_count(),
        )

    def _capture_task(self, train_loader: DataLoader) -> None:
        """Bank layer-k activations for ``docs_per_task`` docs.

        ``doc_selection: random`` (default, byte-identical to all prior runs): the first
        docs from the shuffled loader — a random draw matching the retention-curve ER points.

        ``doc_selection: kcenter``: gather a candidate pool (``selection_pool`` docs), then
        greedy k-center (farthest-point) selection on per-doc mean latents — picks the
        docs_per_task docs that best COVER the task's feature region. Tests the hypothesis
        that random-d5's gap to d50 (78.2 vs 87.3 on dil/k4) is coverage, not count.
        """
        was_training = self.model.training
        self.model.eval()
        selection = str(self.config.get("doc_selection", "random"))
        pool_cap = int(self.config.get("selection_pool", 100)) if selection == "kcenter" else 0
        quota = self.docs_per_task if selection == "random" else max(pool_cap, self.docs_per_task)
        candidates: list[dict] = []
        with torch.no_grad():
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                if "pixel_values" in batch:  # vision-free backbones have none
                    self._pixel_shape = tuple(batch["pixel_values"].shape[1:])
                self._capture = []
                try:
                    self.model(**batch)
                    hidden = self._capture[0]
                finally:
                    self._capture = None
                for i in range(hidden.shape[0]):
                    if len(candidates) >= quota:
                        break
                    candidates.append(
                        {
                            "hidden": hidden[i],
                            "bbox": batch["bbox"][i].cpu(),
                            "attention_mask": batch["attention_mask"][i].cpu(),
                            "labels": batch["labels"][i].cpu(),
                        }
                    )
                if len(candidates) >= quota:
                    break
        if was_training:
            self.model.train()
        if selection == "kcenter" and len(candidates) > self.docs_per_task:
            picked = self._kcenter_select(candidates, self.docs_per_task)
        else:
            picked = candidates[: self.docs_per_task]
        self.store.extend(picked)
        log.info(
            "latent_replay: banked %d docs (store=%d, ~%.1f MB fp16, selection=%s/pool=%d)",
            len(picked),
            len(self.store),
            sum(d["hidden"].numel() for d in self.store) * 2 / 1e6,
            selection,
            len(candidates),
        )

    @staticmethod
    def _kcenter_select(candidates: list[dict], k: int) -> list[dict]:
        """Greedy k-center (farthest-point) over per-doc mean latents of real text tokens.
        Deterministic: seed = the doc nearest the pool centroid, then repeatedly add the
        doc farthest from the selected set. Maximizes feature-region coverage."""
        means = torch.stack(
            [
                d["hidden"][: d["attention_mask"].shape[0]][d["attention_mask"].bool()]
                .float()
                .mean(0)
                for d in candidates
            ]
        )  # (N, d)
        centroid = means.mean(0, keepdim=True)
        first = int(torch.cdist(means, centroid).squeeze(1).argmin())
        chosen = [first]
        dist = torch.cdist(means, means[first : first + 1]).squeeze(1)  # (N,)
        while len(chosen) < min(k, len(candidates)):
            nxt = int(dist.argmax())
            chosen.append(nxt)
            dist = torch.minimum(dist, torch.cdist(means, means[nxt : nxt + 1]).squeeze(1))
        return [candidates[i] for i in chosen]

    def memory_bytes(self) -> int:
        """Raw replay-buffer footprint in bytes — the x-axis of the SLR memory Pareto
        curve for the incumbent (hidden fp16 + bbox/mask/labels int64). Overridden by the
        compressed variants (SpectralMemory, AGLRReplay)."""
        total = 0
        for d in self.store:
            total += d["hidden"].numel() * 2  # fp16 activations
            total += (d["bbox"].numel() + d["attention_mask"].numel() + d["labels"].numel()) * 8
        return total
