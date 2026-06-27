"""LexSlot — lexically-gated slot memories at the forgetting locus (head + late layers).

Subclasses DocCL (inheriting its CE + KD + reservoir-replay + depth-Fisher train_task). Slot
memories integrate via FORWARD HOOKS: a hook on the classifier adds head logit-slots; hooks
on the late encoder-layer modules add representation-slot shifts. So DocCL's standard-forward
train loop is reused unchanged. Per task, an OCR signature updates a task-similarity matrix S;
S derives a per-task slot-gradient mask (set on every slot module) so low-similarity tasks
cannot overwrite unrelated slots (no forgetting) while high-similarity tasks co-train shared
slots (transfer). `slot_depth` controls WHERE slots are placed (head/late, depth-weighted, vs
uniform) and `slot_sharing` controls HOW slots share (soft/hard/off) — the two ablation axes.
"""

from __future__ import annotations

import logging

import torch

from doccl.methods.doccl import DocCL
from doccl.methods.hybrid_routed_prompt import sparse_doc_vectors
from doccl.methods.lexslot_mask import slot_trainable_mask, task_similarity_matrix
from doccl.methods.lexslot_memory import LogitSlots, ReprSlots
from doccl.types import TaskInfo

log = logging.getLogger(__name__)

__all__ = ["LexSlot"]


class LexSlot(DocCL):
    name = "lexslot"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.slot_depth = config.get("slot_depth", "head_late")
        self.slot_sharing = config.get("slot_sharing", "soft")
        self.share_threshold = float(config.get("share_threshold", 0.5))
        self.n_slots_head = int(config.get("n_slots_head", 48))
        self.n_slots_late = int(config.get("n_slots_late", 12))
        self.repr_rank = int(config.get("repr_rank", 16))
        if config.get("lexical_signal", "ocr") != "ocr":
            log.warning(
                "lexslot: lexical_signal=%s not implemented; using ocr",
                config.get("lexical_signal"),
            )
        self.hidden_dim = model.hidden_size
        self.vocab_size = int(self.model.model.config.vocab_size)
        n_labels = self.model.model.classifier.out_features

        # Head logit-slots.
        self.head_slots = LogitSlots(self.n_slots_head, self.hidden_dim, n_labels).to(self.device)
        # Capture the classifier's input tensor (per-token features) via a pre-hook.
        self._cur_feats: torch.Tensor | None = None

        # Late repr-slots: one ReprSlots per targeted late layer.
        self._late_idx = self._late_layer_indices(self.model.num_layers, self.slot_depth)
        self.late_slots = torch.nn.ModuleDict()
        for i in self._late_idx:
            rs = ReprSlots(self.n_slots_late, self.hidden_dim, self.repr_rank).to(self.device)
            self.late_slots[str(i)] = rs

        # Register all slot forward hooks on the live model (detachable for the KD deepcopy).
        self._hook_handles: list = []
        self._register_slot_hooks()

        self._task_sigs: list[torch.Tensor] = []  # per-task OCR signature (V,)

    # ─── backbone-agnostic accessors ──────────────────────────────────────────────
    def _encoder_layers(self):
        """The encoder's per-layer ModuleList, across backbone wrapper families.

        Secondary wrappers (LiLT/BROS/BERT, ``TokenClassificationWrapper``) expose the
        inner encoder via ``model._inner`` (= ``getattr(hf_model, _inner_attr)``);
        ``LayoutLMv3Wrapper`` predates that base and nests it at ``model.layoutlmv3``.
        Both then carry ``.encoder.layer``. We probe ``_inner`` first, then fall back
        to the LayoutLMv3 path, so slot placement is identical on every backbone.
        """
        inner = getattr(self.model, "_inner", None)
        if inner is None:
            inner = getattr(self.model.model, "layoutlmv3", None)
        if inner is None:
            raise RuntimeError(
                f"lexslot: cannot locate the inner encoder on "
                f"{type(self.model).__name__}; expected model._inner or "
                "model.model.layoutlmv3."
            )
        return inner.encoder.layer

    # ─── hook (de)registration ────────────────────────────────────────────────────
    def _register_slot_hooks(self) -> None:
        """Attach the head + late-layer slot forward hooks to the live model."""
        self._hook_handles = []
        cls = self.model.model.classifier
        self._hook_handles.append(cls.register_forward_pre_hook(self._capture_cls_input))
        self._hook_handles.append(cls.register_forward_hook(self._add_head_slots))
        layers = self._encoder_layers()
        for i in self._late_idx:
            rs = self.late_slots[str(i)]
            self._hook_handles.append(layers[i].register_forward_hook(self._make_layer_hook(rs)))

    def _detach_slot_hooks(self) -> None:
        """Remove the slot hooks from the live model (so model.deepcopy is hook-free)."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    # ─── placement ────────────────────────────────────────────────────────────────
    @staticmethod
    def _late_layer_indices(num_layers: int, slot_depth: str) -> list[int]:
        third = max(num_layers // 3, 1)
        if slot_depth == "head_only":
            return []
        if slot_depth == "uniform":
            return list(range(num_layers))
        if slot_depth == "head_late_mid":
            return list(range(third, num_layers))  # mid+late
        return list(range(2 * third, num_layers))  # head_late: late only

    # ─── forward-hook integrations ─────────────────────────────────────────────
    def _capture_cls_input(self, _module, inp):
        # inp is a tuple; inp[0] is (B, L, d) features into the classifier.
        self._cur_feats = inp[0]
        return None

    def _add_head_slots(self, _module, _inp, output):
        """Forward hook on the classifier: add logit-slot bias. output (B, L, C)."""
        if self._cur_feats is None or self._cur_feats.shape[1] != output.shape[1]:
            return output
        return output + self.head_slots.logits_delta(self._cur_feats.to(output.dtype))

    def _make_layer_hook(self, rs: ReprSlots):
        """Build a forward hook that adds rs.repr_delta to a layer's *text* hidden state.

        Encoder-layer outputs differ by backbone:
          - BERT / BROS / LayoutLMv3: ``output[0]`` is the text hidden tensor.
          - LiLT: ``output[0]`` is a ``(text_hidden, layout_hidden)`` tuple — the encoder
            reads ``layer_outputs[0][0]`` (text) and ``layer_outputs[0][1]`` (layout), so
            the shift must apply to the text element and the nesting be preserved.
          - bare tensor (no tuple): shift directly.
        """

        def hook(_module, _inp, output):
            if isinstance(output, tuple):
                first = output[0]
                if isinstance(first, tuple):  # LiLT: ((text, layout), ...)
                    text_hs = first[0]
                    shifted = text_hs + rs.repr_delta(text_hs)
                    new_first = (shifted,) + tuple(first[1:])
                    return (new_first,) + tuple(output[1:])
                shifted = first + rs.repr_delta(first)  # BERT/BROS/LayoutLMv3
                return (shifted,) + tuple(output[1:])
            return output + rs.repr_delta(output)  # bare tensor

        return hook

    # ─── mask derivation ────────────────────────────────────────────────────────
    def _derive_mask(self, task_id, slot_owner, S):  # noqa: N803
        return slot_trainable_mask(task_id, slot_owner, S, self.slot_sharing, self.share_threshold)

    def _all_slot_modules(self):
        return [self.head_slots] + [self.late_slots[k] for k in self.late_slots]

    # ─── optimizer target ────────────────────────────────────────────────────────
    def trainable_parameters(self):
        """Include slot module params alongside backbone params so the optimizer trains them."""
        slot_params = list(self.head_slots.parameters()) + [
            p for rs in self.late_slots.values() for p in rs.parameters()
        ]
        return [p for p in self.model.parameters() if p.requires_grad] + slot_params

    # ─── lifecycle ──────────────────────────────────────────────────────────────
    def before_task(self, task: TaskInfo, train_loader) -> None:
        super().before_task(task, train_loader)

        # Grow the head logit-slots' proj if the classifier head expanded (CIL).
        new_n_labels = self.model.model.classifier.out_features
        self.head_slots.expand_labels(new_n_labels)

        # 1. OCR signature for this task (one pass over input_ids).
        sig = torch.zeros(self.vocab_size)
        for batch in train_loader:
            ids = batch["input_ids"]
            if torch.is_tensor(ids):
                sig += sparse_doc_vectors(ids, self.vocab_size).sum(0).cpu()
        # ensure list slot for this task id
        while len(self._task_sigs) <= task.task_id:
            self._task_sigs.append(torch.zeros(self.vocab_size))
        self._task_sigs[task.task_id] = sig

        # 2. similarity + per-task gradient mask on every slot module; claim fresh slots.
        S = task_similarity_matrix(self._task_sigs)  # noqa: N806
        for mod in self._all_slot_modules():
            # claim a fresh block for this task: the unclaimed slots become this task's.
            fresh = [s for s, o in enumerate(mod.slot_owner) if o == -1]
            claim = fresh[: max(1, mod.n_slots // max(len(self._task_sigs), 1))]
            mask = self._derive_mask(task.task_id, mod.slot_owner, S)
            mod.set_grad_mask(mask)
            mod.set_owner(claim, task.task_id)

    def after_task(self, task: TaskInfo, train_loader) -> None:
        # DocCL.after_task deepcopies self.model to build the KD teacher. The slot forward
        # hooks are bound methods of this LexSlot instance, so deepcopy would try to copy the
        # slot Parameters (non-leaf after a backward, via the grad-mask hook) and crash. Detach
        # the slot hooks from the live model around the deepcopy, then re-register them, so the
        # teacher is a pure-DocCL copy and the live model keeps its slots.
        self._detach_slot_hooks()
        try:
            super().after_task(task, train_loader)
        finally:
            self._register_slot_hooks()
        log.info(
            "lexslot: task %d done; depth=%s sharing=%s late_layers=%s",
            task.task_id,
            self.slot_depth,
            self.slot_sharing,
            self._late_idx,
        )
