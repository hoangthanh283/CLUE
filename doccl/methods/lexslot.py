"""LexSlot — lexically-gated slot memories at the forgetting locus (head + late layers).

**Standalone method.** LexSlot is a *total approach* on its own: a partially-frozen
backbone (lower encoder frozen, head + late layers + slots trainable) trained with plain
cross-entropy — **no replay buffer, no KD teacher, no Fisher penalty.** Retention comes
entirely from two lexically-driven, mutually-consistent mechanisms operating at the
forgetting locus (the head + late layers the diagnosis implicates):

  1. **Gradient isolation (train time).** A per-slot gradient mask, derived from the
     inter-task OCR-signature similarity S, scales each slot's gradient so a low-similarity
     task cannot overwrite an unrelated task's slots (no forgetting) while high-similarity
     tasks co-train shared slots (transfer).
  2. **Inference lexical gating (forward time, train + eval consistent).** A slot owned by
     task t fires in proportion to cos(sig(doc), sig_t). So at eval on a task-τ document,
     task-τ slots fire strongly and foreign slots are suppressed — routing the right
     memory to the right document at prediction time, which the gradient mask alone cannot do.

Slots integrate via FORWARD HOOKS: a hook on the classifier adds head logit-slots; hooks on
the late encoder-layer modules add representation-slot shifts. The inference gate is
installed by a single forward pre-hook on the wrapper model (so it fires on every forward —
train, eval, and early-stop val — with no loop duplication). `slot_depth` controls WHERE
slots are placed (head/late, depth-weighted, vs uniform) and `slot_sharing` controls HOW
slots share (soft/hard/off) — the two ablation axes.

Partial-freeze invariant: the layers that carry slots are exactly the layers that stay
trainable. `freeze_lower=true` freezes the encoder embeddings + every non-slot-bearing
layer (i.e. all layers NOT in `_late_idx`), so the slot placement and the trainable set are
guaranteed aligned — slots never sit on frozen features, and frozen features never drift
under the slots.
"""

from __future__ import annotations

import logging

import torch

from doccl.methods.hybrid_routed_prompt import sparse_doc_vectors
from doccl.methods.lexslot_mask import slot_trainable_mask, task_similarity_matrix
from doccl.methods.lexslot_memory import LogitSlots, ReprSlots
from doccl.methods.naive import NaiveFineTune
from doccl.types import TaskInfo

log = logging.getLogger(__name__)

__all__ = ["LexSlot"]


class LexSlot(NaiveFineTune):
    """Standalone lexically-gated slot-memory method (no DocCL base).

    Inherits NaiveFineTune's plain-CE ``train_task`` / ``evaluate`` unchanged — the
    inference lexical gate is installed by a forward pre-hook on the wrapper, so it is
    active on every forward (train, eval, early-stop val) with no loop duplication.
    """

    name = "lexslot"

    def __init__(self, model, config):
        # NaiveFineTune -> ContinualMethod.__init__ (model/config/state/device/amp). No
        # DocCL state (no theta_star / fisher / teacher / buffer) is created here.
        super().__init__(model, config)
        self.slot_depth = config.get("slot_depth", "head_late")
        self.slot_sharing = config.get("slot_sharing", "soft")
        self.share_threshold = float(config.get("share_threshold", 0.5))
        self.n_slots_head = int(config.get("n_slots_head", 48))
        self.n_slots_late = int(config.get("n_slots_late", 12))
        self.repr_rank = int(config.get("repr_rank", 16))
        # Expected number of CL tasks: fixes the per-task slot budget so each task claims a
        # disjoint block of n_slots // n_tasks slots (mirrors HRP's task-pinned slot pool).
        # Without this the first task would greedily claim ALL fresh slots (a real bug:
        # later tasks would own none and, under slot_sharing=off, could update nothing).
        self.n_tasks = int(config.get("n_tasks", 10))
        # Inference lexical gating (the new forward-time mechanism). Default on; an
        # ablation can disable it to isolate the gradient-mask-only regime.
        self.infer_gate = bool(config.get("infer_gate", True))
        # Partial freeze: freeze every encoder layer that does NOT carry slots (plus the
        # embeddings), keeping the slot-bearing late layers + classifier head + slots
        # trainable. Keeps slot placement and the trainable set exactly aligned.
        self.freeze_lower = bool(config.get("freeze_lower", True))
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

        # Register all slot forward hooks on the live model.
        self._hook_handles: list = []
        self._register_slot_hooks()
        # Inference-gate pre-hook on the wrapper: fires on every forward (train/eval/val)
        # and stashes a per-batch (B, n_slots) gate on each slot module before the head /
        # late-layer hooks read it. Registered once; never detached (CIL classifier
        # replacement does not orphan a wrapper-level hook).
        self._gate_handle = self.model.register_forward_pre_hook(
            self._install_infer_gate, with_kwargs=True
        )

        self._task_sigs: list[torch.Tensor] = []  # per-task OCR signature (V,)
        # Cache of the stacked task signatures (T, V) on the active device, rebuilt when a
        # new task signature is appended so the per-forward gate matmul allocates nothing.
        self._sigs_cache: torch.Tensor | None = None
        self._sigs_cache_len: int = -1

        # Apply the partial freeze once. expand_classifier (CIL) never touches the encoder
        # layers, so the freeze persists for the whole run.
        self._freeze_lower_encoder()

    def _inner_module(self):
        """The inner encoder submodule (carries ``.encoder.layer`` + ``.embeddings``).

        Secondary wrappers (LiLT/BROS/BERT, ``TokenClassificationWrapper``) expose it via
        the ``_inner`` property (= ``getattr(hf_model, _inner_attr)``);
        ``LayoutLMv3Wrapper`` predates that base and nests it at ``model.model.layoutlmv3``.
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
        return inner

    def _encoder_layers(self):
        """The encoder's per-layer ModuleList, across backbone wrapper families."""
        return self._inner_module().encoder.layer

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
        """Remove the slot hooks from the live model."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

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

    def _task_sigs_matrix(self) -> torch.Tensor | None:
        """Stacked (T, V) task signatures on the active device, L2-normalised, cached
        per task count.

        The signatures are normalised so that ``doc @ sigs.T`` is a true cosine in
        [0,1] (bag-of-tokens vectors are non-negative). Without normalisation the
        result scales with ``‖sig‖`` (a task's total token count) and is unbounded,
        letting a large task's slots fire with a gate of hundreds on every document
        and overwhelming the base head. ``sparse_doc_vectors`` already normalises the
        per-document side; this fixes the signature side to match.
        """
        if not self._task_sigs:
            return None
        if self._sigs_cache is None or self._sigs_cache_len != len(self._task_sigs):
            stacked = torch.stack(self._task_sigs).to(self.device)
            norms = stacked.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self._sigs_cache = stacked / norms
            self._sigs_cache_len = len(self._task_sigs)
        return self._sigs_cache

    def _install_infer_gate(self, _module, _args, kwargs) -> None:
        """Forward pre-hook on the wrapper: install the per-document lexical gate on
        every slot module before the head / late-layer hooks read it.

        For a batch of documents, each slot owned by task t fires in proportion to
        cos(sig(doc), sig_t); unclaimed slots fire fully (1.0). The result is a
        (B, n_slots) tensor stashed on each slot module as ``_infer_gate``; the slot
        deltas multiply their activation by it. No-op (leaves ``_infer_gate=None``) when
        ``infer_gate`` is off or no task signature exists yet (task 0 has no prior to
        gate against, and unclaimed slots default to 1.0 anyway).
        """
        if not self.infer_gate:
            return
        sigs = self._task_sigs_matrix()  # (T, V) on device, or None
        if sigs is None:
            return
        ids = kwargs.get("input_ids")
        if not torch.is_tensor(ids):
            return
        # (B, V) L2-normalised doc signatures; cosine = doc @ sigs.T.
        doc = sparse_doc_vectors(ids, self.vocab_size)  # (B, V) on ids.device
        cos = doc @ sigs.to(doc.device).T  # (B, T)
        for mod in self._all_slot_modules():
            owner = torch.as_tensor(
                mod.slot_owner, device=cos.device, dtype=torch.long
            )  # (n_slots,)
            g = torch.ones(cos.shape[0], mod.n_slots, device=cos.device, dtype=cos.dtype)
            claimed = owner >= 0
            if claimed.any():
                idx = owner.clamp(min=0)  # safe gather index for claimed slots
                g[:, claimed] = cos[:, idx[claimed]]  # (B, n_claimed)
            mod._infer_gate = g

    def _derive_mask(self, task_id, slot_owner, S):  # noqa: N803
        return slot_trainable_mask(task_id, slot_owner, S, self.slot_sharing, self.share_threshold)

    def _all_slot_modules(self):
        return [self.head_slots] + [self.late_slots[k] for k in self.late_slots]

    def _freeze_lower_encoder(self) -> None:
        """Freeze every encoder layer that does NOT carry a slot (+ embeddings).

        The slot-bearing layers (``_late_idx``) and the classifier head stay trainable;
        everything below is frozen so stable pretrained features flow into the slots
        without drifting. The trainable set is therefore exactly aligned with the slot
        placement — slots never sit on frozen features, frozen features never drift.
        """
        if not self.freeze_lower:
            return
        inner = self._inner_module()
        # Embeddings are always below the slot locus -> freeze.
        for p in inner.embeddings.parameters():
            p.requires_grad = False
        layers = inner.encoder.layer
        trainable = set(self._late_idx)
        for i, layer in enumerate(layers):
            on = i in trainable
            for p in layer.parameters():
                p.requires_grad = on
        n_train_layers = len(trainable)
        n_layers = len(layers)
        log.info(
            "lexslot: partial freeze — trainable encoder layers %s/%s (slot-bearing); "
            "head + slots trainable.",
            n_train_layers,
            n_layers,
        )

    def trainable_parameters(self):
        """Slot params + the unfrozen backbone params (head + late layers; lower frozen)."""
        slot_params = list(self.head_slots.parameters()) + [
            p for rs in self.late_slots.values() for p in rs.parameters()
        ]
        return [p for p in self.model.parameters() if p.requires_grad] + slot_params

    def before_task(self, task: TaskInfo, train_loader) -> None:
        # No DocCL.before_task (no buffer/fisher/teacher to init). NaiveFineTune's
        # before_task is the base no-op, so there is nothing to super(). The classifier
        # head was already expanded by train.py before this hook fires.

        # Grow the head logit-slots' proj if the classifier head expanded (CIL).
        new_n_labels = self.model.model.classifier.out_features
        self.head_slots.expand_labels(new_n_labels)

        # CRITICAL: a CIL expand_classifier REPLACES self.model.model.classifier with a NEW
        # nn.Linear object, orphaning the head-slot forward hooks (they stayed on the old
        # object). Re-point all slot hooks at the CURRENT modules at the start of every
        # task, so the head slots actually contribute during training for every CIL task —
        # not just task 0. (DIL never grows the head, so this is a harmless re-attach
        # there.) The wrapper-level inference-gate hook is NOT touched (CIL replacement
        # does not orphan it).
        self._detach_slot_hooks()
        self._register_slot_hooks()

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
        # Invalidate the stacked-signature cache so the inference gate picks up the new
        # task signature at the next forward.
        self._sigs_cache = None
        self._sigs_cache_len = -1

        # 2. similarity + per-task gradient mask on every slot module; claim fresh slots.
        S = task_similarity_matrix(self._task_sigs)  # noqa: N806
        for mod in self._all_slot_modules():
            # Claim a FIXED-SIZE disjoint block for this task: n_slots // n_tasks slots from
            # the still-unclaimed pool (>=1). This keeps a per-task budget so every task owns
            # its own slots — NOT a greedy grab of all fresh slots by task 0.
            per_task = max(1, mod.n_slots // max(self.n_tasks, 1))
            fresh = [s for s, o in enumerate(mod.slot_owner) if o == -1]
            claim = fresh[:per_task]
            mask = self._derive_mask(task.task_id, mod.slot_owner, S)
            mod.set_grad_mask(mask)
            mod.set_owner(claim, task.task_id)

    def after_task(self, task: TaskInfo, train_loader) -> None:
        # Standalone: no KD teacher to deepcopy, no Fisher to accumulate, no buffer to
        # update. Isolation is the per-slot gradient mask set in before_task for the NEXT
        # task; nothing to snapshot here. (The slot hooks stay registered — there is no
        # teacher deepcopy to detach them around.)
        log.info(
            "lexslot: task %d done (standalone); depth=%s sharing=%s late_layers=%s",
            task.task_id,
            self.slot_depth,
            self.slot_sharing,
            self._late_idx,
        )
