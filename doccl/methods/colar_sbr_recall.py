"""CoLaR-SBR-Recall — a memory box of session brain-states, recalled at inference.

CoLaR-SBR's bank is write-only at test time. Here the bank also stores, per session, a
small LoRA delta of the upper trunk (layers >= k); block b is evaluated UNDER the delta of
the session that created it, so e_b = 1 - p_b(O_b) is computed by the network state that
learned those tags. The base trunk is frozen after task 0 (session 0's "delta" is the base
itself), so old sessions are forgetting-free by construction and a stored delta can never
go stale — the LARM failure (a correction above k on a moving trunk) cannot occur.

Routing needs no task id: SBR's per-session O_b is the retrieval key (v2 supervision rule
unchanged). Closest prior art: ICE (frozen PLM + per-session classifier, no adapter, no
O_b), MoE-LoRA CL (separate router), CL-LoRA (shared softmax: diag 93 but AA 18.7 here).

Implementation note: LoRA is hand-rolled (``SessionLoRA`` wraps the Q/K/V ``nn.Linear`` of
layers >= k) instead of PEFT — PEFT re-paths the inner model and wraps the classifier in a
per-adapter ModulesToSaveWrapper, which breaks SBR's single shared head and feature hook.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torch.utils.data import DataLoader

from doccl.methods.colar_sbr import CoLaRSBR, aggregate_blocks, block_scores
from doccl.types import TaskInfo

log = logging.getLogger(__name__)

__all__ = ["CoLaRSBRRecall", "SessionLoRA"]


class SessionLoRA(nn.Module):
    """``nn.Linear`` plus one (A, B) low-rank pair per session; ``active`` selects which
    (``None`` = base only). A is Kaiming-uniform, B zero, so a fresh delta is the identity."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank, self.scale = rank, alpha / rank
        self.A = nn.ParameterList()
        self.B = nn.ParameterList()
        self.active: int | None = None

    def add_session(self) -> int:
        d_in, d_out = self.base.in_features, self.base.out_features
        a = nn.Parameter(torch.empty(self.rank, d_in, device=self.base.weight.device))
        nn.init.kaiming_uniform_(a, a=5**0.5)
        self.A.append(a)
        self.B.append(nn.Parameter(torch.zeros(d_out, self.rank, device=self.base.weight.device)))
        return len(self.A) - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.active is None:
            return y
        a, b = self.A[self.active], self.B[self.active]
        return y + (x.to(a.dtype) @ a.t() @ b.t()).to(y.dtype) * self.scale


class CoLaRSBRRecall(CoLaRSBR):
    """CoLaR-SBR whose blocks are read under per-session recalled trunk deltas."""

    name = "colar_sbr_recall"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.rank = int(config.get("lora_rank", 8))
        self.alpha = float(config.get("lora_alpha", 16))
        self.targets = list(config.get("target_modules", ["query", "key", "value"]))
        self.loras: list[SessionLoRA] = []
        for layer in self._encoder_layers()[self.split_layer_k :]:
            for name, mod in list(layer.named_modules()):
                if name.rsplit(".", 1)[-1] in self.targets and isinstance(mod, nn.Linear):
                    parent = layer.get_submodule(name.rsplit(".", 1)[0]) if "." in name else layer
                    wrapped = SessionLoRA(mod, self.rank, self.alpha)
                    setattr(parent, name.rsplit(".", 1)[-1], wrapped)
                    self.loras.append(wrapped)
        self.block_delta: list[int | None] = []  # per block: delta index (None = base)
        self._cur_delta: int | None = None
        log.info(
            "colar_sbr_recall: %d LoRA sites (r=%d) on layers >= %d",
            len(self.loras),
            self.rank,
            self.split_layer_k,
        )

    # ------------------------------------------------------------------ deltas
    def _set_delta(self, idx: int | None) -> None:
        for m in self.loras:
            m.active = idx

    def _new_delta(self) -> int:
        idx = None
        for m in self.loras:
            idx = m.add_session()
        return idx

    def _apply_freeze_map(self) -> None:
        # Whole trunk frozen (base never moves again); plasticity = current delta + head.
        self.model.freeze_backbone()
        for p in self.model.sbr_o.parameters():
            p.requires_grad = True
        self._train_only_current_delta()
        log.info(
            "colar_sbr_recall: base frozen after task 0; trainable params=%d",
            self.model.trainable_param_count(),
        )

    def _train_only_current_delta(self) -> None:
        for m in self.loras:
            for i, (a, b) in enumerate(zip(m.A, m.B, strict=True)):
                a.requires_grad = b.requires_grad = i == self._cur_delta

    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        n_blocks = len(self.blocks)
        super().before_task(task, train_loader)
        if task.task_id == 0:
            self._cur_delta = None  # session 0 trains the base itself
        else:
            # ponytail: a session that adds no block (DIL / revisit) reuses the delta of the
            # first block it fully trains; a fresh session gets a fresh delta.
            full = [b for b, mode in self._plan(self._cur_session) if mode == "full"]
            if len(self.blocks) > n_blocks or not full:
                self._cur_delta = self._new_delta()
            else:
                self._cur_delta = self.block_delta[full[0]]
            self._train_only_current_delta()
        for _ in range(len(self.blocks) - n_blocks):
            self.block_delta.append(self._cur_delta)
        self._set_delta(self._cur_delta)

    # ------------------------------------------------------------------ replay
    def _replay_forward(self, replay: dict[str, torch.Tensor]):
        """Each banked doc is replayed under the delta of the session that banked it."""
        sessions = replay["session"].tolist()
        deltas = [self._session_delta(s) for s in sessions]
        outs, losses = None, []
        for d in dict.fromkeys(deltas):
            rows = [i for i, x in enumerate(deltas) if x == d]
            sub = {
                k: (v[rows] if torch.is_tensor(v) and v.shape[:1] == (len(deltas),) else v)
                for k, v in replay.items()
            }
            self._set_delta(d)
            outs = super()._replay_forward(sub)
            losses.append(outs.loss * len(rows))
        self._set_delta(self._cur_delta)
        outs.loss = torch.stack(losses).sum() / len(deltas)
        return outs

    def _session_delta(self, session: int) -> int | None:
        full = [b for b, mode in self._plan(session) if mode == "full"]
        return self.block_delta[full[0]] if full else None

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _predict(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self._hook_classifier()
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        if not self.blocks:
            out = self.model(**inputs)
            return torch.zeros(out.logits.shape[:2], dtype=torch.long, device=out.logits.device)
        bls, n_head = [], None
        for d in dict.fromkeys(self.block_delta):  # one forward per distinct delta
            self._set_delta(d)
            out = self.model(**inputs)
            n_head = out.logits.shape[-1]
            for b, bd in enumerate(self.block_delta):
                if bd == d:
                    bls.append((b, self._block_logits(out.logits, self._feat, b)))
        self._set_delta(self._cur_delta)
        bls = [bl for _, bl in sorted(bls)]
        if self._dump is not None:
            e, tags = block_scores(bls, self.blocks)
            mask = batch["labels"] != -100
            self._dump["e"].append(e[mask].half().cpu())
            self._dump["tag"].append(tags[mask].short().cpu())
            self._dump["gold"].append(batch["labels"][mask].short().cpu())
        return aggregate_blocks(
            bls, self.blocks, self.threshold, getattr(self.model, "_allowed", None), n_head
        )

    def memory_bytes(self) -> int:
        lora = sum(a.numel() + b.numel() for m in self.loras for a, b in zip(m.A, m.B, strict=True))
        return super().memory_bytes() + lora * 2  # fp16-storable deltas
