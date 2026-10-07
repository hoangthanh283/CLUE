"""CoLaR++ — CoLaR with head re-alignment on real stored latents (image CIL).

Built on three measured facts (docs/RESULTS_REPORT_2026-10.md):
  * forgetting sits at the classifier head (97-100 % of Fisher-weighted displacement, ViT);
  * SLCA's whole gain over its slow-trunk baseline is post-task head alignment (H2b);
  * re-grounding the head on real whole-sample latents beats Gaussian summaries (F3b).

Switches (all off ⇒ plain CoLaR / raw latent replay with ``rank_r: 0``):
  head_align_epochs  CA: after each task, run every stored latent through the current
                     layers k..L (eval, no grad), cache the classifier-input features and
                     retrain the head on them, class-balanced (SLCA defaults: SGD 5e-3, 10 ep).
  trunk_lr_scale     SL: plastic trunk layers at ``lr * scale``; head at ``lr``.
  weight_align       WA: rescale new-class head rows to the old-class mean row norm.
  replay_balance     BAL: "class" ⇒ replay batches drawn uniformly over stored classes.
  quant              Q8: "int8" ⇒ stored factors int8 with per-row scale.
  token_pool         POOL: p>1 ⇒ CLS + p×p average-pooled patch tokens (ViT only; the
                     encoder layers accept any sequence length after the embeddings).
"""

from __future__ import annotations

import logging
import math
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from doccl.methods.colar import CoLaR
from doccl.methods.latent_replay import LatentReplay
from doccl.types import TaskInfo  # noqa: F401  (signature types)

log = logging.getLogger(__name__)


def _q8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    return torch.round(x / scale).clamp(-127, 127).to(torch.int8), scale.to(torch.float16)


def _pool_tokens(h: torch.Tensor, p: int) -> torch.Tensor:
    """(1 + n², d) → (1 + (n/p)², d): keep CLS, average-pool the patch grid."""
    cls, patches = h[:1], h[1:]
    n = int(math.isqrt(patches.shape[0]))
    if n * n != patches.shape[0]:
        return h  # not a square patch grid (e.g. document tokens): leave untouched
    grid = patches.T.reshape(1, -1, n, n).float()
    pooled = F.avg_pool2d(grid, p).reshape(h.shape[1], -1).T.to(h.dtype)
    return torch.cat([cls, pooled], dim=0)


class CoLaRPP(CoLaR):
    name = "colar_pp"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.ca_epochs = int(config.get("head_align_epochs", 0))
        self.ca_lr = float(config.get("head_align_lr", 5e-3))
        self.ca_batch = int(config.get("head_align_batch", 64))
        self.trunk_lr_scale = float(config.get("trunk_lr_scale", 1.0))
        self.weight_align = bool(config.get("weight_align", False))
        self.replay_balance = str(config.get("replay_balance", "none"))
        self.quant = str(config.get("quant", "none"))
        self.token_pool = int(config.get("token_pool", 1))
        # Anti-memorisation (Amendment 6): Phase A showed the replay loss → 0 within a few
        # epochs (the stored latents are memorised), so replay stops carrying signal.
        self.latent_distill = float(config.get("latent_distill", 0.0))  # LD: logit-MSE weight
        self.token_drop = float(config.get("token_drop", 0.0))  # TD: patch-token drop rate
        self._seen_before = 0

    # ---------------------------------------------------------------- storage
    def _capture_task(self, train_loader: DataLoader) -> None:
        n_before = len(self.store)
        LatentReplay._capture_task(self, train_loader)  # raw bank first, then encode
        for d in self.store[n_before:]:
            h = d.pop("hidden").float()
            if self.token_pool > 1:
                h = _pool_tokens(h, self.token_pool)
            if self.rank_r > 0:
                u, s, vh = torch.linalg.svd(h, full_matrices=False)
                r = min(self.rank_r, s.shape[0])
                parts = {"us": u[:, :r] * s[:r], "v": vh[:r]}
            else:
                parts = {"hidden": h}
            for key, t in parts.items():
                if self.quant == "int8":
                    d[key], d[f"{key}_scale"] = _q8(t)
                else:
                    d[key] = t.to(torch.float16)
        if self.latent_distill > 0:
            self._bank_logits(self.store[n_before:])
        log.info(
            "colar_pp: encoded %d samples (rank=%d pool=%d quant=%s) store=%.1f MB",
            len(self.store) - n_before,
            self.rank_r,
            self.token_pool,
            self.quant,
            self.memory_bytes() / 1e6,
        )

    @staticmethod
    def _deq(d: dict, key: str) -> torch.Tensor:
        t = d[key]
        return t.float() * d[f"{key}_scale"].float() if f"{key}_scale" in d else t.float()

    def _decode(self, d: dict) -> torch.Tensor:
        if "hidden" in d:
            return self._deq(d, "hidden")
        return self._deq(d, "us") @ self._deq(d, "v")

    def _stack_replay(
        self, docs: list[dict[str, torch.Tensor]], augment: bool = False
    ) -> dict[str, torch.Tensor]:
        hidden = torch.stack([self._decode(d) for d in docs])
        if augment and self.token_drop > 0 and "bbox" not in docs[0] and hidden.shape[1] > 2:
            # keep CLS + a random subset of patch tokens (same count per batch, own draw per
            # sample); ViT layers above the embeddings accept any sequence length.
            n_patch = hidden.shape[1] - 1
            keep = max(1, round(n_patch * (1 - self.token_drop)))
            idx = torch.rand(hidden.shape[0], n_patch).argsort(dim=1)[:, :keep].sort(dim=1).values
            patches = torch.gather(
                hidden[:, 1:], 1, idx.unsqueeze(-1).expand(-1, -1, hidden.shape[2])
            )
            hidden = torch.cat([hidden[:, :1], patches], dim=1)
        replay = {
            "hidden": hidden.to(torch.float16),
            "labels": torch.stack([d["labels"] for d in docs]),
        }
        for k in ("bbox", "attention_mask", "input_ids"):
            if k in docs[0]:
                replay[k] = torch.stack([d[k] for d in docs])
        if "logits" in docs[0]:
            width = max(d["logits"].shape[0] for d in docs)
            replay["logits"] = torch.stack(
                [F.pad(d["logits"].float(), (0, width - d["logits"].shape[0])) for d in docs]
            )
            replay["logit_width"] = torch.tensor([d["logits"].shape[0] for d in docs])
        return replay

    def _sample_replay(self) -> dict[str, torch.Tensor] | None:
        if not self.store:
            return None
        return self._stack_replay([self.store[i] for i in self._sample_indices()], augment=True)

    def _replay_forward(self, replay: dict[str, torch.Tensor]):
        out = super()._replay_forward(replay)
        if self.latent_distill > 0 and "logits" in replay and out.loss is not None:
            tgt = replay["logits"].to(out.logits.device)
            width = replay["logit_width"].to(out.logits.device)
            c = min(out.logits.shape[-1], tgt.shape[-1])
            mask = (torch.arange(c, device=tgt.device)[None] < width[:, None]).float()
            mse = (((out.logits[:, :c].float() - tgt[:, :c]) ** 2) * mask).sum() / mask.sum()
            out.loss = out.loss + self.latent_distill * mse
        return out

    @torch.no_grad()
    def _bank_logits(self, docs: list[dict]) -> None:
        """Store each new sample's logits under the model that just learned its task."""
        was_training = self.model.training
        self.model.eval()
        try:
            for i in range(0, len(docs), self.ca_batch):
                chunk = docs[i : i + self.ca_batch]
                replay = {k: v for k, v in self._stack_replay(chunk).items() if k != "logits"}
                logits = super()._replay_forward(replay).logits.float().cpu()
                for d, row in zip(chunk, logits, strict=True):
                    d["logits"] = row.to(torch.float16)
        finally:
            if was_training:
                self.model.train()

    def _sample_indices(self) -> list[int]:
        n = min(self.replay_batch_size, len(self.store))
        if self.replay_balance != "class" or self.store[0]["labels"].dim() != 0:
            return random.sample(range(len(self.store)), n)
        by_cls: dict[int, list[int]] = {}
        for i, d in enumerate(self.store):
            by_cls.setdefault(int(d["labels"]), []).append(i)
        classes = list(by_cls)
        return [random.choice(by_cls[random.choice(classes)]) for _ in range(n)]

    def memory_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for d in self.store
            for t in d.values()
            if torch.is_tensor(t)
        )

    # ---------------------------------------------------------------- training
    def _make_optimizer(self) -> torch.optim.Optimizer:
        if self.trunk_lr_scale == 1.0:
            return super()._make_optimizer()
        lr = self.config.get("lr", 5e-5)
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        head = [p for n, p in named if "classifier" in n]
        trunk = [p for n, p in named if "classifier" not in n]
        return torch.optim.AdamW(
            [{"params": head, "lr": lr}, {"params": trunk, "lr": lr * self.trunk_lr_scale}],
            weight_decay=self.config.get("weight_decay", 0.01),
        )

    def _head(self) -> torch.nn.Linear:
        return self.model.model.classifier

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        if self.weight_align and self._seen_before > 0:
            self._weight_align(self._seen_before)
        super().after_task(task, train_loader)  # freeze map (task 0) + bank this task
        if self.ca_epochs > 0 and self.store:
            self._align_head()
        self._seen_before = self._head().out_features  # classes seen through this task

    @torch.no_grad()
    def _weight_align(self, n_old: int) -> None:
        w = self._head().weight
        if n_old >= w.shape[0]:
            return
        old_norm = w[:n_old].norm(dim=1).mean()
        new_norm = w[n_old:].norm(dim=1).mean().clamp(min=1e-8)
        w[n_old:] *= old_norm / new_norm

    @torch.no_grad()
    def _stored_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Classifier-input features of every stored latent under the current trunk."""
        was_training = self.model.training
        self.model.eval()
        feats: list[torch.Tensor] = []
        captured: list[torch.Tensor] = []
        hook = self._head().register_forward_pre_hook(lambda _m, a: captured.append(a[0]))
        try:
            for i in range(0, len(self.store), self.ca_batch):
                docs = self.store[i : i + self.ca_batch]
                captured.clear()
                with self._amp_autocast():
                    self._replay_forward(self._stack_replay(docs))
                x = captured[0]
                feats.append((x[:, 0] if x.dim() == 3 else x).float())
        finally:
            hook.remove()
            if was_training:
                self.model.train()
        labels = torch.stack([d["labels"] for d in self.store]).to(self.device)
        return torch.cat(feats), labels

    def _align_head(self) -> None:
        x, y = self._stored_features()
        head = self._head()
        opt = torch.optim.SGD(head.parameters(), lr=self.ca_lr, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.ca_epochs)
        # class-balanced sampling weights
        counts = torch.bincount(y, minlength=head.out_features).float()
        w = (1.0 / counts.clamp(min=1))[y]
        n = x.shape[0]
        for _ in range(self.ca_epochs):
            order = torch.multinomial(w, n, replacement=True)
            for i in range(0, n, self.ca_batch):
                idx = order[i : i + self.ca_batch]
                with torch.enable_grad():
                    loss = F.cross_entropy(head(x[idx]), y[idx])
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            sched.step()
        log.info(
            "colar_pp: head aligned on %d stored features (%d classes)", n, int((counts > 0).sum())
        )
