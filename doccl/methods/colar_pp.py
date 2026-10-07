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
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from doccl.methods.colar import CoLaR
from doccl.methods.frozen_ptm import RanPAC, cls_features
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


class LoRALinear(torch.nn.Module):
    """y = base(x) + (x Aᵀ Bᵀ)·(α/r); base frozen; B zero-init so the wrap is exact at start."""

    def __init__(self, base: torch.nn.Linear, r: int, alpha: float | None = None):
        super().__init__()
        self.base = base
        self.r = r
        self.scale = (alpha or r) / r
        self.lora_a = torch.nn.Parameter(
            torch.empty(r, base.in_features, device=base.weight.device)
        )
        self.lora_b = torch.nn.Parameter(
            torch.zeros(base.out_features, r, device=base.weight.device)
        )
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            self.base(x) + (x @ self.lora_a.T.to(x.dtype)) @ self.lora_b.T.to(x.dtype) * self.scale
        )

    @torch.no_grad()
    def merge_and_reset(self) -> None:
        self.base.weight += (self.lora_b @ self.lora_a).to(self.base.weight.dtype) * self.scale
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.lora_b.zero_()


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
        # Amendment 7 (RCA-driven): S1 analytic head re-solved on stored latents recomputed
        # through the CURRENT trunk; S2 feature anchoring of replayed latents.
        self.analytic_head = str(config.get("analytic_head", "none"))  # none | rp
        self.rp_fit = str(config.get("rp_fit", "stored+current"))  # stored | stored+current
        self.feature_anchor = float(config.get("feature_anchor", 0.0))
        self.dump_final = bool(config.get("dump_final", False))
        # Amendment 8. I1: drift-compensated full-data class Gaussians for head alignment;
        # I2: low-rank (LoRA) trunk adaptation after task 0. (I4 = storage switches above.)
        self.drift_comp = bool(config.get("drift_comp", False))
        self.ca_samples = int(config.get("ca_samples_per_cls", 256))
        self.trunk_adapt = str(config.get("trunk_adapt", "full"))  # full | lora
        self.lora_rank = int(config.get("lora_rank", 8))
        self.lora_lr = float(config.get("lora_lr", 5e-4))
        self._cls_mean: dict[int, torch.Tensor] = {}
        self._cls_cov: dict[int, torch.Tensor] = {}
        self._f_before: torch.Tensor | None = None
        self._rp = (
            RanPAC(model, {"rp_dim": config.get("rp_dim", 10000), "rp_seed": 0})
            if self.analytic_head == "rp"
            else None
        )
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
        if self.feature_anchor > 0:
            feats, _ = self._features_of(self.store[n_before:])
            for d, f in zip(self.store[n_before:], feats.cpu(), strict=True):
                d["feat"] = f.to(torch.float16)
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
        if "feat" in docs[0]:
            replay["feat"] = torch.stack([d["feat"] for d in docs])
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
        captured: list[torch.Tensor] = []
        anchor = self.feature_anchor > 0 and "feat" in replay and torch.is_grad_enabled()
        hook = (
            self._head().register_forward_pre_hook(lambda _m, a: captured.append(a[0]))
            if anchor
            else None
        )
        try:
            out = super()._replay_forward(replay)
        finally:
            if hook is not None:
                hook.remove()
        if anchor and captured and out.loss is not None:
            x = captured[0]
            cur = (x[:, 0] if x.dim() == 3 else x).float()
            ref = replay["feat"].to(cur.device).float()
            out.loss = out.loss + self.feature_anchor * (1 - F.cosine_similarity(cur, ref)).mean()
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
        if self.trunk_adapt == "lora" and any(
            isinstance(m, LoRALinear) for m in self.model.modules()
        ):
            lr = self.config.get("lr", 5e-5)
            named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
            lora = [p for n, p in named if "lora_" in n]
            rest = [p for n, p in named if "lora_" not in n]
            return torch.optim.AdamW(
                [{"params": rest, "lr": lr}, {"params": lora, "lr": self.lora_lr}],
                weight_decay=self.config.get("weight_decay", 0.01),
            )
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

    def _apply_freeze_map(self) -> None:
        super()._apply_freeze_map()
        if self.trunk_adapt != "lora":
            return
        for layer in self._encoder_layers()[self.split_layer_k :]:
            for p in layer.parameters():
                p.requires_grad = False
            for parent in list(layer.modules()):
                for name, child in list(parent.named_children()):
                    if isinstance(child, torch.nn.Linear):
                        setattr(parent, name, LoRALinear(child, self.lora_rank))
        log.info("colar_pp: LoRA r=%d on layers >= %d", self.lora_rank, self.split_layer_k)

    @torch.no_grad()
    def _merge_lora(self) -> None:
        for m in self.model.modules():
            if isinstance(m, LoRALinear):
                m.merge_and_reset()

    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        super().before_task(task, train_loader)
        # I1: features of every stored (old-task) latent under the trunk BEFORE this task.
        self._f_before = (
            self._stored_features()[0].cpu() if self.drift_comp and self.store else None
        )

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        if self.weight_align and self._seen_before > 0:
            self._weight_align(self._seen_before)
        if self.trunk_adapt == "lora":
            self._merge_lora()  # each task's update is rank-r; fresh adapters next task
        n_old = len(self.store)
        super().after_task(task, train_loader)  # freeze map (task 0) + bank this task
        if self.drift_comp:
            self._update_class_stats(train_loader, n_old)
            if self.ca_epochs > 0:
                self._align_head_gaussian()
        elif self.ca_epochs > 0 and self.store:
            self._align_head()
        if self._rp is not None and self.store:
            self._fit_analytic_head(train_loader)
        self._seen_before = self._head().out_features  # classes seen through this task
        if self.dump_final and task.is_last and getattr(self, "out_dir", None):
            out = Path(self.out_dir)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), out / "final_model.pt")
            torch.save(self.store, out / "final_store.pt")

    def _fit_analytic_head(self, train_loader: DataLoader) -> None:
        """S1: re-solve the RP ridge head from scratch on stored latents recomputed through
        the current trunk (+ the current task's full training set), class-balanced."""
        x, y = self._stored_features()
        if self.rp_fit == "stored+current":
            fc, yc = cls_features(self.model, train_loader, self.device)
            x, y = torch.cat([x, fc]), torch.cat([y, yc])
        counts = torch.bincount(y).float()
        weights = (counts.max() / counts.clamp(min=1))[y]  # every class carries equal mass
        self._rp.fit(x, y, weights)
        self.model.train()

    def evaluate(self, eval_loaders):
        if self._rp is None or not self._rp._fitted():
            return super().evaluate(eval_loaders)
        return RanPAC.evaluate(self._rp, eval_loaders)

    @torch.no_grad()
    def _update_class_stats(self, train_loader: DataLoader, n_old: int) -> None:
        """I1. (a) Shift every old class mean by the mean drift of its stored latents,
        measured exactly by re-running them through the trunk before/after this task
        (semantic-drift compensation with exact anchors). (b) Add full-data mean/cov for
        the classes of the task just learned."""
        if self._f_before is not None and n_old > 0:
            f_after, y_old = self._features_of(self.store[:n_old])
            delta = f_after.cpu() - self._f_before
            for c in y_old.unique().tolist():
                if c in self._cls_mean:
                    self._cls_mean[c] += delta[(y_old == c).cpu()].mean(0)
        f, y = cls_features(self.model, train_loader, self.device)
        f, y = f.cpu(), y.cpu()
        eye = torch.eye(f.shape[1])
        for c in y.unique().tolist():
            fc = f[y == c]
            self._cls_mean[c] = fc.mean(0)
            self._cls_cov[c] = torch.cov(fc.T) + 1e-4 * eye
        self.model.train()

    def _align_head_gaussian(self) -> None:
        """SLCA-style head alignment on samples from the drift-corrected class Gaussians."""
        xs, ys = [], []
        for c in sorted(self._cls_mean):
            try:
                dist = torch.distributions.MultivariateNormal(
                    self._cls_mean[c], covariance_matrix=self._cls_cov[c]
                )
            except (ValueError, RuntimeError):
                dist = torch.distributions.MultivariateNormal(
                    self._cls_mean[c], covariance_matrix=torch.diag(torch.diag(self._cls_cov[c]))
                )
            xs.append(dist.sample((self.ca_samples,)))
            ys.append(torch.full((self.ca_samples,), c, dtype=torch.long))
        x, y = torch.cat(xs).to(self.device), torch.cat(ys).to(self.device)
        self._train_head(x, y)

    @torch.no_grad()
    def _weight_align(self, n_old: int) -> None:
        w = self._head().weight
        if n_old >= w.shape[0]:
            return
        old_norm = w[:n_old].norm(dim=1).mean()
        new_norm = w[n_old:].norm(dim=1).mean().clamp(min=1e-8)
        w[n_old:] *= old_norm / new_norm

    def _stored_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Classifier-input features of every stored latent under the current trunk."""
        return self._features_of(self.store)

    @torch.no_grad()
    def _features_of(self, docs_all: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        was_training = self.model.training
        self.model.eval()
        feats: list[torch.Tensor] = []
        captured: list[torch.Tensor] = []
        hook = self._head().register_forward_pre_hook(lambda _m, a: captured.append(a[0]))
        try:
            for i in range(0, len(docs_all), self.ca_batch):
                docs = docs_all[i : i + self.ca_batch]
                captured.clear()
                with self._amp_autocast():
                    self._replay_forward(self._stack_replay(docs))
                x = captured[0]
                feats.append((x[:, 0] if x.dim() == 3 else x).float().clone())  # drop the view
        finally:
            hook.remove()
            if was_training:
                self.model.train()
        labels = torch.stack([d["labels"] for d in docs_all]).to(self.device)
        return torch.cat(feats), labels

    def _align_head(self) -> None:
        x, y = self._stored_features()
        self._train_head(x, y)

    def _train_head(self, x: torch.Tensor, y: torch.Tensor) -> None:
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
        log.info("colar_pp: head aligned on %d features (%d classes)", n, int((counts > 0).sum()))
