"""LCA — Local Classifier Alignment for Continual Learning (ICLR 2026), ported to doc-IE.

Faithful port of github.com/tungts1101/LCA (Tran, Vargas, Than) to per-token document
information extraction (LayoutLMv3 token classification). LCA's recipe, per task t:

  1. Fine-tune the backbone + classifier head on task t (SGD + cosine).
  2. Estimate, for every class, a feature **Gaussian** (mean μ_c, covariance Σ_c) over the
     backbone's features (here: per-BIO-class over TOKEN features).
  3. **TIES-merge** the per-task backbone parameter sets onto the base (``ties_merge``,
     verbatim from LCA `helper.merge`, lamb=1.0, topk=100).
  4. **Align** (skipped on task 0): sample synthetic features from each class's N(μ_c, Σ_c)
     and re-train the **classifier head** on them with CE + ``robust_weight``·(intra-class
     pairwise loss-variance) + ``entropy_weight``·(entropy). This re-aligns the merged
     classifier to the merged backbone's feature distribution — generative feature replay
     through the head, NO raw exemplars stored.

The novelty vs plain merging is step 4: LCA shows (and our DocMERGE RCA independently found)
that merging the backbone alone leaves the classifier mis-aligned; the alignment step fixes
it. term2/term3 are the paper's "robust"/entropy regularizers on the sampled-feature CE.

Differences from the image-CIL original, all faithful in spirit:
  * features are per-TOKEN (BIO labels), so μ_c/Σ_c are estimated over scored tokens
    (labels != -100), excluding the dominant 'O' class from sampling is optional;
  * a single GROWING head (our CIL remapper) replaces LCA's per-task heads — the align step
    retrains that one head, which is exactly LCA's classifier alignment.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F  # noqa: N812 — canonical torch alias (repo-wide)
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader
from tqdm import tqdm

from doccl.eval.metrics import compute_token_f1
from doccl.methods.naive import NaiveFineTune
from doccl.methods.ties_merge import merge_state_dicts
from doccl.types import EvalMetrics, TaskInfo, TrainMetrics

log = logging.getLogger(__name__)

__all__ = ["LCA"]


class LCA(NaiveFineTune):
    """Local Classifier Alignment (see module docstring). Standard-forward CL method."""

    name = "lca"

    def __init__(self, model, config):
        super().__init__(model, config)
        # LCA training recipe (paper defaults).
        self.base_lr = float(config.get("lr", 1e-2))
        self.weight_decay = float(config.get("weight_decay", 5e-4))
        self.epochs = int(config.get("epochs", 10))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        # Merge (LCA helper.merge defaults).
        self.merge_method = config.get("merge_method", "ties")
        self.merge_coef = float(config.get("merge_coef", 1.0))
        self.merge_topk = int(config.get("merge_topk", 100))
        # Classifier-alignment (CA) defaults.
        self.ca_lr = float(config.get("ca_lr", 5e-3))
        self.ca_epochs = int(config.get("ca_epochs", 10))
        self.ca_batch_size = int(config.get("ca_batch_size", 64))
        self.ca_samples_per_cls = int(config.get("ca_samples_per_cls", 256))
        self.ca_robust_weight = float(config.get("ca_robust_weight", 0.1))
        self.ca_entropy_weight = float(config.get("ca_entropy_weight", 0.0))
        self.ca_feature_n_batches = int(config.get("ca_feature_n_batches", 16))
        self.ca_skip_O = bool(config.get("ca_skip_O", True))  # exclude 'O' from align

        self._task_idx = -1
        self._base_backbone: dict[str, torch.Tensor] | None = None  # θ_base (LMC anchor)
        self._task_backbones: list[dict[str, torch.Tensor]] = []  # per-task θ_t (backbone)
        # Per-class feature Gaussians, keyed by class index (grows with the head).
        self._class_means: dict[int, torch.Tensor] = {}
        self._class_covs: dict[int, torch.Tensor] = {}

    # ─── helpers: which params are "backbone" (everything but the classifier head) ──
    def _backbone_state(self) -> dict[str, torch.Tensor]:
        return {
            n: p.detach().clone().cpu()
            for n, p in self.model.model.named_parameters()
            if "classifier" not in n
        }

    @property
    def _head(self) -> torch.nn.Linear:
        return self.model.model.classifier

    # ─── lifecycle ──────────────────────────────────────────────────────────────
    def before_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        self._task_idx = task.task_id
        if self._base_backbone is None:
            # θ_base captured once (after the first task's head expansion the loop did).
            self._base_backbone = self._backbone_state()

    def train_task(
        self, task: TaskInfo, train_loader: DataLoader, val_loader: DataLoader | None = None
    ) -> TrainMetrics:
        """Fine-tune backbone + head on task t with LCA's SGD + cosine recipe."""
        self.model.train()
        params = [p for p in self.model.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=self.base_lr, momentum=0.9, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6
        )
        stopper = self.make_early_stopper(val_loader)
        total_loss, n_steps = 0.0, 0
        for epoch in range(self.epochs):
            self.model.train()
            pbar = tqdm(train_loader, desc=f"lca T{task.task_id} ep{epoch+1}", leave=False)
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                optimizer.zero_grad()
                out = self.model(
                    input_ids=batch["input_ids"],
                    bbox=batch["bbox"],
                    pixel_values=batch.get("pixel_values"),
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                )
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                optimizer.step()
                total_loss += float(loss.item())
                n_steps += 1
            scheduler.step()
            if (
                stopper.enabled
                and val_loader is not None
                and stopper.step(self.current_task_val_f1(val_loader), self.model, epoch)
            ):
                break
        stopper.restore_best(self.model)
        return TrainMetrics(
            task_id=task.task_id, loss=total_loss / max(n_steps, 1), n_steps=n_steps
        )

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        # (1) class feature Gaussians for the classes seen this task.
        self._compute_class_stats(train_loader)
        # (2) stash this task's backbone, then TIES-merge all task backbones onto base.
        self._task_backbones.append(self._backbone_state())
        self._merge_backbones()
        # (3) align the classifier on sampled features (LCA skips task 0).
        if task.task_id > 0 and self._class_means:
            self._align()
        log.info(
            "lca: task %d done; merged %d backbones; classes with stats=%d",
            task.task_id,
            len(self._task_backbones),
            len(self._class_means),
        )

    # ─── (1) per-class feature Gaussians over TOKEN features ─────────────────────
    @torch.no_grad()
    def _compute_class_stats(self, loader: DataLoader) -> None:
        """Estimate (μ_c, Σ_c) over token features for each class present in this task."""
        self.model.eval()
        feats_by_cls: dict[int, list[torch.Tensor]] = {}
        for bi, batch in enumerate(loader):
            if bi >= self.ca_feature_n_batches:
                break
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            feats = self.model.token_features(batch)  # (B, L, D)
            labels = batch["labels"][:, : feats.shape[1]]
            mask = labels != -100
            f = feats[mask].cpu()  # (n_tok, D)
            y = labels[mask].cpu()  # (n_tok,)
            for c in y.unique().tolist():
                feats_by_cls.setdefault(c, []).append(f[y == c])
        dim = self.model.hidden_size
        for c, chunks in feats_by_cls.items():
            x = torch.cat(chunks, dim=0)
            if x.shape[0] < 2:  # need ≥2 tokens for a covariance
                continue
            mean = x.mean(dim=0)
            cov = torch.cov(x.T) + torch.eye(dim) * 1e-4
            self._class_means[c] = mean
            self._class_covs[c] = cov

    # ─── (2) TIES-merge the per-task backbones ──────────────────────────────────
    @torch.no_grad()
    def _merge_backbones(self) -> None:
        if self._base_backbone is None or not self._task_backbones:
            return
        merged = merge_state_dicts(
            self._base_backbone,
            self._task_backbones,
            method=self.merge_method,
            lamb=self.merge_coef,
            topk=self.merge_topk,
        )
        own = dict(self.model.model.named_parameters())
        for name, val in merged.items():
            if name in own:
                own[name].data.copy_(val.to(own[name].device))

    # ─── (3) classifier alignment on sampled Gaussian features ──────────────────
    def _align(self) -> None:
        """Re-train the classifier head on synthetic features ~ N(μ_c, Σ_c) (LCA align)."""
        classes = [c for c in sorted(self._class_means) if not (self.ca_skip_O and c == 0)]
        if not classes:
            return
        data, labels = [], []
        for c in classes:
            mean = self._class_means[c].to(self.device).float()
            cov = self._class_covs[c].to(self.device).float()
            try:
                dist = MultivariateNormal(mean, covariance_matrix=cov)
            except (ValueError, RuntimeError):  # non-PD cov → diagonal fallback
                dist = MultivariateNormal(mean, covariance_matrix=torch.diag(torch.diag(cov)))
            data.append(dist.sample((self.ca_samples_per_cls,)))
            labels.extend([c] * self.ca_samples_per_cls)
        data = torch.cat(data, dim=0).float().to(self.device)
        labels = torch.tensor(labels, dtype=torch.long, device=self.device)
        means_stack = torch.stack([self._class_means[c] for c in sorted(self._class_means)]).to(
            self.device
        )
        cls_order = sorted(self._class_means)

        head = self._head
        for p in head.parameters():
            p.requires_grad = True
        optimizer = torch.optim.SGD(
            head.parameters(), lr=self.ca_lr, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.ca_epochs)
        head.train()
        for _ in range(self.ca_epochs):
            perm = torch.randperm(data.shape[0], device=self.device)
            data, labels = data[perm], labels[perm]
            for i in range(0, data.shape[0], self.ca_batch_size):
                x = data[i : i + self.ca_batch_size]
                y = labels[i : i + self.ca_batch_size]
                logits = head(x)
                loss = self._align_loss(logits, y, x, means_stack, cls_order)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

    def _align_loss(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        x: torch.Tensor,
        means_stack: torch.Tensor,
        cls_order: list[int],
    ) -> torch.Tensor:
        """LCA align loss: CE + robust·(intra-class pairwise loss var) + entropy·(entropy)."""
        loss_vec = F.cross_entropy(logits, y, reduction="none")
        if self.ca_robust_weight == 0 and self.ca_entropy_weight == 0:
            return loss_vec.mean()
        # cluster = label AND nearest-mean agree (LCA).
        dist = torch.cdist(x, means_stack)
        nearest = torch.tensor([cls_order[i] for i in dist.argmin(dim=1).tolist()], device=x.device)
        total = torch.zeros((), device=x.device)
        uniq = y.unique()
        for c in uniq:
            label_mask = y == c
            cluster_mask = (nearest == c) & label_mask
            if cluster_mask.any():
                cl = loss_vec[cluster_mask]
                term1 = cl.mean()
                if cl.numel() >= 2:
                    pd = torch.abs(cl.unsqueeze(1) - cl.unsqueeze(0))
                    eye = ~torch.eye(cl.numel(), dtype=torch.bool, device=x.device)
                    term2 = pd[eye].mean()
                else:
                    term2 = torch.zeros((), device=x.device)
                if self.ca_entropy_weight != 0:
                    p = F.softmax(logits[cluster_mask], dim=1)
                    term3 = (-(p * torch.log(p + 1e-8)).sum(dim=1)).mean()
                else:
                    term3 = torch.zeros((), device=x.device)
            else:
                term1 = loss_vec[label_mask].mean()
                term2 = term3 = torch.zeros((), device=x.device)
            total = total + term1 + self.ca_robust_weight * term2 + self.ca_entropy_weight * term3
        return total / max(len(uniq), 1)

    # ─── evaluate (standard forward — single growing head) ──────────────────────
    def evaluate(self, eval_loaders: dict[int, DataLoader]) -> dict[int, EvalMetrics]:
        self.model.eval()
        results: dict[int, EvalMetrics] = {}
        id_to_label = getattr(self.model, "id_to_label", None) or {
            i: str(i) for i in range(self.model.model.config.num_labels)
        }
        with torch.no_grad():
            for tid, loader in eval_loaders.items():
                preds_all, labels_all = [], []
                for batch in loader:
                    batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                    out = self.model(
                        input_ids=batch["input_ids"],
                        bbox=batch["bbox"],
                        pixel_values=batch.get("pixel_values"),
                        attention_mask=batch.get("attention_mask"),
                    )
                    logits = out.logits
                    preds = logits.argmax(dim=-1)
                    labels = batch["labels"][:, : logits.shape[1]]
                    mask = labels != -100
                    preds_all.extend(preds[mask].cpu().tolist())
                    labels_all.extend(labels[mask].cpu().tolist())
                m = compute_token_f1(preds_all, labels_all, id_to_label)
                results[tid] = EvalMetrics(
                    task_id=tid,
                    f1=m["f1"],
                    precision=m["precision"],
                    recall=m["recall"],
                    n_samples=len(labels_all),
                )
        return results
