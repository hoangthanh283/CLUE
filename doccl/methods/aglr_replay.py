"""AGLRReplay — a doc-IE port of AGLR-CL (arXiv 2505.08524), the Gate-0 comparator.

AGLR-CL ("Attention-based Generative Latent Replay") is the single nearest prior art to
SLR: frozen-feature generative latent replay via **per-class Gaussians**, with an
**attention-salience filter** on which token embeddings feed the fit, validated on
whole-slide-image classification. This ports that recipe faithfully into the same
frozen-trunk / layer-``k`` replay hook SLR uses, so the Gate-0 comparison varies ONLY the
memory representation:

- ``aglr_replay`` (this): memory indexed **by class** — a diagonal Gaussian per
  (task, label) in layer-``k`` activation space; replayed by drawing, for each carrier
  token, from the Gaussian of that token's label.
- ``spectral_memory`` (SLR): memory indexed **by the forgetting subspace** — rank-``r``
  SVD factors of the pooled activation matrix; replayed by sampling in the subspace.

If SLR cannot beat this comparator at equal-or-lower bytes, SLR *is* AGLR-CL and there is
no paper (the razor). Everything else — freeze map, capture/inject hook, CL loop — is
inherited from ``LatentReplay`` unchanged.

Attention-salience filter (``attn_filter: true``): AGLR-CL keeps only the most salient
patch embeddings. The doc-IE analogue keeps the top-fraction tokens by hidden-state
L2 norm within each document (a content-salience proxy available without exposing the
attention maps of the frozen encoder), so the Gaussians are fit on informative tokens.

``attn_keep: 1.0`` disables the filter (all masked tokens used).
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from doccl.methods.latent_replay import LatentReplay
from doccl.types import TaskInfo

log = logging.getLogger(__name__)

__all__ = ["AGLRReplay"]

_VAR_FLOOR = 1e-4  # shrinkage floor on per-dim variances (matches gauss_replay)
_FIT_MAX_TOKENS = 20_000


class AGLRReplay(LatentReplay):
    """Per-(task×class) Gaussian latent replay at encoder layer ``split_layer_k``."""

    name = "aglr_replay"

    def __init__(self, model, config):
        config = dict(config)
        config.setdefault("docs_per_task", 0)  # synthesise, don't bank raw docs
        super().__init__(model, config)
        self.carriers_per_task = int(config.get("carriers_per_task", 4))
        self.attn_keep = float(config.get("attn_keep", 0.5))  # top-fraction salient tokens
        self.d = int(self.model.hidden_size)
        # Width of the layer-k hidden (text + image patches for LayoutLMv3); the injected
        # replay hidden must match it. Learned at capture.
        self._hidden_width: int | None = None
        # task_id -> {"gaussians": {class_id: {"mean","var"}}, "carriers": [skeleton]}
        self._bank: dict[int, dict] = {}

    # ── capture + salience filter + fit ───────────────────────────────────────

    def _fit_gaussians(self, task: TaskInfo, loader: DataLoader) -> None:
        x, y, carriers = self._capture_with_labels(loader)
        if x.shape[0] < 2 or not carriers:
            log.warning(
                "aglr_replay: task %d — too few tokens/carriers, skipping fit", task.task_id
            )
            return
        if x.shape[0] > _FIT_MAX_TOKENS:
            keep = torch.randperm(x.shape[0])[:_FIT_MAX_TOKENS]
            x, y = x[keep], y[keep]
        gaussians: dict[int, dict] = {}
        for c in y.unique().tolist():
            xc = x[y == c]
            if xc.shape[0] < 2:
                continue
            gaussians[int(c)] = {
                "mean": xc.mean(0),
                "var": xc.var(0).clamp(min=_VAR_FLOOR),
            }
        # Stack per-class (mean, std) into (C, d) tensors + a label→row map so replay is a
        # single gather + broadcast instead of a per-token Python loop.
        cids = sorted(gaussians.keys())
        mean_stack = (
            torch.stack([gaussians[c]["mean"] for c in cids]) if cids else torch.empty(0, self.d)
        )
        std_stack = (
            torch.stack([gaussians[c]["var"].sqrt() for c in cids])
            if cids
            else torch.empty(0, self.d)
        )
        self._bank[task.task_id] = {
            "gaussians": gaussians,
            "mean_stack": mean_stack,  # (C, d)
            "std_stack": std_stack,  # (C, d)
            "label_to_row": {c: i for i, c in enumerate(cids)},
            "carriers": carriers,
        }
        log.info(
            "aglr_replay: task %d — %d class-Gaussians (%d tokens, keep=%.2f, %d carriers); "
            "bank now %d tasks, ~%.3f MB",
            task.task_id,
            len(gaussians),
            x.shape[0],
            self.attn_keep,
            len(carriers),
            len(self._bank),
            self.memory_bytes() / 1e6,
        )

    @torch.no_grad()
    def _capture_with_labels(self, loader: DataLoader):
        """Single-pass capture returning (features, labels, carriers) with the salience
        filter applied per document. Kept separate from ``_salient`` so labels stay aligned
        with the kept feature rows."""
        was_training = self.model.training
        self.model.eval()
        feats: list[torch.Tensor] = []
        labs: list[torch.Tensor] = []
        carriers: list[dict[str, torch.Tensor]] = []
        n = 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
            if "pixel_values" in batch:
                self._pixel_shape = tuple(batch["pixel_values"].shape[1:])
            self._capture = []
            try:
                self.model(**batch)
                hidden = self._capture[0].float()  # (b, seq, d)
            finally:
                self._capture = None
            if self._hidden_width is None:
                self._hidden_width = hidden.shape[1]  # e.g. 709 for LayoutLMv3 (text + patches)
            labels = batch["labels"].cpu()
            am = batch["attention_mask"].bool().cpu()
            valid = (labels != -100) & am
            text_len = am.shape[1]
            text_hidden = hidden[:, :text_len, :]  # drop image-patch positions for the fit
            for i in range(hidden.shape[0]):
                if len(carriers) < self.carriers_per_task:
                    carriers.append(
                        {
                            "bbox": batch["bbox"][i].cpu(),
                            "attention_mask": batch["attention_mask"][i].cpu(),
                            "labels": labels[i],
                        }
                    )
                m = valid[i]
                f = text_hidden[i][m]
                lab = labels[i][m]
                if f.shape[0] == 0:
                    continue
                k = max(1, int(round(self.attn_keep * f.shape[0])))
                if k < f.shape[0]:
                    idx = torch.topk(f.norm(dim=1), k).indices
                    f, lab = f[idx], lab[idx]
                feats.append(f)
                labs.append(lab)
                n += f.shape[0]
            if n >= 2 * _FIT_MAX_TOKENS:
                break
        if was_training:
            self.model.train()
        x = torch.cat(feats) if feats else torch.empty(0, self.d)
        y = torch.cat(labs) if labs else torch.empty(0, dtype=torch.long)
        return x, y, carriers

    # ── replay reconstruction ─────────────────────────────────────────────────

    def _sample_replay(self) -> dict[str, torch.Tensor] | None:
        """Reconstruct ``[b, seq, d]`` hiddens: for each carrier token, draw from the
        class-Gaussian of that token's label (fallback: nearest available class)."""
        keys = [t for t, v in self._bank.items() if v["gaussians"]]
        if not keys:
            return None
        b = self.replay_batch_size
        # Reconstruct at the captured layer-k width; text positions are keyed by the carrier
        # labels, image-patch positions (beyond the label length) draw from any class and are
        # inert for the loss (the model's downstream slice keeps only text-position logits).
        hiddens, bboxes, masks, labels_out = [], [], [], []
        for _ in range(b):
            tk = keys[torch.randint(len(keys), ()).item()]
            entry = self._bank[tk]
            carrier = entry["carriers"][torch.randint(len(entry["carriers"]), ()).item()]
            mean_stack, std_stack = entry["mean_stack"], entry["std_stack"]  # (C,d),(C,d)
            l2r, n_cls = entry["label_to_row"], mean_stack.shape[0]
            lab = carrier["labels"]  # (text_len,), may contain -100
            w = self._hidden_width if self._hidden_width is not None else lab.shape[0]
            rows = torch.randint(n_cls, (w,))
            for t in range(min(w, lab.shape[0])):
                row = l2r.get(int(lab[t].item()))
                if row is not None:
                    rows[t] = row
            # Vectorised: gather per-token class stats, one broadcast sample.
            hidden = mean_stack[rows] + std_stack[rows] * torch.randn(w, self.d)  # (w, d)
            hiddens.append(hidden.to(torch.float16))
            bboxes.append(carrier["bbox"])
            masks.append(carrier["attention_mask"])
            labels_out.append(lab)
        return {
            "hidden": torch.stack(hiddens),
            "bbox": torch.stack(bboxes),
            "attention_mask": torch.stack(masks),
            "labels": torch.stack(labels_out),
        }

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        if task.task_id == 0:
            self._apply_freeze_map()
        self._fit_gaussians(task, train_loader)

    def memory_bytes(self) -> int:
        total = 0
        for entry in self._bank.values():
            for g in entry["gaussians"].values():
                total += (g["mean"].numel() + g["var"].numel()) * 4
            total += sum(
                (c["bbox"].numel() + c["attention_mask"].numel() + c["labels"].numel()) * 8
                for c in entry["carriers"]
            )
        return total
