"""SpectralMemory — SLR-minimal (Spectral Latent Replay, falsification-chain → method).

The proposed core of SLR. Where ``latent_replay`` banks raw per-document layer-``k``
activations, SpectralMemory banks, per task, a **shared rank-``r`` forgetting-subspace
basis** plus **per-class Gaussians expressed in that basis**, plus a handful of **carrier
skeletons** — the ``bbox``/``attention_mask``/``labels`` of a few real documents, but NOT
their activations. At replay, each carrier token's feature is synthesised from *its own
class* Gaussian in the subspace and draped over the carrier, then injected at layer ``k``
through the inherited pre-hook.

The razor (see ``CL4IE/wiki/ideas/2026-07-10-spectral-latent-replay-icml.md``):

- SpectralMemory indexes memory **by the forgetting subspace** — the top-``r`` right
  singular vectors of the centred activation matrix, the low-rank directions the NTK
  theorem (arXiv 2606.18024) says forgetting concentrates in. Class structure lives
  *inside* that subspace (per-class coordinates), not as ``d``-dimensional per-class means.
- ``aglr_replay`` (AGLR-CL) indexes memory **by class** — full ``d``-dimensional Gaussians
  per label, no shared low-rank basis. The Gate-0 comparison is exactly this basis-first
  vs class-first factorisation behind the same replay hook.

Label-coupling is load-bearing: the replay feature for a token MUST come from that token's
class, or feature and label decorrelate and the replayed head-gradient teaches nothing (a
label-agnostic first version collapsed to naive-level AA — see git history / STATE.md).

Memory per task is ``O(r · d)`` (basis) + ``O(C · r)`` (per-class in-subspace stats) + a
few carrier skeletons — versus ``O(N · seq · d)`` raw activations (≈36× smaller at r=16 on
LayoutLMv3). The compression is the contribution; Gate 1 sweeps ``r`` for the Pareto front.

Validity: everything below layer ``k`` is frozen after task 0 (inherited freeze map), so
the banked statistics describe a distribution that never drifts — the property that killed
every plastic-space memory in the chain (LexSlot, LexMem v1–v5).

``rank_r: 0`` disables the fit (degenerate control; no replay).
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from doccl.methods.latent_replay import LatentReplay
from doccl.types import TaskInfo

log = logging.getLogger(__name__)

__all__ = ["SpectralMemory"]

_SVD_MAX_TOKENS = 20_000  # cap the activation matrix rows fed to torch.linalg.svd


class SpectralMemory(LatentReplay):
    """Per-task rank-``r`` spectral latent replay at encoder layer ``split_layer_k``."""

    name = "spectral_memory"

    def __init__(self, model, config):
        config = dict(config)
        # Raw-doc banking is off; we synthesise activations from the spectral model.
        config.setdefault("docs_per_task", 0)
        super().__init__(model, config)
        self.rank_r = int(config.get("rank_r", 16))
        self.carriers_per_task = int(config.get("carriers_per_task", 4))
        self.d = int(self.model.hidden_size)
        # Width of the hidden entering layer k. For LayoutLMv3 this is text + image-patch
        # tokens (e.g. 512 + 197 = 709), wider than the text attention_mask; the injected
        # replay hidden must match it (the pre-hook replaces the whole layer-k input).
        # Learned at capture time.
        self._hidden_width: int | None = None
        # task_id -> {"mean": (d,), "basis": (r, d),
        #             "classes": {class_id: {"cmean": (r,), "cstd": (r,)}},
        #             "carriers": [skeleton]}
        self._spectral: dict[int, dict] = {}

    # ── capture (all tokens for the fit; a few carriers for the scaffold) ──────

    @torch.no_grad()
    def _capture_all_tokens(self, loader: DataLoader):
        """Push the loader through layer ``k`` and collect masked token features WITH their
        labels, plus a small set of carrier skeletons (real bbox/mask/labels, no activations).

        Labels are captured so the spectral model can be **class-conditional**: the replay
        feature for a carrier token must be drawn from that token's class, or feature and
        label decorrelate and the replayed head-gradient teaches nothing (the bug the first
        label-agnostic version hit — AA collapsed to naive levels).

        Reuses the inherited ``_capture`` flag + pre-hook: setting ``self._capture`` to a
        list makes ``_pre_hook`` append the layer-``k`` input instead of injecting.
        """
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
                hidden = self._capture[0].float()  # (b, seq, d) on CPU fp16 → fp32
            finally:
                self._capture = None
            if self._hidden_width is None:
                self._hidden_width = hidden.shape[1]  # e.g. 709 for LayoutLMv3
            labels = batch["labels"].cpu()
            am = batch["attention_mask"].bool().cpu()
            valid = (labels != -100) & am  # (b, seq_text)
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
                feats.append(text_hidden[i][m])  # (n_i, d) — real (non-pad, labelled) tokens
                labs.append(labels[i][m])
                n += int(m.sum())
            if n >= 2 * _SVD_MAX_TOKENS:
                break
        if was_training:
            self.model.train()
        x = torch.cat(feats) if feats else torch.empty(0, self.d)
        y = torch.cat(labs) if labs else torch.empty(0, dtype=torch.long)
        if x.shape[0] > _SVD_MAX_TOKENS:
            keep = torch.randperm(x.shape[0])[:_SVD_MAX_TOKENS]
            x, y = x[keep], y[keep]
        return x, y, carriers

    def _fit_spectral(self, task: TaskInfo, loader: DataLoader) -> None:
        """Fit a rank-``r`` forgetting-subspace basis (shared, the SLR identity) plus
        **per-class** low-rank Gaussians expressed in that basis (label-coupled, so replay
        teaches the head). Memory: O(r·d) for the basis + O(C·r) for the per-class stats.
        """
        if self.rank_r <= 0:
            return
        x, y, carriers = self._capture_all_tokens(loader)
        if x.shape[0] < 2 or not carriers:
            log.warning(
                "spectral_memory: task %d — too few tokens/carriers, skipping fit", task.task_id
            )
            return
        mean = x.mean(0)  # (d,)
        xc = x - mean
        r = min(self.rank_r, xc.shape[1], xc.shape[0])
        # Top-r right singular vectors = the forgetting subspace (the low-rank directions
        # the NTK theorem says forgetting concentrates in). This basis is the SLR identity.
        _, _, vh = torch.linalg.svd(xc, full_matrices=False)
        basis = vh[:r]  # (r, d), orthonormal rows
        proj = xc @ basis.T  # (n, r) — tokens in the subspace
        # Per-class stats IN the subspace: mean coordinate + per-direction std.
        classes: dict[int, dict] = {}
        for c in y.unique().tolist():
            pc = proj[y == c]
            if pc.shape[0] < 1:
                continue
            classes[int(c)] = {
                "cmean": pc.mean(0),  # (r,)
                "cstd": (pc.std(0) if pc.shape[0] > 1 else torch.ones(r)).clamp(min=1e-3),  # (r,)
            }
        # Stack per-class stats into (C, r) tensors + a label→row map so replay is a single
        # gather + matmul instead of a per-token Python loop (57ms→sub-ms at width 709).
        cids = sorted(classes.keys())
        cmean_stack = torch.stack([classes[c]["cmean"] for c in cids]) if cids else torch.empty(0, r)
        cstd_stack = torch.stack([classes[c]["cstd"] for c in cids]) if cids else torch.empty(0, r)
        self._spectral[task.task_id] = {
            "mean": mean.cpu(),
            "basis": basis.cpu(),
            "classes": {c: {k: v.cpu() for k, v in s.items()} for c, s in classes.items()},
            "cmean_stack": cmean_stack.cpu(),  # (C, r)
            "cstd_stack": cstd_stack.cpu(),  # (C, r)
            "label_to_row": {c: i for i, c in enumerate(cids)},  # class_id → row in stacks
            "carriers": carriers,
        }
        log.info(
            "spectral_memory: task %d — rank-%d basis + %d class-Gaussians (%d tokens, %d carriers); "
            "bank now %d tasks, ~%.3f MB",
            task.task_id,
            r,
            len(classes),
            x.shape[0],
            len(carriers),
            len(self._spectral),
            self.memory_bytes() / 1e6,
        )

    # ── replay reconstruction ─────────────────────────────────────────────────

    def _sample_replay(self) -> dict[str, torch.Tensor] | None:
        """Reconstruct a **label-coupled** replay batch: for each carrier token with label
        ``c``, draw its feature from class ``c``'s Gaussian expressed in the forgetting
        subspace (``mean + (cmean + N(0,cstd²)) @ basis``). Feature and label stay coupled,
        so the replayed head-gradient teaches the old task; image-patch positions (no label)
        draw from any class and are inert for the loss."""
        keys = [t for t, v in self._spectral.items() if v["classes"]]
        if not keys:
            return None
        b = self.replay_batch_size
        width = (
            self._hidden_width
            if self._hidden_width is not None
            else self._spectral[keys[0]]["carriers"][0]["bbox"].shape[0]
        )
        hiddens, bboxes, masks, labels_out = [], [], [], []
        for _ in range(b):
            tk = keys[torch.randint(len(keys), ()).item()]
            model = self._spectral[tk]
            basis, gmean = model["basis"], model["mean"]  # (r,d), (d,)
            cmean_stack, cstd_stack = model["cmean_stack"], model["cstd_stack"]  # (C,r),(C,r)
            l2r, n_cls = model["label_to_row"], cmean_stack.shape[0]
            carrier = model["carriers"][torch.randint(len(model["carriers"]), ()).item()]
            lab = carrier["labels"]  # (text_len,), may contain -100
            r = basis.shape[0]
            # Map each of `width` positions to a class row: real label → its row; -100 /
            # image-patch / unseen label → a random class row (inert for the loss).
            rows = torch.randint(n_cls, (width,))
            for t in range(min(width, lab.shape[0])):
                row = l2r.get(int(lab[t].item()))
                if row is not None:
                    rows[t] = row
            # Vectorised sample in the subspace, then one matmul to feature space.
            z = cmean_stack[rows] + torch.randn(width, r) * cstd_stack[rows]  # (width, r)
            hidden = gmean + z @ basis  # (width, d)
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
        # Freeze map on task 0 (inherited), then fit this task's spectral model on the
        # frozen-trunk activations. Skip the raw-doc capture (docs_per_task=0).
        if task.task_id == 0:
            self._apply_freeze_map()
        self._fit_spectral(task, train_loader)

    def memory_bytes(self) -> int:
        """Total banked footprint in bytes — the number the Pareto curve plots.

        Per task: the shared basis + global mean (fp32) + per-class in-subspace stats
        (cmean + cstd, fp32) + carrier skeletons (int64 layout/labels)."""
        total = 0
        for m in self._spectral.values():
            total += (m["mean"].numel() + m["basis"].numel()) * 4
            total += sum(
                (s["cmean"].numel() + s["cstd"].numel()) * 4 for s in m["classes"].values()
            )
            total += sum(
                (c["bbox"].numel() + c["attention_mask"].numel() + c["labels"].numel()) * 8
                for c in m["carriers"]
            )
        return total
