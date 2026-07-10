"""CoresetMemory — the one feature-replay variant not falsified at Gate 0.

Gate 0 (2026-07-10) killed both generative feature-replay variants for doc-IE:
SpectralMemory (rank-`r` subspace synthesis, AA 41.9) and AGLRReplay (full-`d` per-class
Gaussian synthesis, AA 39.4) both collapse (SROIE final F1 ~3), while real-activation
replay (latent_replay) grounds the head to AA 87.3 at the same frozen boundary. Diagnosis:
a per-class *marginal* Gaussian — even full-`d` — discards the joint feature structure real
activations carry, so the head can't hold its boundary against the trunk's real new-task
features.

CoresetMemory attacks that root cause directly: instead of *synthesising* features from a
fitted distribution, it stores a **small coreset of REAL layer-`k` activation vectors** —
k-means centroids per (task, class). Centroids are actual points in feature space, so they
preserve joint structure (a centroid is a real co-occurrence of the 768 dims, not a product
of marginals). This is iCaRL/coreset selection lifted into the frozen latent space. Replay
draws real centroids draped over carrier skeletons and injects at layer `k`.

Memory: `m` centroids × `d` × #classes × #tasks (real fp16 vectors). At m=8 on LayoutLMv3
that is ~0.5 MB — comparable bytes to the Gaussian variants, but every stored vector is a
real activation. The test: does storing FEW REAL feature modes (vs a fitted marginal) let
the head survive? If yes → a genuine buffer-free-ish method. If no → generative *and*
coreset feature replay both fail → the falsification chain is complete and airtight.

Subclasses ``AGLRReplay`` purely to reuse its (correct) ``_capture_with_labels`` — the
frozen-boundary capture, layer-`k` width handling, salience filter, and carrier logic are
identical. Only the fit (k-means, not Gaussians) and reconstruction (real centroids, not
sampled) differ.

``centroids_per_class: 1`` degenerates to class-mean replay (a real single-point coreset);
higher values give the head multiple real modes per class.
"""

from __future__ import annotations

import logging

import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from doccl.methods.aglr_replay import AGLRReplay
from doccl.types import TaskInfo

log = logging.getLogger(__name__)

__all__ = ["CoresetMemory"]


class CoresetMemory(AGLRReplay):
    """Per-(task×class) real-activation coreset (k-means centroids) replayed at layer ``k``."""

    name = "coreset_memory"

    def __init__(self, model, config):
        config = dict(config)
        super().__init__(model, config)
        self.centroids_per_class = int(config.get("centroids_per_class", 8))
        # task_id -> {"centroids": (M_total, d), "rows_for": {class_id: LongTensor of row idx},
        #             "carriers": [skeleton]}
        self._coreset: dict[int, dict] = {}

    # ── fit: k-means coreset of real activations per class ────────────────────

    def _fit_coreset(self, task: TaskInfo, loader: DataLoader) -> None:
        x, y, carriers = self._capture_with_labels(loader)  # reused from AGLRReplay
        if x.shape[0] < 1 or not carriers:
            log.warning(
                "coreset_memory: task %d — too few tokens/carriers, skipping fit", task.task_id
            )
            return
        cent_blocks: list[torch.Tensor] = []
        rows_for: dict[int, torch.Tensor] = {}
        cursor = 0
        for c in sorted(set(y.tolist())):
            xc = x[y == c]
            m = min(self.centroids_per_class, xc.shape[0])
            if m <= 0:
                continue
            if m == 1 or xc.shape[0] <= self.centroids_per_class:
                # Fewer points than requested centroids → keep the points themselves (real).
                cents = xc[:m]
            else:
                # k-means on real activations; centroids are convex means of real points, so
                # they still lie on the data manifold and carry joint dim structure.
                km = KMeans(n_clusters=m, n_init=3, max_iter=100, random_state=0)
                km.fit(xc.numpy())
                cents = torch.from_numpy(km.cluster_centers_).float()
            rows_for[int(c)] = torch.arange(cursor, cursor + cents.shape[0])
            cent_blocks.append(cents)
            cursor += cents.shape[0]
        if not cent_blocks:
            return
        self._coreset[task.task_id] = {
            "centroids": torch.cat(cent_blocks).cpu(),  # (M_total, d)
            "rows_for": {c: r.cpu() for c, r in rows_for.items()},
            "carriers": carriers,
        }
        log.info(
            "coreset_memory: task %d — %d centroids across %d classes (%d tokens, keep=%.2f, "
            "%d carriers); bank now %d tasks, ~%.3f MB",
            task.task_id,
            cursor,
            len(rows_for),
            x.shape[0],
            self.attn_keep,
            len(carriers),
            len(self._coreset),
            self.memory_bytes() / 1e6,
        )

    # ── replay: draw REAL centroids, label-coupled ────────────────────────────

    def _sample_replay(self) -> dict[str, torch.Tensor] | None:
        keys = [t for t, v in self._coreset.items() if v["rows_for"]]
        if not keys:
            return None
        b = self.replay_batch_size
        hiddens, bboxes, masks, labels_out = [], [], [], []
        for _ in range(b):
            tk = keys[torch.randint(len(keys), ()).item()]
            entry = self._coreset[tk]
            cents = entry["centroids"]  # (M, d)
            rows_for = entry["rows_for"]
            all_rows = torch.cat(list(rows_for.values()))
            carrier = entry["carriers"][torch.randint(len(entry["carriers"]), ()).item()]
            lab = carrier["labels"]  # (text_len,), may contain -100
            w = self._hidden_width if self._hidden_width is not None else lab.shape[0]
            # Per position: pick a random centroid ROW from that token's class; -100 /
            # image-patch / unseen-label positions draw from any centroid (inert for loss).
            pick = torch.empty(w, dtype=torch.long)
            for t in range(w):
                c = int(lab[t].item()) if t < lab.shape[0] else -100
                rws = rows_for.get(c)
                if rws is None or rws.numel() == 0:
                    rws = all_rows
                pick[t] = rws[torch.randint(rws.numel(), ()).item()]
            hidden = cents[pick]  # (w, d) — REAL centroid vectors
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

    # ── lifecycle + accounting ────────────────────────────────────────────────

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        if task.task_id == 0:
            self._apply_freeze_map()
        self._fit_coreset(task, train_loader)

    def memory_bytes(self) -> int:
        total = 0
        for entry in self._coreset.values():
            total += entry["centroids"].numel() * 2  # real activations stored fp16
            total += sum(
                (c["bbox"].numel() + c["attention_mask"].numel() + c["labels"].numel()) * 8
                for c in entry["carriers"]
            )
        return total
