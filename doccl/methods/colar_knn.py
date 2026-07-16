"""CoLaR-kNN — read-side memory: a non-parametric kNN readout over CoLaR's own store.

R1 of the read-side-memory direction. Every prior memory attempt in this repo was a
training-time WRITE (LARM's layer-k correction, LexSlot slots, nullspace_analytic) and
died under the redistribution law: training-time levers only move retention between
tasks. This method instead READS the store at inference. Two facts make that sound:

1. Layers < k are frozen after task 0, so a test doc's layer-k features come from the
   same function that produced every banked factor — the datastore cannot go stale
   (the exact failure that killed LARM cannot occur below k).
2. Forgetting is head-localized (Finding 1); a kNN vote over banked (feature, label)
   pairs is a head that cannot forget by construction.

At eval, token probabilities are blended in probability space (kNN-LM precedent):
``p = (1 - lambda) * softmax(head_logits) + lambda * p_knn``, where p_knn is the
softmax(cos/tau)-weighted label vote of the top-k datastore neighbors of the token's
layer-k feature. ``knn_lambda=0`` short-circuits to plain CoLaR (byte-identical eval);
``knn_lambda=1`` is the pure memory-head ablation. Training is untouched — any AA gain
is non-redistributive by design. Adds zero buffer bytes (the datastore is a view over
the factors CoLaR already banks; ``memory_bytes`` is inherited unchanged).
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from doccl.eval.metrics import compute_token_f1
from doccl.methods.colar import CoLaR
from doccl.methods.latent_datastore import LatentDatastore
from doccl.types import EvalMetrics, TaskInfo

log = logging.getLogger(__name__)

__all__ = ["CoLaRKNN"]


class CoLaRKNN(CoLaR):
    """CoLaR + kNN readout over the banked layer-k datastore at eval time."""

    name = "colar_knn"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.knn_lambda = float(config.get("knn_lambda", 0.3))
        self.knn_top_k = int(config.get("knn_top_k", 20))
        self.knn_tau = float(config.get("knn_tau", 0.1))
        self.knn_class_balance = bool(config.get("knn_class_balance", False))
        self.datastore = LatentDatastore()

    def after_task(self, task: TaskInfo, train_loader: DataLoader) -> None:
        super().after_task(task, train_loader)
        self.datastore.invalidate()

    def evaluate(self, eval_loaders: dict[int, DataLoader]) -> dict[int, EvalMetrics]:
        # lambda=0 (ablation control) and the pre-task-0 zero-shot FWT reading (empty
        # store) take the inherited parametric path — byte-identical to plain CoLaR.
        if self.knn_lambda <= 0 or not self.store:
            return super().evaluate(eval_loaders)

        self.datastore.build(self.store)
        tau = self.knn_tau if self.knn_tau > 0 else None
        self.model.eval()
        id_to_label = getattr(self.model, "id_to_label", None)
        if not id_to_label:
            id_to_label = {i: str(i) for i in range(self.model.model.config.num_labels)}

        results: dict[int, EvalMetrics] = {}
        try:
            with torch.no_grad():
                for tid, loader in eval_loaders.items():
                    all_preds, all_labels = [], []
                    for batch in loader:
                        batch = {
                            k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)
                        }
                        self._capture = []  # reuse the layer-k pre-hook in capture mode
                        try:
                            outputs = self.model(
                                **{k: v for k, v in batch.items() if k != "labels"}
                            )
                            layer_k_in = self._capture[0]  # (B, L_full, d) fp16 cpu
                        finally:
                            self._capture = None
                        labels = batch["labels"]
                        p_head = outputs.logits.float().softmax(-1)  # (B, L_text, C)
                        query = layer_k_in[:, : labels.shape[1]]  # drop visual positions
                        p_knn = self.datastore.knn_label_dist(
                            query,
                            self.knn_top_k,
                            tau,
                            p_head.shape[-1],
                            class_balance=self.knn_class_balance,
                            device=self.device,
                        )
                        p = (1.0 - self.knn_lambda) * p_head + self.knn_lambda * p_knn
                        preds = p.argmax(-1)
                        mask = labels != -100
                        all_preds.extend(preds[mask].cpu().tolist())
                        all_labels.extend(labels[mask].cpu().tolist())
                    metrics = compute_token_f1(all_preds, all_labels, id_to_label)
                    results[tid] = EvalMetrics(
                        task_id=tid,
                        f1=metrics["f1"],
                        precision=metrics["precision"],
                        recall=metrics["recall"],
                        n_samples=len(all_labels),
                    )
        finally:
            self.datastore.free_device()
        return results
