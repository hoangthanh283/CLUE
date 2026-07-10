"""ProxyLatentReplay (PLaR) — zero-private-storage replay via pseudo-labeled PUBLIC docs.

Prescribed by the consistency law (STATE.md "GATE-0 FINAL", 2026-07-10): replay grounds the
shared head iff the replayed content preserves whole-document (feature, position, label)
co-occurrence — marginal memories (class Gaussians, real coresets, carrier scaffolds) all
collapse, and the single-variable control shows +27 AA from consistency alone. PLaR keeps
the consistency and removes the *private storage*:

At each task boundary, pass K documents from a **public proxy corpus** (WildReceipt by
default — different corpus from every dil/cil task) through the just-trained model and bank,
per doc, the whole layer-``k`` latent + bbox/mask + **pseudo-labels from the current head**
(argmax, optional confidence floor). The banked bundle is internally consistent by
construction: the proxy doc is real (feature[t] and bbox[t] from the same token) and labels
are bound once at bank time. The frozen trunk (inherited freeze map) makes the latents
immortal; banking-once makes the pseudo-labels drift-free.

Privacy: **no customer bytes are ever retained** — the buffer contains only public-corpus
content. This is deployable where even latent storage of real user documents is disallowed.

Everything except *what gets banked* is inherited from ``LatentReplay`` (the variant proven
to work at AA 87.3): freeze map, capture/inject hook, replay loss, CL loop integration.

Config keys (see ``configs/method/proxy_latent_replay.yaml``): ``proxy_dataset``
(``wildreceipt``), ``docs_per_task`` (proxy docs banked per boundary), ``pseudo_conf_tau``
(min softmax confidence; below → -100/ignored; 0 disables).
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

from doccl.methods.latent_replay import LatentReplay

log = logging.getLogger(__name__)

__all__ = ["ProxyLatentReplay"]


class ProxyLatentReplay(LatentReplay):
    """Whole-document latent replay from a pseudo-labeled public proxy pool."""

    name = "proxy_latent_replay"

    def __init__(self, model, config):
        super().__init__(model, config)
        self.proxy_dataset = str(config.get("proxy_dataset", "wildreceipt"))
        self.pseudo_conf_tau = float(config.get("pseudo_conf_tau", 0.0))
        # Soft (dark-knowledge, DER-style) pseudo-labels: bank the head's full logit
        # distribution per token and replay with soft cross-entropy instead of hard argmax.
        # Motivated by the d5/d50 finding: hard argmax starves sparse-entity tasks (SROIE)
        # of class signal (~99% "O"), while dense-label tasks (FUNSD) ground fine — the
        # full distribution carries boundary information even where argmax says "O".
        self.soft_labels = bool(config.get("soft_labels", False))
        self.soft_T = float(config.get("soft_T", 1.0))
        # Task-conditioned pseudo-labeling: at boundary t, mask the head's logits to task
        # t's label subset before argmax/softmax. Counters PROXY-DOMAIN CAPTURE (measured
        # 2026-07-11: the post-task-t head assigns ZERO task-t-class labels to proxy tokens
        # for t>0 — task 0 claims the proxy domain and replay self-reinforces it). Forcing
        # each boundary's fresh proxies into task-t's classes makes them carry task-t
        # boundary signal; the soft variant absorbs the extra label noise.
        self.task_masked_labels = bool(config.get("task_masked_labels", False))
        self._current_task = None  # stashed by after_task for the masking
        self._proxy_loader: DataLoader | None = None

    # ── proxy pool ────────────────────────────────────────────────────────────

    def _get_proxy_loader(self) -> DataLoader:
        """Build the public-pool loader lazily (once). Uses the process-default encoder,
        which train.py sets for the active backbone before datasets are constructed."""
        if self._proxy_loader is None:
            if self.proxy_dataset == "wildreceipt":
                from doccl.data.wildreceipt import WildReceiptDataset

                ds = WildReceiptDataset(split="train")
            else:
                raise ValueError(f"unknown proxy_dataset: {self.proxy_dataset}")
            # num_workers=0: workers fork the dataset working set (OOM risk on the 6GB box).
            self._proxy_loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
            log.info(
                "proxy_latent_replay: proxy pool '%s' ready (%d docs available)",
                self.proxy_dataset,
                len(ds),
            )
        return self._proxy_loader

    # ── banking: whole proxy docs, pseudo-labeled by the current head ─────────

    def after_task(self, task, train_loader: DataLoader) -> None:
        self._current_task = task  # for task-conditioned label masking
        super().after_task(task, train_loader)

    def _task_label_ids(self) -> list[int] | None:
        """Unified head ids of the current task's label subset (None = no masking)."""
        if not self.task_masked_labels or self._current_task is None:
            return None
        l2i = getattr(self.model, "label_to_id", None)
        if not l2i:
            return None
        ids = [l2i[name] for name in self._current_task.label_set if name in l2i]
        return ids or None

    def _capture_task(self, train_loader: DataLoader) -> None:
        """Bank ``docs_per_task`` PUBLIC docs instead of private training docs.

        ``train_loader`` (the private data) is deliberately ignored — that is the method.
        For each proxy doc: one forward gives BOTH the layer-``k`` hidden (via the inherited
        capture hook) and the current head's logits; bank {hidden, bbox, mask, pseudo-labels}.
        """
        was_training = self.model.training
        self.model.eval()
        loader = self._get_proxy_loader()
        stored = 0
        kept_tok, total_tok = 0, 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
                if "pixel_values" in batch:
                    self._pixel_shape = tuple(batch["pixel_values"].shape[1:])
                self._capture = []
                try:
                    out = self.model(
                        **{k: v for k, v in batch.items() if k != "labels"}
                    )  # no labels → no loss; logits sliced to text length by the HF head
                    hidden = self._capture[0]  # (b, seq_k, d) fp16 CPU via the hook
                finally:
                    self._capture = None
                logits = out.logits
                task_ids = self._task_label_ids()
                if task_ids is not None:
                    # Task-conditioned: only the current task's classes compete. Banked
                    # soft logits are masked too, so soft-CE anchors the within-task
                    # distribution rather than re-teaching the captured cross-task one.
                    keep = torch.full(
                        (logits.shape[-1],), False, dtype=torch.bool, device=logits.device
                    )
                    keep[torch.tensor(task_ids, device=logits.device)] = True
                    logits = logits.masked_fill(~keep, -1e4)  # fp16-safe "-inf"
                probs = logits.softmax(-1)  # (b, text_len, n_labels)
                conf, pseudo = probs.max(-1)
                am = batch["attention_mask"].bool()
                pseudo = pseudo.masked_fill(~am, -100)
                if self.pseudo_conf_tau > 0:
                    pseudo = pseudo.masked_fill(conf < self.pseudo_conf_tau, -100)
                kept_tok += int((pseudo != -100).sum())
                total_tok += int(am.sum())
                for i in range(hidden.shape[0]):
                    if stored >= self.docs_per_task:
                        break
                    doc = {
                        "hidden": hidden[i],
                        "bbox": batch["bbox"][i].cpu(),
                        "attention_mask": batch["attention_mask"][i].cpu(),
                        "labels": pseudo[i].cpu(),
                    }
                    if self.soft_labels:
                        # dark knowledge, banked once → drift-free (never regenerated);
                        # uses the task-masked logits when task_masked_labels is on
                        doc["logits"] = logits[i].detach().cpu().to(torch.float16)
                    self.store.append(doc)
                    stored += 1
                if stored >= self.docs_per_task:
                    break
        if was_training:
            self.model.train()
        # Class histogram of this boundary's pseudo-labels — the self-diagnosis for the
        # sparse-class starvation question (does the head put ANY mass on the just-learned
        # task's classes when labeling proxy docs?).
        recent = [d["labels"] for d in self.store[-stored:]] if stored else []
        if recent:
            lab = torch.cat(recent)
            lab = lab[lab != -100]
            ids, counts = lab.unique(return_counts=True)
            top = sorted(zip(ids.tolist(), counts.tolist(), strict=True), key=lambda x: -x[1])[:8]
            log.info("proxy_latent_replay: pseudo-label class histogram (top-8): %s", top)
        log.info(
            "proxy_latent_replay: banked %d PUBLIC docs (store=%d, ~%.1f MB fp16; "
            "pseudo-label keep=%.0f%%, tau=%.2f, soft=%s)",
            stored,
            len(self.store),
            sum(d["hidden"].numel() for d in self.store) * 2 / 1e6,
            100 * kept_tok / max(total_tok, 1),
            self.pseudo_conf_tau,
            self.soft_labels,
        )

    # ── soft-label replay loss (dark knowledge) ───────────────────────────────

    def _replay_forward(self, replay: dict[str, torch.Tensor]):
        """Hard mode: inherited (CE vs banked argmax labels). Soft mode: same injected
        forward but WITHOUT labels, then soft cross-entropy against the banked logit
        distribution — every masked token contributes full boundary information, so
        sparse-entity tasks are not starved by an all-"O" argmax."""
        if not self.soft_labels or "logits" not in replay:
            return super()._replay_forward(replay)
        bbox = replay["bbox"].to(self.device)
        attention_mask = replay["attention_mask"].to(self.device)
        hard = replay["labels"].to(self.device)
        b = bbox.shape[0]
        processor = getattr(self.model, "processor", None)
        pad_id = processor.tokenizer.pad_token_id if processor is not None else 0
        dummy_ids = torch.full((b, bbox.shape[1]), pad_id, dtype=torch.long, device=self.device)
        dummy_pixels = (
            torch.zeros((b, *self._pixel_shape), device=self.device)
            if self._pixel_shape is not None
            else None
        )
        self._inject = replay["hidden"]
        try:
            out = self.model(
                input_ids=dummy_ids,
                bbox=bbox,
                pixel_values=dummy_pixels,
                attention_mask=attention_mask,
            )
        finally:
            self._inject = None
        cur = out.logits  # (b, text, C_now)
        tgt = replay["logits"].to(self.device).float()  # (b, text, C_bank)
        c = min(cur.shape[-1], tgt.shape[-1])  # head may have grown since banking (CIL)
        mask = attention_mask.bool() & (hard != -100)
        t = self.soft_T
        if mask.any():
            logp = torch.log_softmax(cur[..., :c] / t, dim=-1)
            p = torch.softmax(tgt[..., :c] / t, dim=-1)
            loss = -(p * logp).sum(-1)[mask].mean() * (t * t)
        else:
            loss = cur.sum() * 0.0
        return type("Out", (), {"loss": loss, "logits": cur})()
