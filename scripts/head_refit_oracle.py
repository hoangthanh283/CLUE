"""Head-refit oracle: how much of naive forgetting is readout-only?

The canonical Davari-style decomposition (Probing Representation Forgetting,
CVPR 2022), missing from our chain: train NAIVELY through the DIL sequence
(maximal forgetting, no protection), then FREEZE the drifted trunk and re-fit a
single linear head on pooled features from ALL tasks' train splits. The probe's
per-task F1 answers three questions at once:

  1. Are old-task features still linearly separable in the drifted trunk
     (representation forgetting vs readout forgetting)?
  2. What is the ceiling for EVERY head-only method (slots, KV memory,
     Gaussian replay, prototype classifiers) on this benchmark?
  3. How much of the naive->joint gap (AA ~41 -> ~88.7) is recoverable
     without touching the encoder?

Also fits per-task probes (head refit on one task's features only) — the
per-task representational ceiling — and reports the naive final row for the
same run as the baseline the probe is rescuing.

References on dil (3 seeds): naive AA ~41 / joint ~88.7 / ER ~87.9.
Probe AA >= ~85 => features intact, retention is a head-realignment problem
(Gaussian feature replay is then well-founded). Probe AA <= ~70 => features
drift more than CKA suggests — itself a finding.

GPU: one naive sequential run (~35 min) + feature extraction + probe fits.
Run from CLUE/:  uv run python scripts/head_refit_oracle.py
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from doccl.data.scenarios import get_scenario
from doccl.eval.metrics import compute_token_f1
from doccl.methods.naive import NaiveFineTune
from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

SEED = 42
BATCH_SIZE = 2
NAIVE_CFG = {"lr": 5e-5, "weight_decay": 0.01, "epochs": 10, "max_grad_norm": 1.0}
PROBE_EPOCHS = 30
PROBE_LR = 1e-3
PROBE_BATCH = 4096


def _loader(ds, shuffle=False):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def extract_features(model, ds, device):
    """Classifier-input features + labels for every supervised token, in order."""
    feats_box: list[torch.Tensor] = []
    cur: dict = {}

    def pre_hook(_m, inp):
        cur["f"] = inp[0]

    handle = model.model.classifier.register_forward_pre_hook(pre_hook)
    model.eval()
    xs, ys = [], []
    for batch in _loader(ds):
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        model(**{k: v for k, v in batch.items() if k != "labels"})
        f = cur["f"]  # (B, L, d)
        labels = batch["labels"]
        mask = labels != -100
        am = batch.get("attention_mask")
        if am is not None and am.shape == labels.shape:
            mask = mask & am.bool()
        xs.append(f[mask].detach().half().cpu())
        ys.append(labels[mask].detach().cpu())
    handle.remove()
    del feats_box
    return torch.cat(xs), torch.cat(ys)


def fit_probe(train_sets, d, n_labels, device):
    """Fit a fresh linear head on the given [(X, y), ...] feature sets."""
    x = torch.cat([t[0] for t in train_sets]).float()
    y = torch.cat([t[1] for t in train_sets])
    head = torch.nn.Linear(d, n_labels).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=PROBE_LR, weight_decay=0.01)
    n = x.shape[0]
    for ep in range(PROBE_EPOCHS):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, PROBE_BATCH):
            idx = perm[i : i + PROBE_BATCH]
            xb, yb = x[idx].to(device), y[idx].to(device)
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(xb), yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        if ep % 10 == 9:
            log.info("probe ep%d loss=%.4f (n=%d)", ep + 1, total / n, n)
    return head


@torch.no_grad()
def eval_probe(head, x, y, id_to_label, device):
    preds = []
    for i in range(0, x.shape[0], PROBE_BATCH):
        preds.append(head(x[i : i + PROBE_BATCH].float().to(device)).argmax(-1).cpu())
    m = compute_token_f1(torch.cat(preds).tolist(), y.tolist(), id_to_label)
    return round(m["f1"], 2)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scenario = get_scenario("dil")
    names = [t.task_name for t in scenario.tasks]
    labels0 = scenario.tasks[0].label_set
    model = LayoutLMv3Wrapper(model_name="microsoft/layoutlmv3-base", num_labels=len(labels0))
    model.label_to_id = {l: i for i, l in enumerate(labels0)}  # noqa: E741
    model.id_to_label = {i: l for l, i in model.label_to_id.items()}  # noqa: E741
    # BEFORE method construction: ContinualMethod derives self.device from the
    # model's parameters (base.py:95), exactly as train.py does at line ~418.
    model = model.to(device)
    if device == "cuda":
        model.enable_gradient_checkpointing()
    method = NaiveFineTune(model, dict(NAIVE_CFG))
    method.amp_enabled = device == "cuda"

    # ── 1. Naive sequential training (maximal forgetting) ────────────────────
    for task, train_ds, eval_ds in zip(
        scenario.tasks, scenario.train_datasets, scenario.eval_datasets, strict=True
    ):
        log.info("naive: training task %d (%s)", task.task_id, task.task_name)
        method.before_task(task, _loader(train_ds, shuffle=True))
        method.train_task(task, _loader(train_ds, shuffle=True), _loader(eval_ds))
        method.after_task(task, _loader(train_ds, shuffle=True))

    eval_loaders = {i: _loader(ds) for i, ds in enumerate(scenario.eval_datasets)}
    naive_final = {names[i]: round(r.f1, 2) for i, r in method.evaluate(eval_loaders).items()}
    log.info("naive final row: %s", naive_final)

    # ── 2. Freeze trunk, extract features everywhere ─────────────────────────
    for p in model.parameters():
        p.requires_grad = False
    train_feats = []
    eval_feats = []
    for i, (tr, ev) in enumerate(zip(scenario.train_datasets, scenario.eval_datasets, strict=True)):
        xt, yt = extract_features(model, tr, device)
        xe, ye = extract_features(model, ev, device)
        train_feats.append((xt, yt))
        eval_feats.append((xe, ye))
        log.info("features task %d: train %d tokens, eval %d tokens", i, len(yt), len(ye))

    d = train_feats[0][0].shape[1]
    n_labels = len(labels0)
    id_to_label = model.id_to_label

    # ── 3. Pooled probe (THE oracle) + per-task probes (per-task ceilings) ──
    pooled = fit_probe(train_feats, d, n_labels, device)
    pooled_f1 = {
        names[i]: eval_probe(pooled, *eval_feats[i], id_to_label, device) for i in range(3)
    }
    per_task_f1 = {}
    for i in range(3):
        h = fit_probe([train_feats[i]], d, n_labels, device)
        per_task_f1[names[i]] = eval_probe(h, *eval_feats[i], id_to_label, device)

    aa_naive = round(sum(naive_final.values()) / 3, 2)
    aa_pooled = round(sum(pooled_f1.values()) / 3, 2)
    result = {
        "seed": SEED,
        "naive_final_row": naive_final,
        "naive_AA_final": aa_naive,
        "pooled_probe_f1": pooled_f1,
        "pooled_probe_AA": aa_pooled,
        "per_task_probe_f1": per_task_f1,
        "refs": {"naive_AA": 41.3, "joint_AA": 88.7, "er_AA": 87.9, "lexmem_v3b_AA": 66.0},
    }
    log.info(
        "ORACLE: pooled-probe AA=%.2f (naive %.2f) — per-task %s", aa_pooled, aa_naive, pooled_f1
    )
    Path("results/head_refit_oracle.json").write_text(json.dumps(result, indent=2))
    log.info("saved -> results/head_refit_oracle.json")


if __name__ == "__main__":
    main()
