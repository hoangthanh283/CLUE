"""Gate 0 probe: are layer-k input features label-separable enough for a kNN readout?

Read-side-memory prerequisite (R1 `colar_knn` go/no-go). CoLaR freezes everything below
``split_layer_k`` after task 0, so layer-k input features are drift-free and its per-doc
SVD store is a stable labeled datastore — usable at INFERENCE, not just as replay
gradients. Before building that readout, this probe answers the one question it hinges
on: does a brute-force cosine-kNN over layer-k token features predict token labels at a
non-trivial F1?

No training anywhere: the probe runs on the raw pretrained trunk. That is a CONSERVATIVE
lower bound — the real datastore is captured from the post-task-0 (FUNSD-tuned) trunk,
which can only be more separable than pretrained.

Per dil task: datastore = that task's TRAIN tokens (labels != -100, text positions only —
LayoutLMv3 appends 197 visual patch tokens after the 512 text positions; labels cover only
the text slice), queries = the task's TEST tokens, seqeval F1 via the same
``compute_token_f1`` the CL loop uses, so numbers are directly comparable to paper F1s.

Sweep: k in {4, 8} x top_k in {5, 20} x weighting in {uniform, softmax tau=0.1}.

Go/no-go (plan): kNN F1 non-trivial on >= 2/3 tasks; SROIE > ~10-15 entity-F1 clears the
all-"O" degenerate baseline (seqeval scores entities only, so all-O predicts F1 = 0).

GPU: ~2000 doc forwards at batch 2 + small matmuls — minutes, no backward pass.
Run from CLUE/:  uv run python scripts/gate0_knn_probe.py
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
from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

SEED = 42
BATCH_SIZE = 2
K_LAYERS = (4, 8)
TOP_KS = (5, 20)
TAU = 0.1
QUERY_CHUNK = 1024


def _loader(ds):
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


@torch.no_grad()
def extract_layer_inputs(model, ds, device, layer_idxs):
    """Layer-k INPUT features + labels for every supervised text token, both k at once.

    Returns {k: (X (N, d) fp16 cpu, y (N,) int64 cpu)} — same capture point as
    LatentReplay._pre_hook (hidden entering encoder.layer[k]).
    """
    cur: dict[int, torch.Tensor] = {}

    def _mk(k):
        def hook(_m, args, kwargs):
            cur[k] = args[0] if args else kwargs["hidden_states"]

        return hook

    layers = model.model.layoutlmv3.encoder.layer
    handles = [layers[k].register_forward_pre_hook(_mk(k), with_kwargs=True) for k in layer_idxs]
    model.eval()
    xs: dict[int, list] = {k: [] for k in layer_idxs}
    ys: list[torch.Tensor] = []
    try:
        for batch in _loader(ds):
            batch = {n: v.to(device) for n, v in batch.items() if torch.is_tensor(v)}
            model(**{n: v for n, v in batch.items() if n != "labels"})
            labels = batch["labels"]
            mask = labels != -100
            am = batch.get("attention_mask")
            if am is not None and am.shape == labels.shape:
                mask = mask & am.bool()
            for k in layer_idxs:
                f = cur[k][:, : labels.shape[1]]  # drop visual patch positions
                xs[k].append(f[mask].detach().half().cpu())
            ys.append(labels[mask].cpu())
    finally:
        for h in handles:
            h.remove()
    y = torch.cat(ys)
    return {k: (torch.cat(xs[k]), y) for k in layer_idxs}


@torch.no_grad()
def knn_predict(train_x, train_y, test_x, top_k, tau, n_labels, device):
    """Chunked cosine-kNN label vote. tau=None -> uniform vote, else softmax(sim/tau)."""
    tr = torch.nn.functional.normalize(train_x.float(), dim=1).to(device)
    ty = train_y.to(device)
    preds = []
    for i in range(0, test_x.shape[0], QUERY_CHUNK):
        q = torch.nn.functional.normalize(test_x[i : i + QUERY_CHUNK].float(), dim=1).to(device)
        sim = q @ tr.T
        vals, idx = sim.topk(min(top_k, tr.shape[0]), dim=1)
        w = torch.ones_like(vals) if tau is None else torch.softmax(vals / tau, dim=1)
        dist = torch.zeros(vals.shape[0], n_labels, device=device)
        dist.scatter_add_(1, ty[idx], w)
        preds.append(dist.argmax(1).cpu())
    return torch.cat(preds)


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scenario = get_scenario("dil")
    labels0 = scenario.tasks[0].label_set  # dil: unified label space
    id_to_label = dict(enumerate(labels0))
    n_labels = len(labels0)
    model = LayoutLMv3Wrapper(model_name="microsoft/layoutlmv3-base", num_labels=n_labels)
    model = model.to(device)

    results: dict[str, dict] = {}
    for task, train_ds, eval_ds in zip(
        scenario.tasks, scenario.train_datasets, scenario.eval_datasets, strict=True
    ):
        name = task.task_name
        log.info("extracting features: %s", name)
        tr = extract_layer_inputs(model, train_ds, device, K_LAYERS)
        te = extract_layer_inputs(model, eval_ds, device, K_LAYERS)
        results[name] = {}
        for k in K_LAYERS:
            x_tr, y_tr = tr[k]
            x_te, y_te = te[k]
            for top_k in TOP_KS:
                for weighting, tau in (("uniform", None), ("softmax", TAU)):
                    preds = knn_predict(x_tr, y_tr, x_te, top_k, tau, n_labels, device)
                    f1 = compute_token_f1(preds.tolist(), y_te.tolist(), id_to_label)["f1"]
                    key = f"k{k}_top{top_k}_{weighting}"
                    results[name][key] = round(f1, 2)
                    log.info(
                        "%s %s: F1=%.2f (datastore=%d, queries=%d)",
                        name,
                        key,
                        f1,
                        len(y_tr),
                        len(y_te),
                    )
        del tr, te  # free before the next task (RAM cap)

    best = {name: max(cells.items(), key=lambda kv: kv[1]) for name, cells in results.items()}
    out = {
        "seed": SEED,
        "trunk": "pretrained (conservative lower bound; real store uses post-task-0 trunk)",
        "per_task": results,
        "best_per_task": {n: {"config": c, "f1": f} for n, (c, f) in best.items()},
        "go_criteria": "non-trivial F1 on >=2/3 tasks; SROIE > ~10-15 clears all-O baseline",
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/gate0_knn_probe.json").write_text(json.dumps(out, indent=2))
    log.info("GATE 0 best per task: %s", out["best_per_task"])
    log.info("saved -> results/gate0_knn_probe.json")


if __name__ == "__main__":
    main()
