"""RCA kill-tests #1/#2: eval-time readout-marginal corrections on retrained naive.

The RCA showed naive's forgetting is a readout-marginal snap (head output marginal on old
tasks tracks the just-trained gold marginal, cos 0.997). These corrections test whether the
snap is REVERSIBLE with only per-task gold label marginals (num_labels floats/task) —
pre-registered decision rules in docs/RCA_KILLTESTS_PREREG_2026-07.md. BBSE itself does not
apply (it assumes a fixed classifier); the analog for a prior-shift-in-the-head is marginal
reweighting of the final softmax:

  (a) marginal_match  — task-ID oracle: per-class scaling iterated until the corrected
      predicted marginal equals the old task's stored gold marginal.
  (b) prior_ratio     — one-step reweight by q_old(y)/q_last(y): the literal snap inverse.
  (c) per_doc_em      — task-ID-FREE: Saerens–Latinne EM per document (a doc's ~512 tokens
      re-estimate its own marginal; source prior = last-trained task's marginal).

Stages (GPU box runs `train` once; `correct` is pure CPU/numpy):
  uv run python scripts/rca_readout_fixes.py --stage train
  uv run python scripts/rca_readout_fixes.py --stage correct
Outputs under results/rca/killtests/: naive_logits_task{t}.npz, train_meta.json,
readout_fixes.json, readout_fixes.md
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from doccl.data.scenarios import get_scenario
from doccl.eval.metrics import compute_per_class_f1, compute_token_f1
from doccl.methods.marginal_methods import gold_marginal
from doccl.methods.naive import NaiveFineTune
from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper
from doccl.pilot.run_pilot import set_seed

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

OUT_DIR = Path("results/rca/killtests")
SMOKE_AA, SMOKE_TOL = 37.9, 2.0  # Tier B naive final FULL AA — retrain drift gate
_EPS = 1e-8


# ---------------------------------------------------------------- train stage


def stage_train(seed: int = 42, batch_size: int = 2) -> None:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scenario = get_scenario("dil")
    labels = scenario.tasks[0].label_set
    model = LayoutLMv3Wrapper(model_name="microsoft/layoutlmv3-base", num_labels=len(labels))
    model.label_to_id = {la: i for i, la in enumerate(labels)}
    model.id_to_label = {i: la for la, i in model.label_to_id.items()}
    model = model.to(device)
    model.enable_gradient_checkpointing()
    config = yaml.safe_load(Path("configs/method/naive.yaml").read_text())
    config.pop("name", None)
    method = NaiveFineTune(model, config)
    method.amp_enabled = device.type == "cuda"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    marginals = {}
    eval_loaders: dict[int, DataLoader] = {}
    for t, task in enumerate(scenario.tasks):
        train_loader = DataLoader(scenario.train_datasets[t], batch_size=batch_size, shuffle=True)
        eval_loaders[t] = DataLoader(scenario.eval_datasets[t], batch_size=batch_size)
        marginals[t] = gold_marginal(train_loader, len(labels)).tolist()
        log.info("--- naive task %d (%s) ---", t, task.task_name)
        method.before_task(task, train_loader)
        method.train_task(task, train_loader, val_loader=eval_loaders[t])
        method.after_task(task, train_loader)

    # Final model: dump per-token softmax + labels + doc ids for every task's eval set.
    model.eval()
    final_row = {}
    with torch.no_grad():
        for t, loader in eval_loaders.items():
            probs, golds, docs = [], [], []
            doc = 0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                out = model(**{k: v for k, v in batch.items() if k != "labels"})
                p = torch.softmax(out.logits, dim=-1)
                lab = batch["labels"]
                for row_p, row_l in zip(p, lab, strict=True):  # one row = one document
                    keep = row_l != -100
                    # float32 (red-team fix): fp16 floors extinct-class probs (<6e-8) to
                    # exactly 0, which no downstream correction can resurrect — that would
                    # conflate "correction failed" with "information destroyed at storage".
                    probs.append(row_p[keep].cpu().numpy().astype(np.float32))
                    golds.append(row_l[keep].cpu().numpy().astype(np.int16))
                    docs.append(np.full(int(keep.sum()), doc, dtype=np.int32))
                    doc += 1
            probs, golds, docs = np.concatenate(probs), np.concatenate(golds), np.concatenate(docs)
            np.savez_compressed(
                OUT_DIR / f"naive_logits_task{t}.npz", probs=probs, labels=golds, docs=docs
            )
            f1 = compute_token_f1(probs.argmax(-1).tolist(), golds.tolist(), model.id_to_label)[
                "f1"
            ]
            final_row[t] = f1
            log.info("task %d final FULL f1 %.1f (%d tokens, %d docs)", t, f1, len(golds), doc)

    aa = float(np.mean(list(final_row.values())))
    meta = {
        "seed": seed,
        "labels": labels,
        "task_names": [t.task_name for t in scenario.tasks],
        "gold_marginals": marginals,
        "final_row_full_f1": final_row,
        "final_AA": aa,
        "smoke_pass": bool(abs(aa - SMOKE_AA) <= SMOKE_TOL),
    }
    (OUT_DIR / "train_meta.json").write_text(json.dumps(meta, indent=2))
    log.info(
        "final AA %.2f (smoke gate %.1f±%.1f: %s)", aa, SMOKE_AA, SMOKE_TOL, meta["smoke_pass"]
    )


# ---------------------------------------------------------------- corrections


def reweight(probs: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-class reweight + renormalize. probs (N, C), w (C,)."""
    p = probs * w[None, :]
    return p / p.sum(-1, keepdims=True).clip(min=_EPS)


def marginal_match(probs: np.ndarray, q_target: np.ndarray, iters: int = 50) -> np.ndarray:
    """Iterate per-class scaling until the mean corrected marginal matches q_target."""
    w = np.ones_like(q_target)
    for _ in range(iters):
        m = reweight(probs, w).mean(0)
        w = w * (q_target / m.clip(min=_EPS))
        w = w / w.sum()
    return reweight(probs, w)


def prior_ratio(
    probs: np.ndarray, q_old: np.ndarray, q_last: np.ndarray, alpha: float = 1e-4
) -> np.ndarray:
    """One-step reweight by q_old/q_last. Laplace-smooth the DENOMINATOR (red-team fix):
    q_last = CORD's marginal has exact-zero KEY/HEADER mass — a bare epsilon clip yields
    ~1e5-1e7 weights that amplify numerical noise into spurious argmax flips."""
    q_last_s = (q_last + alpha) / (q_last + alpha).sum()
    return reweight(probs, q_old / q_last_s)


def prior_ratio_weights(q_old: np.ndarray, q_last: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    q_last_s = (q_last + alpha) / (q_last + alpha).sum()
    return q_old / q_last_s


def per_doc_em(
    probs: np.ndarray, docs: np.ndarray, q_source: np.ndarray, iters: int = 5, alpha: float = 1e-4
) -> np.ndarray:
    """Saerens–Latinne EM per document: re-estimate the doc's own marginal, task-ID-free.

    q_source is Laplace-smoothed (code-review fix): an exact-zero source-prior class
    (CORD emits no KEY/HEADER) makes pi[c]=0 a permanent EM fixed point — the weight
    pi[c]/q_source[c] is 0 at iteration 1 and stays 0 regardless of document evidence.
    """
    q_source = (q_source + alpha) / (q_source + alpha).sum()
    out = np.empty_like(probs, dtype=np.float64)
    for d in np.unique(docs):
        idx = docs == d
        p = probs[idx].astype(np.float64)
        pi = q_source.copy()
        for _ in range(iters):
            post = reweight(p, pi / q_source)
            pi = post.mean(0)
        out[idx] = reweight(p, pi / q_source)
    return out


def stage_correct() -> None:
    meta = json.loads((OUT_DIR / "train_meta.json").read_text())
    if not meta["smoke_pass"]:
        raise SystemExit(f"smoke gate FAILED (AA {meta['final_AA']:.2f}) — do not adjudicate")
    labels = meta["labels"]
    id_to_label = dict(enumerate(labels))
    n_tasks = len(meta["task_names"])
    q = {int(t): np.asarray(m, dtype=np.float64) for t, m in meta["gold_marginals"].items()}
    q_last = q[n_tasks - 1]

    results: dict = {"meta": {k: meta[k] for k in ("seed", "final_AA", "final_row_full_f1")}}
    lines = [
        "# Readout-fix kill-test results",
        "",
        "| task | variant | F1 | KEY | HEADER | VALUE | O-acc |",
        "|---|---|---|---|---|---|---|",
    ]
    entity_cols = {
        e: [i for i, tag in enumerate(labels) if "-" in tag and tag.split("-", 1)[1] == e]
        for e in ("KEY", "HEADER")
    }
    for t in range(n_tasks):
        d = np.load(OUT_DIR / f"naive_logits_task{t}.npz")
        probs, golds, docs = d["probs"].astype(np.float64), d["labels"], d["docs"]
        variants = {
            "uncorrected": probs,
            "marginal_match": marginal_match(probs, q[t]),
            "prior_ratio": prior_ratio(probs, q[t], q_last),
            "per_doc_em": per_doc_em(probs, docs, q_last),
        }
        results[str(t)] = {}
        if t < n_tasks - 1:  # old tasks: pre-registered saturation + weight diagnostics
            # Saturation check (red-team fix): distinguishes "correction mechanism
            # ineffective" from "KEY/HEADER mass already destroyed in the stored logits"
            # — the two KILL readings with opposite method-chapter implications.
            results[str(t)]["diagnostics"] = {
                f"uncorrected_{e.lower()}_prob_mass": float(probs[:, cols].sum(1).mean())
                for e, cols in entity_cols.items()
            }
            w = prior_ratio_weights(q[t], q_last)
            results[str(t)]["diagnostics"]["prior_ratio_max_weight"] = float(w.max())
            if w.max() > 1e4:
                lines.append(f"⚠ task {t}: prior_ratio max weight {w.max():.1e} > 1e4")
        for name, p in variants.items():
            preds = p.argmax(-1)
            f1 = compute_token_f1(preds.tolist(), golds.tolist(), id_to_label)
            per_class = compute_per_class_f1(preds.tolist(), golds.tolist(), id_to_label)
            o_mask = golds == labels.index("O")
            o_acc = float((preds[o_mask] == labels.index("O")).mean()) if o_mask.any() else np.nan
            results[str(t)][name] = {
                "f1": f1["f1"],
                "per_class": per_class,
                "o_row_acc": o_acc,
            }
            key, hdr, val = (
                per_class.get(c, {}).get("f1", float("nan")) for c in ("KEY", "HEADER", "VALUE")
            )
            lines.append(
                f"| {meta['task_names'][t]} | {name} | {f1['f1']:.1f} | {key:.1f} "
                f"| {hdr:.1f} | {val:.1f} | {o_acc*100:.1f} |"
            )
    # Headline (amended prereg rule 1): pooled AA AND old-tasks-only AA per variant,
    # plus the cord non-regression guard (≤ 2 pts drop under any variant).
    for name in ("uncorrected", "marginal_match", "prior_ratio", "per_doc_em"):
        aa = float(np.mean([results[str(t)][name]["f1"] for t in range(n_tasks)]))
        aa_old = float(np.mean([results[str(t)][name]["f1"] for t in range(n_tasks - 1)]))
        cord_delta = (
            results[str(n_tasks - 1)][name]["f1"] - results[str(n_tasks - 1)]["uncorrected"]["f1"]
        )
        results.setdefault("AA", {})[name] = aa
        results.setdefault("AA_old", {})[name] = aa_old
        results.setdefault("cord_delta", {})[name] = float(cord_delta)
        guard = "" if cord_delta >= -2.0 else " ⚠cord-regression"
        lines.append(f"| **AA / AA_old** | {name} | {aa:.1f} / {aa_old:.1f}{guard} | | | | |")
    (OUT_DIR / "readout_fixes.json").write_text(json.dumps(results, indent=2, default=float))
    (OUT_DIR / "readout_fixes.md").write_text("\n".join(lines) + "\n")
    log.info("AA per variant: %s", results["AA"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["train", "correct"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=2)
    args = ap.parse_args()
    if args.stage == "train":
        stage_train(seed=args.seed, batch_size=args.batch_size)
    else:
        stage_correct()


if __name__ == "__main__":
    main()
