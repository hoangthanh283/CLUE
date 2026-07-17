"""RCA Tier B: instrumented core-6 baseline re-run with boundary probes (dil, LayoutLMv3).

Runs ONE method through the dil sequence (FUNSD→SROIE→CORD, unified head) at the standard
train.py budget (epochs from the method's yaml, val-F1 early stop on the task's eval set),
and at every task boundary captures what no saved artifact contains:

  1. Modality-ablated eval on ALL seen tasks — trained FULL, evaluated under
     {FULL, TEXT_ONLY, TEXT_LAYOUT, IMAGE_LAYOUT}: attributes retained vs forgotten F1
     to modality streams (the multimodal-correlation probe the pilot never ran).
  2. Per-class F1 + token-level confusion matrix per seen task per boundary — the error
     taxonomy (entity→O extinction vs entity→entity confusion vs B/I flips).
  3. Fisher-weighted parameter displacement per component AND per depth (run_pilot's
     localizer, previously naive-only) — extends Finding 2 to every mechanism family.

Config comes straight from configs/method/<name>.yaml (flat dict, no hydra). Caveat
recorded in the output JSON: colar freezes the backbone after task 0, so its
param_groups are head-only from boundary 1→2 on — absent keys mean "not trainable",
not "no displacement".

Run (one method):
  uv run python scripts/rca_baselines.py --method naive --seed 42 \
      --batch-size 2 --gradient-checkpointing --output results/rca/dil_naive_seed42_rca.json
Chain all six detached: scripts/run_rca_baselines.sh
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
from doccl.eval.confusion import confusion_counts
from doccl.eval.fisher import (
    empirical_fisher_diagonal,
    fisher_weighted_displacement,
    snapshot_params,
)
from doccl.eval.metrics import compute_per_class_f1, compute_token_f1
from doccl.methods.colar import CoLaR
from doccl.methods.der import DERpp
from doccl.methods.er import ER
from doccl.methods.ewc import EWC
from doccl.methods.lwf import LwF
from doccl.methods.marginal_methods import LogitAdjust, MarginalAnchor
from doccl.methods.naive import NaiveFineTune
from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper
from doccl.pilot.run_pilot import set_seed
from doccl.types import ModalityMask

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

METHOD_CLASSES = {
    "naive": NaiveFineTune,
    "ewc": EWC,
    "lwf": LwF,
    "er": ER,
    "der_pp": DERpp,
    "colar": CoLaR,
    # RCA kill-tests #4/#5 (docs/RCA_KILLTESTS_PREREG_2026-07.md)
    "marginal_kl": MarginalAnchor,
    "logit_adjust": LogitAdjust,
}
EVAL_MASKS = (
    ModalityMask.FULL,
    ModalityMask.TEXT_ONLY,
    ModalityMask.TEXT_LAYOUT,
    ModalityMask.IMAGE_LAYOUT,
)


def load_method_config(name: str) -> dict:
    cfg = yaml.safe_load(Path(f"configs/method/{name}.yaml").read_text())
    cfg.pop("name", None)
    return cfg


@torch.no_grad()
def eval_under_mask(model, loader, mask: ModalityMask, id_to_label: dict, device) -> dict:
    """Direct-forward eval (bypasses method.evaluate overrides — uniform across methods)."""
    model.eval()
    n_labels = len(id_to_label)
    all_preds: list[int] = []
    all_labels: list[int] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = model(**{k: v for k, v in batch.items() if k != "labels"}, modality_mask=mask)
        preds = out.logits.argmax(-1)
        labels = batch["labels"]
        keep = labels != -100
        all_preds.extend(preds[keep].cpu().tolist())
        all_labels.extend(labels[keep].cpu().tolist())
    return {
        "f1": compute_token_f1(all_preds, all_labels, id_to_label)["f1"],
        "per_class": compute_per_class_f1(all_preds, all_labels, id_to_label),
        "confusion": confusion_counts(all_preds, all_labels, n_labels).tolist(),
    }


def run_one_method(
    method_name: str,
    seed: int,
    output_path: Path,
    batch_size: int = 2,
    gradient_checkpointing: bool = True,
    num_workers: int = 0,
    fisher_n_samples: int = 200,
    freeze_trunk: bool = False,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scenario = get_scenario("dil")
    labels0 = scenario.tasks[0].label_set  # dil: unified label space, fixed head
    config = load_method_config(method_name)
    model = LayoutLMv3Wrapper(model_name="microsoft/layoutlmv3-base", num_labels=len(labels0))
    model.label_to_id = {label: i for i, label in enumerate(labels0)}
    model.id_to_label = {i: label for label, i in model.label_to_id.items()}
    model = model.to(device)
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
    if freeze_trunk:
        # RCA kill-test #3: head-only training from task 0 — pure-head causality probe.
        model.freeze_backbone()
    method = METHOD_CLASSES[method_name](model, config)
    method.amp_enabled = device.type == "cuda"

    boundaries: list[dict] = []
    prev_params = prev_fisher = None
    eval_loaders: dict[int, DataLoader] = {}

    for task_idx, task in enumerate(scenario.tasks):
        log.info("--- %s task %d (%s) ---", method_name, task_idx, task.task_name)
        train_loader = DataLoader(
            scenario.train_datasets[task_idx],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        eval_loader = DataLoader(
            scenario.eval_datasets[task_idx],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        eval_loaders[task_idx] = eval_loader

        method.before_task(task, train_loader)
        method.train_task(task, train_loader, val_loader=eval_loader)
        method.after_task(task, train_loader)  # colar's freeze map fires after task 0

        record: dict = {"after_task": task_idx, "task_name": task.task_name}

        # Forgetting localizer (run_pilot order: displacement uses PREVIOUS task's Fisher).
        cur_params = snapshot_params(model)
        if task_idx > 0 and prev_fisher is not None:
            record["displacement_by_group"] = fisher_weighted_displacement(
                prev_fisher, prev_params, cur_params, model.param_groups, model
            )
            record["displacement_by_depth"] = fisher_weighted_displacement(
                prev_fisher, prev_params, cur_params, model.param_groups_by_depth, model
            )
        prev_params = cur_params
        fisher_pp = empirical_fisher_diagonal(
            model, train_loader, n_samples=fisher_n_samples, device=device
        )
        prev_fisher = {k: v.detach().cpu() for k, v in fisher_pp.items()}
        del fisher_pp

        # Modality-ablated eval + per-class + confusion, on every seen task.
        record["eval"] = {
            tid: {
                mask.value: eval_under_mask(model, loader, mask, model.id_to_label, device)
                for mask in EVAL_MASKS
            }
            for tid, loader in eval_loaders.items()
        }
        full_row = {t: e[ModalityMask.FULL.value]["f1"] for t, e in record["eval"].items()}
        log.info("boundary %d FULL row: %s", task_idx, full_row)
        boundaries.append(record)

        # Crash-safe: persist after every boundary (a 3-task run is hours on the 2060).
        result = {
            "method": f"{method_name}_frozen" if freeze_trunk else method_name,
            "freeze_trunk": freeze_trunk,
            "seed": seed,
            "scenario": "dil",
            "labels": labels0,
            "masks": [m.value for m in EVAL_MASKS],
            "caveats": [
                "n=1 seed — provisional",
                "colar: backbone frozen after task 0 -> param_groups head-only from "
                "boundary 1->2 on (absent keys = not trainable, not zero displacement)",
            ],
            "boundaries": boundaries,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, default=float))

    final_full = np.mean(
        [boundaries[-1]["eval"][t][ModalityMask.FULL.value]["f1"] for t in eval_loaders]
    )
    log.info("%s done — final FULL AA=%.1f -> %s", method_name, final_full, output_path)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True, choices=sorted(METHOD_CLASSES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--fisher-samples", type=int, default=200)
    ap.add_argument("--freeze-trunk", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    run_one_method(
        args.method,
        args.seed,
        args.output,
        batch_size=args.batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        num_workers=args.num_workers,
        fisher_n_samples=args.fisher_samples,
        freeze_trunk=args.freeze_trunk,
    )


if __name__ == "__main__":
    main()
