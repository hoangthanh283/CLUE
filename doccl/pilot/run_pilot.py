"""Pilot study: per-component forgetting analysis.

Runs naive sequential training (FUNSD → CORD → SROIE) for five conditions:
    Cb: BERT-base                       external unimodal text baseline
    C1: LayoutLMv3 text-only            (text; image + layout zeroed)
    C2: LayoutLMv3-no-text              (image + layout)
    C3: LayoutLMv3-no-image             (text + layout)
    C4: LayoutLMv3-full                 (text + image + layout)

At each task boundary, captures:
    - Per-token CKA between consecutive checkpoints per layer (representational drift)
    - Per-parameter empirical Fisher (importance to the current task) per group
    - Old-task-Fisher-weighted parameter displacement per component group AND per
      depth bucket — the forgetting localizer (review C4)
    - Final accuracy on each seen task

Cb (BERT) gives the unimodal contrast the characterization needs (review C1/M1):
the question "does the multimodal encoder forget *differently* from a unimodal
one?" is answered by comparing Cb's depth gradient against C1–C4.

Results saved to results/pilot/{condition}_seed{seed}.json for downstream
analysis. This is the diagnostic core of the AAAI paper.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from doccl.data.bert_adapter import BertKIEAdapter
from doccl.data.scenarios import build_pilot
from doccl.eval.cka import collect_activations, linear_cka
from doccl.eval.fisher import (
    empirical_fisher_diagonal,
    fisher_per_group,
    fisher_weighted_displacement,
    snapshot_params,
)
from doccl.eval.metrics import CLMetricsTracker, compute_token_f1
from doccl.models.bert_wrapper import BertTokenClassificationWrapper
from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper
from doccl.types import ModalityMask

log = logging.getLogger(__name__)


# Condition → modality mask for the LayoutLMv3 variants (Cb uses a separate model).
_LAYOUTLM_CONDITIONS = {
    "c1_text": ModalityMask.TEXT_ONLY,
    "c2_no_text": ModalityMask.IMAGE_LAYOUT,
    "c3_no_image": ModalityMask.TEXT_LAYOUT,
    "c4_full": ModalityMask.FULL,
}
ALL_CONDITIONS = ["cb_bert", *_LAYOUTLM_CONDITIONS.keys()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def run_pilot_condition(
    condition: str,
    seed: int,
    output_dir: Path,
    epochs_per_task: int = 10,
    batch_size: int = 8,
    cka_n_samples: int = 2000,
    fisher_n_samples: int = 500,
    task_order: list[int] | None = None,
    gradient_checkpointing: bool = False,
) -> dict:
    """Run one condition × seed of the pilot study.

    Args:
        condition: one of ``ALL_CONDITIONS``.
        seed: random seed.
        output_dir: where to save results.
        cka_n_samples: number of *valid tokens* for per-token CKA (review M2 asks
            for N ≥ 500; default 2000).
        fisher_n_samples: number of documents for the Fisher estimate.
        task_order: optional permutation of [0,1,2] over (FUNSD, CORD, SROIE) for
            the stability check; default order if None.

    Returns:
        Summary dict of metrics (also saved to disk).
    """
    if condition not in ALL_CONDITIONS:
        raise ValueError(f"Unknown condition {condition!r}; expected one of {ALL_CONDITIONS}")
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 80)
    log.info(f"Pilot condition: {condition}, seed={seed}, order={task_order}, device={device}")
    log.info("=" * 80)

    scenario = build_pilot(order=task_order)
    model, modality_mask = _build_condition_model(condition, scenario)
    model = model.to(device)
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
        log.info("Gradient checkpointing enabled (lower memory, slower).")

    is_bert = condition == "cb_bert"
    fisher_mask = None if is_bert else modality_mask
    train_dss, eval_dss = _build_datasets(scenario, model, is_bert)

    # Reservoirs for analysis
    cka_records: list[dict] = []  # one per task boundary
    fisher_records: list[dict] = []  # one per task (Fisher importance level)
    displacement_records: list[dict] = []  # one per task boundary (forgetting)
    accuracy_records: list[dict] = []  # one per task evaluation

    tracker = CLMetricsTracker(num_tasks=len(scenario.tasks))
    eval_loaders: dict[int, DataLoader] = {}

    prev_activations: dict[str, torch.Tensor] | None = None
    prev_fisher_pp: dict[str, torch.Tensor] | None = None  # F^{(t-1)} (per-param, CPU)
    prev_params: dict[str, torch.Tensor] | None = None  # θ^{t-1} snapshot (CPU)

    for task_idx, task in enumerate(scenario.tasks):
        log.info(f"\n--- Task {task_idx}: {task.task_name} ---")

        # Expand classifier for new labels
        new_labels = [l for l in task.label_set if l not in model.label_to_id]
        if new_labels:
            model.expand_classifier(new_labels)
            model = model.to(device)

        train_loader = DataLoader(
            train_dss[task_idx], batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True,
        )
        eval_loader = DataLoader(
            eval_dss[task_idx], batch_size=batch_size, shuffle=False,
            num_workers=2, pin_memory=True,
        )
        eval_loaders[task_idx] = eval_loader

        # ─── Capture pre-training activations for CKA (before training task) ───
        # Both checkpoints are probed on the SAME inputs (this task's eval set):
        # pre with the previous checkpoint here, post below.
        if task_idx > 0:
            log.info("Capturing pre-training activations for CKA...")
            prev_activations = _capture_activations(
                model, eval_loader, fisher_mask, cka_n_samples, device
            )

        # ─── Train naively on this task ────────────────────────────────────────
        _train_naive(model, train_loader, modality_mask, epochs_per_task, device, is_bert)

        # ─── Capture post-training activations + per-token CKA vs pre-capture ──
        if prev_activations is not None:
            log.info("Capturing post-training activations...")
            cur_activations = _capture_activations(
                model, eval_loader, fisher_mask, cka_n_samples, device
            )
            common = cur_activations.keys() & prev_activations.keys()
            cka = {
                layer: linear_cka(prev_activations[layer], cur_activations[layer])
                for layer in common
            }
            cka_records.append({"task_boundary": f"{task_idx-1}_to_{task_idx}", "cka": cka})
            log.info(f"CKA at boundary {task_idx-1}→{task_idx}: {cka}")
            prev_activations = None

        # ─── Fisher importance per group (under the condition's mask) ───────────
        log.info("Computing Fisher information...")
        fisher_pp = empirical_fisher_diagonal(
            model, train_loader, n_samples=fisher_n_samples, device=device,
            modality_mask=fisher_mask,
        )
        fisher_grouped = fisher_per_group(fisher_pp, model.param_groups, model)
        fisher_records.append({"task_idx": task_idx, "fisher_per_group": fisher_grouped})
        log.info(f"Fisher per group: {fisher_grouped}")

        # ─── Fisher-weighted displacement = forgetting localizer (review C4) ───
        cur_params = snapshot_params(model)
        if task_idx > 0 and prev_fisher_pp is not None and prev_params is not None:
            by_group = fisher_weighted_displacement(
                prev_fisher_pp, prev_params, cur_params, model.param_groups, model
            )
            by_depth = fisher_weighted_displacement(
                prev_fisher_pp, prev_params, cur_params, model.param_groups_by_depth, model
            )
            displacement_records.append({
                "task_boundary": f"{task_idx-1}_to_{task_idx}",
                "by_group": by_group,
                "by_depth": by_depth,
            })
            log.info(f"Fisher-weighted displacement (depth) {task_idx-1}→{task_idx}: {by_depth}")
        # Carry θ^t and F^{(t)} forward (on CPU to free GPU memory).
        prev_params = cur_params
        prev_fisher_pp = {k: v.detach().cpu() for k, v in fisher_pp.items()}
        del fisher_pp

        # ─── Evaluate on all seen tasks ────────────────────────────────────────
        log.info("Evaluating on all seen tasks...")
        eval_results = _evaluate_all(model, eval_loaders, modality_mask, device, is_bert)
        for tid, m in eval_results.items():
            log.info(f"  Task {tid}: F1={m['f1']:.2f}")
        tracker.update(task_idx, eval_results)
        accuracy_records.append({"task_idx": task_idx, "results": eval_results})

    # ─── Save results ──────────────────────────────────────────────────────────
    order = task_order or [0, 1, 2]
    summary = {
        "condition": condition,
        "seed": seed,
        "task_order": order,
        "cka_records": cka_records,
        "fisher_records": fisher_records,
        "displacement_records": displacement_records,
        "accuracy_records": accuracy_records,
        "cl_metrics": tracker.summary(),
        "matrix": tracker.matrix.tolist(),
    }
    order_suffix = "" if order == [0, 1, 2] else "_ord" + "".join(str(i) for i in order)
    out_path = output_dir / f"{condition}_seed{seed}{order_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nSaved pilot results to {out_path}")
    log.info(f"Final: AA={tracker.average_accuracy():.2f} BWT={tracker.backward_transfer():.2f}")

    return summary


def _build_condition_model(condition: str, scenario):
    """Construct the model and modality mask for a pilot condition.

    Cb is a real external BERT-base text encoder; C1–C4 are the same LayoutLMv3
    backbone run under different input-modality masks (C1 = real text-only, which
    keeps ``input_ids``). Returns ``(model, modality_mask)`` where the mask is
    ``None`` for the BERT condition.
    """
    n_labels = len(scenario.tasks[0].label_set)
    if condition == "cb_bert":
        model: torch.nn.Module = BertTokenClassificationWrapper(num_labels=n_labels)
        mask = None
    else:
        model = LayoutLMv3Wrapper(num_labels=n_labels)
        mask = _LAYOUTLM_CONDITIONS[condition]

    model.label_to_id = {l: i for i, l in enumerate(scenario.tasks[0].label_set)}
    model.id_to_label = {i: l for l, i in model.label_to_id.items()}
    return model, mask


def _build_datasets(scenario, model, is_bert: bool) -> tuple[list[Dataset], list[Dataset]]:
    """Return per-task (train, eval) datasets, BERT-tokenised for the Cb condition.

    For Cb we re-tokenise the same documents with BERT WordPiece (``BertKIEAdapter``)
    so the baseline is a genuine unimodal text encoder rather than a re-use of the
    RoBERTa ids (review C1/M1).
    """
    if not is_bert:
        return list(scenario.train_datasets), list(scenario.eval_datasets)
    tok = model.tokenizer
    train = [BertKIEAdapter(ds, tok) for ds in scenario.train_datasets]
    eval_ = [BertKIEAdapter(ds, tok) for ds in scenario.eval_datasets]
    return train, eval_


def _train_naive(
    model,
    loader: DataLoader,
    modality_mask: ModalityMask | None,
    epochs: int,
    device: torch.device,
    is_bert: bool,
) -> None:
    """Naive sequential training for one task."""
    from tqdm import tqdm
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-5, weight_decay=0.01
    )
    extra = {} if is_bert else {"modality_mask": modality_mask}
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"ep{epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            optimizer.zero_grad()
            out = model(**batch, **extra)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix({"loss": f"{out.loss.item():.3f}"})


def _capture_activations(
    model,
    loader: DataLoader,
    modality_mask: ModalityMask | None,
    max_samples: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Collect per-token activations from the model's CKA probe layers."""
    return collect_activations(
        model, loader, model.cka_layers, max_samples=max_samples, device=device,
        token_level=True, modality_mask=modality_mask,
    )


def _evaluate_all(
    model,
    eval_loaders: dict[int, DataLoader],
    modality_mask: ModalityMask | None,
    device: torch.device,
    is_bert: bool,
) -> dict[int, dict[str, float]]:
    """Evaluate on all seen tasks."""
    model.eval()
    extra = {} if is_bert else {"modality_mask": modality_mask}
    out = {}
    with torch.no_grad():
        for tid, loader in eval_loaders.items():
            preds, golds = [], []
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                outputs = model(
                    **{k: v for k, v in batch.items() if k != "labels"}, **extra
                )
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                p = logits.argmax(-1)
                lbl = batch["labels"]
                m = lbl != -100
                preds.extend(p[m].cpu().tolist())
                golds.extend(lbl[m].cpu().tolist())
            out[tid] = compute_token_f1(preds, golds, model.id_to_label)
    return out


def main():
    """CLI entry: run conditions × seeds."""
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS)
    # ≥5 seeds so the cross-condition MWU floor 2/C(n1+n2,n2) drops below the
    # Bonferroni threshold (review C3): 5 vs 5 → 2/C(10,5)=0.0079 < 0.0167.
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7, 1, 2])
    parser.add_argument("--output_dir", type=Path, default=Path("results/pilot"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--task_order", nargs="+", type=int, default=None,
        help="Permutation of 0 1 2 over (FUNSD CORD SROIE); e.g. '2 1 0' for the "
             "alternate-order stability check. Default: 0 1 2.",
    )
    parser.add_argument(
        "--cka_n_samples", type=int, default=2000,
        help="Number of valid tokens for per-token CKA (review M2: N ≥ 500).",
    )
    parser.add_argument(
        "--fisher_n_samples", type=int, default=500,
        help="Document count for the Fisher estimate.",
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true",
        help="Recompute activations in backward to fit limited-VRAM GPUs.",
    )
    args = parser.parse_args()

    for cond in args.conditions:
        for seed in args.seeds:
            run_pilot_condition(
                condition=cond,
                seed=seed,
                output_dir=args.output_dir,
                epochs_per_task=args.epochs,
                batch_size=args.batch_size,
                cka_n_samples=args.cka_n_samples,
                fisher_n_samples=args.fisher_n_samples,
                task_order=args.task_order,
                gradient_checkpointing=args.gradient_checkpointing,
            )


if __name__ == "__main__":
    main()
