"""Pilot study: per-component forgetting analysis.

Runs naive sequential training (FUNSD → CORD → SROIE) for 4 conditions:
    C1: BERT-base                      (text only)
    C2: LayoutLMv3-no-text              (image + layout)
    C3: LayoutLMv3-no-image             (text + layout)
    C4: LayoutLMv3-full                 (text + image + layout)

At each task boundary, captures:
    - CKA between consecutive checkpoints per layer
    - Fisher information per parameter group
    - Final accuracy on each seen task

Results saved to results/pilot/{condition}_{seed}/ for downstream analysis.

This is the diagnostic core of the AAAI paper (Section 4).
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from doccl.data.scenarios import build_pilot
from doccl.eval.cka import collect_activations, linear_cka
from doccl.eval.fisher import empirical_fisher_diagonal, fisher_per_group
from doccl.eval.metrics import CLMetricsTracker, compute_token_f1
from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper
from doccl.types import ModalityMask

log = logging.getLogger(__name__)


# Layer names to track via CKA — selected to span text/image/fusion components
LAYERS_TO_TRACK = [
    # Text branch (low / mid / high)
    "model.layoutlmv3.encoder.layer.0",
    "model.layoutlmv3.encoder.layer.5",
    "model.layoutlmv3.encoder.layer.11",
    # Embeddings (where text+layout fuse)
    "model.layoutlmv3.embeddings",
    # Patch embed (vision branch)
    "model.layoutlmv3.patch_embed",
    # Final classifier
    "model.classifier",
]


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
    cka_n_samples: int = 500,
    fisher_n_samples: int = 200,
    task_order: list[int] | None = None,
    gradient_checkpointing: bool = False,
) -> dict:
    """Run one condition × seed of the pilot study.

    Args:
        condition: one of {"c1_bert", "c2_no_text", "c3_no_image", "c4_full"}
        seed: random seed
        output_dir: where to save results
        task_order: optional permutation of [0,1,2] over (FUNSD, CORD, SROIE) for
            the §6.1.3 stability check; default order if None.

    Returns:
        Summary dict of metrics (also saved to disk).
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 80)
    log.info(f"Pilot condition: {condition}, seed={seed}, order={task_order}, device={device}")
    log.info("=" * 80)

    # Build scenario
    scenario = build_pilot(order=task_order)

    # Build model based on condition
    model, modality_mask = _build_condition_model(condition, scenario)
    model = model.to(device)
    if gradient_checkpointing:
        model.enable_gradient_checkpointing()
        log.info("Gradient checkpointing enabled (lower memory, slower).")

    # Reservoirs for analysis
    cka_records: list[dict] = []  # one per task boundary
    fisher_records: list[dict] = []  # one per task
    accuracy_records: list[dict] = []  # one per task evaluation

    tracker = CLMetricsTracker(num_tasks=len(scenario.tasks))
    eval_loaders: dict[int, DataLoader] = {}

    prev_activations: dict[str, torch.Tensor] | None = None

    for task_idx, task in enumerate(scenario.tasks):
        log.info(f"\n--- Task {task_idx}: {task.task_name} ---")

        # Expand classifier for new labels
        new_labels = [l for l in task.label_set if l not in model.label_to_id]
        if new_labels:
            model.expand_classifier(new_labels)
            model = model.to(device)

        train_loader = DataLoader(
            scenario.train_datasets[task_idx],
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        eval_loader = DataLoader(
            scenario.eval_datasets[task_idx],
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        eval_loaders[task_idx] = eval_loader

        # ─── Capture pre-training activations & Fisher (before training task) ──
        if task_idx > 0:
            log.info("Capturing pre-training activations for CKA...")
            _ = _capture_activations(model, eval_loader, modality_mask, cka_n_samples, device)

        # ─── Train naively on this task ────────────────────────────────────────
        _train_naive(model, train_loader, modality_mask, epochs_per_task, device)

        # ─── Capture post-training activations ─────────────────────────────────
        log.info("Capturing post-training activations...")
        cur_activations = _capture_activations(
            model, eval_loader, modality_mask, cka_n_samples, device
        )

        # CKA: layer-wise similarity between consecutive checkpoints
        if prev_activations is not None:
            cka = {
                layer: linear_cka(prev_activations[layer], cur_activations[layer])
                for layer in cur_activations.keys() & prev_activations.keys()
            }
            cka_records.append({
                "task_boundary": f"{task_idx-1}_to_{task_idx}",
                "cka": cka,
            })
            log.info(f"CKA at boundary {task_idx-1}→{task_idx}: {cka}")
        prev_activations = cur_activations

        # ─── Fisher per group ──────────────────────────────────────────────────
        log.info("Computing Fisher information...")
        fisher = empirical_fisher_diagonal(
            model, train_loader, n_samples=fisher_n_samples, device=device
        )
        param_groups = model.param_groups
        fisher_grouped = fisher_per_group(fisher, param_groups, model)
        fisher_records.append({
            "task_idx": task_idx,
            "fisher_per_group": fisher_grouped,
        })
        log.info(f"Fisher per group: {fisher_grouped}")

        # ─── Evaluate on all seen tasks ────────────────────────────────────────
        log.info("Evaluating on all seen tasks...")
        eval_results = _evaluate_all(model, eval_loaders, modality_mask, device)
        for tid, m in eval_results.items():
            log.info(f"  Task {tid}: F1={m['f1']:.2f}")
        tracker.update(task_idx, eval_results)
        accuracy_records.append({
            "task_idx": task_idx,
            "results": eval_results,
        })

    # ─── Save results ──────────────────────────────────────────────────────────
    order = task_order or [0, 1, 2]
    summary = {
        "condition": condition,
        "seed": seed,
        "task_order": order,
        "cka_records": cka_records,
        "fisher_records": fisher_records,
        "accuracy_records": accuracy_records,
        "cl_metrics": tracker.summary(),
        "matrix": tracker.matrix.tolist(),
    }
    # Suffix non-default orders so the alternate-order stability runs do not collide
    # with the default-order results (both are pooled by doccl.pilot.analyze).
    order_suffix = "" if order == [0, 1, 2] else "_ord" + "".join(str(i) for i in order)
    out_path = output_dir / f"{condition}_seed{seed}{order_suffix}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nSaved pilot results to {out_path}")
    log.info(f"Final: AA={tracker.average_accuracy():.2f} BWT={tracker.backward_transfer():.2f}")

    return summary


def _build_condition_model(
    condition: str, scenario
) -> tuple[LayoutLMv3Wrapper, ModalityMask]:
    """Construct the model and modality mask for a pilot condition.

    For C1 (BERT), we instead use LayoutLMv3 with TEXT_ONLY mask. This is a
    compromise: a true BERT comparison would require a separate wrapper,
    but TEXT_ONLY mask gives architectural-equivalent unimodal text baseline.

    For full BERT comparison (true external baseline), implement BERTWrapper
    in models/bert_wrapper.py — deferred to W3 if time allows.
    """
    # Initial num_labels: 1 (will expand at first task)
    n_labels = len(scenario.tasks[0].label_set)
    model = LayoutLMv3Wrapper(num_labels=n_labels)

    # Pre-populate label map with first task's labels
    model.label_to_id = {l: i for i, l in enumerate(scenario.tasks[0].label_set)}
    model.id_to_label = {i: l for l, i in model.label_to_id.items()}

    if condition == "c4_full":
        return model, ModalityMask.FULL
    elif condition == "c3_no_image":
        return model, ModalityMask.TEXT_LAYOUT
    elif condition == "c2_no_text":
        return model, ModalityMask.IMAGE_LAYOUT
    elif condition == "c1_bert":
        # Use LayoutLMv3 with TEXT_ONLY mask as a same-architecture text baseline
        # (real BERT wrapper deferred to follow-up implementation)
        return model, ModalityMask.TEXT_ONLY
    else:
        raise ValueError(f"Unknown condition: {condition}")


def _train_naive(
    model: LayoutLMv3Wrapper,
    loader: DataLoader,
    modality_mask: ModalityMask,
    epochs: int,
    device: torch.device,
) -> None:
    """Naive sequential training for one task."""
    from tqdm import tqdm
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-5, weight_decay=0.01
    )
    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"ep{epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            optimizer.zero_grad()
            out = model(**batch, modality_mask=modality_mask)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix({"loss": f"{out.loss.item():.3f}"})


def _capture_activations(
    model: LayoutLMv3Wrapper,
    loader: DataLoader,
    modality_mask: ModalityMask,
    max_samples: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Wrapper around collect_activations that respects modality mask."""
    # Patch model.forward to use the modality_mask. Simpler: monkeypatch via lambda.
    original_forward = model.forward
    model.forward = lambda **kw: original_forward(**kw, modality_mask=modality_mask)
    try:
        acts = collect_activations(model, loader, LAYERS_TO_TRACK, max_samples, device)
    finally:
        model.forward = original_forward
    return acts


def _evaluate_all(
    model: LayoutLMv3Wrapper,
    eval_loaders: dict[int, DataLoader],
    modality_mask: ModalityMask,
    device: torch.device,
) -> dict[int, dict[str, float]]:
    """Evaluate on all seen tasks."""
    model.eval()
    out = {}
    with torch.no_grad():
        for tid, loader in eval_loaders.items():
            preds, golds = [], []
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
                outputs = model(
                    **{k: v for k, v in batch.items() if k != "labels"},
                    modality_mask=modality_mask,
                )
                p = outputs.logits.argmax(-1)
                lbl = batch["labels"]
                m = lbl != -100
                preds.extend(p[m].cpu().tolist())
                golds.extend(lbl[m].cpu().tolist())
            metrics = compute_token_f1(preds, golds, model.id_to_label)
            out[tid] = metrics
    return out


def main():
    """CLI entry: run all 4 conditions × 3 seeds."""
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+",
                        default=["c1_bert", "c2_no_text", "c3_no_image", "c4_full"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    parser.add_argument("--output_dir", type=Path, default=Path("results/pilot"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--task_order", nargs="+", type=int, default=None,
        help="Permutation of 0 1 2 over (FUNSD CORD SROIE); e.g. '2 1 0' for the "
             "alternate-order stability check (§6.1.3). Default: 0 1 2.",
    )
    parser.add_argument(
        "--cka_n_samples", type=int, default=500,
        help="Probe-set size for CKA (lower to save memory on small GPUs).",
    )
    parser.add_argument(
        "--fisher_n_samples", type=int, default=200,
        help="Sample count for the Fisher estimate (lower to save memory).",
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
