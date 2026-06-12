"""Main training entry point for CL experiments.

Usage:
    python scripts/train.py method=naive scenario=single_funsd
    python scripts/train.py method=ewc scenario=cil_cord seed=42
    python scripts/train.py method=der_pp scenario=cil_cord training.epochs=15
"""
from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from doccl.data.scenarios import get_scenario
from doccl.eval.metrics import CLMetricsTracker
from doccl.methods.coda_prompt import CODAPrompt
from doccl.methods.der import DERpp
from doccl.methods.doccl import DocCL_A, DocCL_B, DocCL_C
from doccl.methods.dualprompt import DualPrompt
from doccl.methods.er import ER
from doccl.methods.ewc import EWC
from doccl.methods.l2p import L2P
from doccl.methods.lwf import LwF
from doccl.methods.naive import JointMultiTask, NaiveFineTune
from doccl.methods.o_lora import OLoRA
from doccl.models.layoutlm_wrapper import LayoutLMv3Wrapper

log = logging.getLogger(__name__)


METHOD_REGISTRY = {
    "naive": NaiveFineTune,
    "joint": JointMultiTask,
    "ewc": EWC,
    "lwf": LwF,
    "er": ER,
    "der_pp": DERpp,
    "o_lora": OLoRA,
    "l2p": L2P,
    "dualprompt": DualPrompt,
    "coda_prompt": CODAPrompt,
    # Proposed-method candidates — pre-implemented; one is selected at the Week-4
    # pilot decision gate (CLAUDE.md). Each accepts method.target_component for the
    # component-targeting ablation (Table 6.2).
    "doccl_a": DocCL_A,
    "doccl_b": DocCL_B,
    "doccl_c": DocCL_C,
    # PLACEHOLDER alias for the proposed method. At the Week-4 gate, repoint this to
    # the pilot-selected winner (doccl_a/b/c). Not a pre-lock: the Phase-5 grid for
    # `doccl` is only run *after* the diagnosis selects the candidate.
    "doccl": DocCL_A,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_run_metrics(
    out_dir: Path,
    cfg: DictConfig,
    tracker: CLMetricsTracker,
    method,
    task_times: list[float],
) -> None:
    """Persist a structured per-run ``metrics.json`` for offline ingestion.

    Decouples ``scripts/analyze_results.py`` from W&B: on an offline Vast.ai box
    the LaTeX result tables (6.1/6.3) and forgetting curves are built directly
    from ``results/<run>/{matrix.npy,metrics.json}``. Records the CL metrics, the
    accuracy matrix, per-task wall time, parameter counts, and peak GPU memory
    (the raw signals behind the computational-overhead table).
    """
    peak_mem_mb = None
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    metrics = {
        "method": cfg.method.name,
        "scenario": cfg.scenario.name,
        "seed": int(cfg.seed),
        "target_component": cfg.method.get("target_component"),
        **tracker.to_dict(),
        "wall_time_per_task_s": [float(t) for t in task_times],
        "total_wall_time_s": float(sum(task_times)),
        "mean_time_per_task_s": float(sum(task_times) / max(len(task_times), 1)),
        "total_params": int(method.total_param_count()),
        "trainable_params": int(method.trainable_param_count()),
        "peak_gpu_mem_mb": peak_mem_mb,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Saved metrics to %s", out_dir / "metrics.json")


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    # ─── W&B init ──────────────────────────────────────────────────────────────
    # Include target_component in the run name so component-targeting ablation runs
    # (same method, different bank) get distinct result dirs and are identifiable.
    run_name = f"{cfg.scenario.name}_{cfg.method.name}_seed{cfg.seed}"
    target_component = cfg.method.get("target_component")
    if target_component is not None:
        run_name += f"_{target_component}"
    run = wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[cfg.scenario.name, cfg.method.name, f"seed{cfg.seed}"],
        mode=cfg.wandb.get("mode", "online"),
    )

    # ─── Build scenario ────────────────────────────────────────────────────────
    scenario = get_scenario(cfg.scenario.name, **(cfg.scenario.get("kwargs") or {}))
    log.info("Scenario %s: %d tasks", scenario.name, len(scenario.tasks))

    # ─── Build model ───────────────────────────────────────────────────────────
    # Initial num_labels = first task's label set size
    n_init_labels = len(scenario.tasks[0].label_set)
    model = LayoutLMv3Wrapper(
        model_name=cfg.model.name,
        num_labels=n_init_labels,
    )
    # Pre-populate label maps for first task
    model.label_to_id = {l: i for i, l in enumerate(scenario.tasks[0].label_set)}
    model.id_to_label = {i: l for l, i in model.label_to_id.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    log.info(
        "Model: %s. Total params: %d, Trainable: %d",
        cfg.model.name,
        model.total_param_count(),
        model.trainable_param_count(),
    )

    # ─── Build method ──────────────────────────────────────────────────────────
    method_cls = METHOD_REGISTRY[cfg.method.name]
    method = method_cls(model, OmegaConf.to_container(cfg.method, resolve=True))

    # Activation checkpointing (after any PEFT wrapping) — fits small-VRAM GPUs.
    if cfg.training.get("gradient_checkpointing", False):
        model.enable_gradient_checkpointing()
        log.info("Gradient checkpointing enabled (lower memory, ~20-30%% slower).")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ─── CL loop ───────────────────────────────────────────────────────────────
    tracker = CLMetricsTracker(num_tasks=len(scenario.tasks))
    eval_loaders_seen: dict[int, DataLoader] = {}
    out_dir = Path(cfg.output_dir) / run.name

    # Special path for Joint: concatenate all train datasets and treat as 1 task
    if cfg.method.name == "joint":
        log.info("Joint training mode: concatenating all train datasets")
        # Expand classifier to cover ALL labels across all tasks
        all_labels: list[str] = []
        for t in scenario.tasks:
            for l in t.label_set:
                if l not in model.label_to_id and l not in all_labels:
                    all_labels.append(l)
        if all_labels:
            model.expand_classifier(all_labels)
            model = model.to(device)

        # Use the full-label joint pool when the scenario provides one (CIL scenarios mask
        # out-of-session entities per task, so concatenating train_datasets would feed the
        # same document with conflicting labels and collapse training). Falls back to the
        # per-task datasets for scenarios whose tasks are disjoint documents (e.g. DIL).
        joint_sources = scenario.joint_train_datasets or scenario.train_datasets
        joint_train = ConcatDataset(joint_sources)
        joint_loader = DataLoader(
            joint_train,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )
        # Train once on joint data
        synthetic_task = scenario.tasks[0]
        t0 = time.perf_counter()
        method.train_task(synthetic_task, joint_loader)
        joint_time = time.perf_counter() - t0

        # Evaluate on each task's eval set
        for tid, eval_ds in enumerate(scenario.eval_datasets):
            eval_loaders_seen[tid] = DataLoader(
                eval_ds,
                batch_size=cfg.training.batch_size,
                shuffle=False,
                num_workers=cfg.training.num_workers,
                pin_memory=True,
            )
        results = method.evaluate(eval_loaders_seen)
        # Fill last row of matrix (since Joint = oracle = T-1 step)
        for tid in range(len(scenario.tasks)):
            tracker.matrix[len(scenario.tasks) - 1, tid] = results[tid].f1

        wandb.log({f"final/eval/task_{tid}/f1": r.f1 for tid, r in results.items()})
        wandb.log({"final/AA": tracker.average_accuracy()})
        log.info("Joint final AA: %.2f", tracker.average_accuracy())

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "matrix.npy", tracker.matrix)
        save_run_metrics(out_dir, cfg, tracker, method, [joint_time])

        wandb.finish()
        return

    # Standard CL loop
    task_times: list[float] = []
    for task_idx, task in enumerate(scenario.tasks):
        log.info("=== Task %d/%d: %s ===", task_idx + 1, len(scenario.tasks), task.task_name)

        # Expand classifier for new labels in this task
        new_labels = [l for l in task.label_set if l not in model.label_to_id]
        if new_labels:
            log.info("Expanding classifier with %d new labels", len(new_labels))
            model.expand_classifier(new_labels)
            model = model.to(device)

        train_loader = DataLoader(
            scenario.train_datasets[task_idx],
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )
        eval_loader = DataLoader(
            scenario.eval_datasets[task_idx],
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
        )
        eval_loaders_seen[task_idx] = eval_loader

        # Lifecycle
        method.before_task(task, train_loader)
        t0 = time.perf_counter()
        train_metrics = method.train_task(task, train_loader)
        task_times.append(time.perf_counter() - t0)
        method.after_task(task, train_loader)

        # Evaluate on all seen tasks
        results = method.evaluate(eval_loaders_seen)
        tracker.update(task_idx, {tid: {"f1": r.f1} for tid, r in results.items()})

        wandb.log(
            {
                "task_idx": task_idx,
                "train/loss": train_metrics.loss,
                **{f"eval/task_{tid}/f1": r.f1 for tid, r in results.items()},
            }
        )
        log.info(
            "After task %d: %s",
            task_idx,
            {tid: f"F1={r.f1:.2f}" for tid, r in results.items()},
        )

    # ─── Final summary ─────────────────────────────────────────────────────────
    summary = tracker.summary()
    log.info("Final: AA=%.2f BWT=%.2f AF=%.2f", summary["AA"], summary["BWT"], summary["AF"])
    wandb.log({"final/" + k: v for k, v in summary.items()})

    # Save tracker matrix + structured metrics for offline ingestion
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "matrix.npy", tracker.matrix)
    log.info("Saved matrix to %s", out_dir)
    save_run_metrics(out_dir, cfg, tracker, method, task_times)

    wandb.finish()


if __name__ == "__main__":
    main()
