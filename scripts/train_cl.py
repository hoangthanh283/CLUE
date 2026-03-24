#!/usr/bin/env python3
"""
Continual Learning training script for LayoutLM-based IE.

Runs a sequence of tasks using a chosen CL strategy.
"""
import argparse
import csv
import dataclasses
import json
import warnings as _warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

from torch.utils.data import ConcatDataset, DataLoader
from transformers.utils import logging as hf_logging

from src.config import ExperimentConfig
from src.data.label_space import UNIFIED_LABEL2ID, UNIFIED_LABEL_LIST
from src.data.layoutlm_datasets import LayoutLMDataset, get_dataset_loader
from src.models.layoutlm_models import get_model
from src.training.cl_metrics import save_aaa_curve_plot
from src.training.continual_trainer import create_continual_trainer, get_strategy
from src.utils import load_config, setup_logging

_warnings.filterwarnings(
    "ignore",
    message=r".*`device` argument is deprecated.*",
    category=FutureWarning,
    module=r".*transformers.*",
)


def _setup_experiment(args: argparse.Namespace) -> Tuple[ExperimentConfig, Path, Any]:
    raw = load_config(args.config)
    config = ExperimentConfig.from_dict(raw)
    output_dir = Path(args.output_dir) / config.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config = config.with_output_dir(str(output_dir))
    logger = setup_logging(str(output_dir / "logs"), f"{config.experiment_name}_cl")
    return config, output_dir, logger


def _make_loader_from_dataset(ds, config: ExperimentConfig, is_training: bool) -> DataLoader:
    tc = config.training
    workers = tc.num_workers
    dl_kwargs = {
        "batch_size": tc.batch_size,
        "shuffle": is_training and bool(config.data_processing.get("shuffle_train", False)),
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        if tc.persistent_workers is not None:
            dl_kwargs["persistent_workers"] = tc.persistent_workers
        if tc.prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = tc.prefetch_factor
    return DataLoader(ds, **dl_kwargs)


def _determine_label_space(config: ExperimentConfig, cl_setting, is_joint, joint_fixed_labels, joint_fixed_label2id,
                           global_labels, global_label2id, dataset_loader, is_true_joint):
    """Determine label space for current task based on CL setting and strategy."""
    if cl_setting == "class_il" and config.label_space.get("unified", False):
        return UNIFIED_LABEL_LIST, UNIFIED_LABEL2ID

    if cl_setting == "class_il":
        if is_joint and joint_fixed_labels and not is_true_joint:
            return list(joint_fixed_labels), dict(joint_fixed_label2id)
        for lab in list(dataset_loader.get_label_list()):
            if lab not in global_label2id:
                global_label2id[lab] = len(global_labels)
                global_labels.append(lab)
        return list(global_labels), dict(global_label2id)

    label_list_use = list(dataset_loader.get_label_list())
    return label_list_use, {l: i for i, l in enumerate(label_list_use)}


def _create_joint_datasets(tasks, first_task_config):
    """Combine all individual datasets into a single joint training task."""
    all_train_datasets = [t["_train_ds"] for t in tasks]
    all_eval_datasets = [t["_eval_ds"] for t in tasks]

    combined_train_ds = ConcatDataset(all_train_datasets)
    combined_eval_ds = ConcatDataset(all_eval_datasets)

    combined_train_loader = _make_loader_from_dataset(combined_train_ds, first_task_config, is_training=True)
    combined_eval_loader = _make_loader_from_dataset(combined_eval_ds, first_task_config, is_training=False)

    joint_task = {
        "name": "joint_all",
        "train_loader": combined_train_loader,
        "eval_loader": combined_eval_loader,
        "label_list": tasks[0]["label_list"],
        "id2label": tasks[0]["id2label"],
        "_train_ds": combined_train_ds,
        "_eval_ds": combined_eval_ds,
        "_cum_train_ds": combined_train_ds,
        "_cum_eval_ds": combined_eval_ds,
    }

    for tt in tasks:
        if tt["eval_loader"] is None:
            tt["eval_loader"] = _make_loader_from_dataset(tt["_eval_ds"], first_task_config, is_training=False)

    return joint_task, tasks


def _to_legacy_dict(config: ExperimentConfig) -> Dict[str, Any]:
    """Convert ExperimentConfig to legacy dict shape for components not yet ported (e.g. dataset loaders)."""
    d = {
        "experiment_name": config.experiment_name,
        "output_dir": config.output_dir,
        "cl_setting": config.cl_setting,
        "model": dataclasses.asdict(config.model),
        "training": dataclasses.asdict(config.training),
        "cl_strategy": dataclasses.asdict(config.cl_strategy),
        "label_space": config.label_space,
        "dataset": config.dataset,
        "data_processing": config.data_processing,
        "tasks": config.tasks,
        "evaluation": config.evaluation,
        "output": config.output,
        "neptune": config.neptune,
    }
    return d


def _build_tasks(config: ExperimentConfig) -> Tuple[List[Dict[str, Any]], int, str]:
    tasks_cfg: List[Dict[str, Any]] = config.tasks or [{}]
    cl_setting = config.cl_setting or "task_il"
    tasks: List[Dict[str, Any]] = []
    first_num_labels: int = -1

    # For sequential/ER-like flows we may grow union progressively.
    global_labels: List[str] = []
    global_label2id: Dict[str, int] = {}

    strat_name = (config.cl_strategy.name or "none").lower()
    is_joint = strat_name == "joint"
    # Check if we want true joint training (all datasets at once) or progressive joint
    is_true_joint = is_joint and getattr(config.cl_strategy, "true_joint", True)

    # Pre-compute a fixed global label space for joint baseline when not using unified labels.
    joint_fixed_labels: List[str] = []
    joint_fixed_label2id: Dict[str, int] = {}
    if is_joint and cl_setting == "class_il" and not config.label_space.get("unified", False):
        seen: Dict[str, int] = {}
        for idx, task_overrides in enumerate(tasks_cfg):
            task_config = config.with_task_overrides(task_overrides)
            dataset_loader = get_dataset_loader(_to_legacy_dict(task_config))
            for lab in list(dataset_loader.get_label_list()):
                if lab not in seen:
                    seen[lab] = len(joint_fixed_labels)
                    joint_fixed_labels.append(lab)
        joint_fixed_label2id = dict(seen)

    for idx, task_overrides in enumerate(tasks_cfg):
        task_name = task_overrides.get("name", f"task{idx}")
        task_config = config.with_task_overrides(task_overrides)
        if cl_setting == "class_il" and config.label_space.get("unified", False):
            task_config = dataclasses.replace(task_config, label_space={**task_config.label_space, "unified": True})
        elif cl_setting == "task_il":
            # For task-IL, explicitly disable unified labels to use native task labels
            task_config = dataclasses.replace(task_config, label_space={**task_config.label_space, "unified": False})

        task_config_dict = _to_legacy_dict(task_config)
        dataset_loader = get_dataset_loader(task_config_dict)
        train_dataset, test_dataset, val_dataset = dataset_loader.load_data()

        label_list_use, label2id_use = _determine_label_space(
            config, cl_setting, is_joint, joint_fixed_labels, joint_fixed_label2id,
            global_labels, global_label2id, dataset_loader, is_true_joint
        )

        if first_num_labels == -1:
            first_num_labels = len(label_list_use)

        max_seq_length = task_config.dataset.get("preprocessing", {}).get("max_seq_length", 512)

        # Build per-task datasets so we can optionally concat for joint training.
        train_ds = LayoutLMDataset(
            dataset_loader=dataset_loader,
            hf_dataset=train_dataset,
            tokenizer=dataset_loader.tokenizer,
            label2id=label2id_use,
            max_seq_length=max_seq_length,
        )
        eval_dataset = val_dataset if val_dataset else test_dataset
        eval_ds = LayoutLMDataset(
            dataset_loader=dataset_loader,
            hf_dataset=eval_dataset,
            tokenizer=dataset_loader.tokenizer,
            label2id=label2id_use,
            max_seq_length=max_seq_length,
        )

        if is_joint:
            # Joint baseline now supports both unified and non-unified (union-of-labels) label spaces.
            # Requires class-IL (single head) semantics.
            if config.cl_setting.lower() != "class_il":
                raise ValueError("Joint training baseline requires cl_setting: class_il")

            if not is_true_joint:
                # Progressive joint: accumulate datasets over tasks.
                if idx == 0:
                    cum_train_ds = train_ds
                    cum_eval_ds = eval_ds
                else:
                    # Concat with previous cumulative dataset from last task entry.
                    prev_train = tasks[-1]["_cum_train_ds"]
                    prev_eval = tasks[-1]["_cum_eval_ds"]
                    cum_train_ds = ConcatDataset([prev_train, train_ds])
                    cum_eval_ds = ConcatDataset([prev_eval, eval_ds])
                train_loader = _make_loader_from_dataset(cum_train_ds, task_config, is_training=True)
                eval_loader = _make_loader_from_dataset(cum_eval_ds, task_config, is_training=False)
                cum_train_ds_use = cum_train_ds
                cum_eval_ds_use = cum_eval_ds
            else:
                # True joint: will be handled later by collecting all datasets.
                train_loader = None
                eval_loader = None
                cum_train_ds_use = train_ds
                cum_eval_ds_use = eval_ds
        else:
            train_loader = _make_loader_from_dataset(train_ds, task_config, is_training=True)
            eval_loader = _make_loader_from_dataset(eval_ds, task_config, is_training=False)
            cum_train_ds_use = train_ds
            cum_eval_ds_use = eval_ds

        id2label_use = {ii: ll for ii, ll in enumerate(label_list_use)}
        tasks.append({
            "name": task_name,
            "train_loader": train_loader,
            "eval_loader": eval_loader,
            "label_list": label_list_use,
            "id2label": id2label_use,
            # Keep references to datasets for joint accumulation.
            "_train_ds": train_ds,
            "_eval_ds": eval_ds,
            "_cum_train_ds": cum_train_ds_use,
            "_cum_eval_ds": cum_eval_ds_use,
        })
    if not tasks:
        raise ValueError("No tasks configured for CL training.")

    if is_joint and is_true_joint:
        first_task_config = config.with_task_overrides(tasks_cfg[0])
        joint_task, tasks_for_eval = _create_joint_datasets(tasks, first_task_config)
        return ([joint_task], tasks_for_eval), first_num_labels, cl_setting

    return tasks, first_num_labels, cl_setting


def _validate_strategy(config: ExperimentConfig, tasks: List[Dict[str, Any]], cl_setting: str) -> None:
    strat_name = (config.cl_strategy.name or "none").lower()
    # Extra guardrails for strategies with specific requirements
    if strat_name in {"lwf"}:
        # LwF uses distillation between student and a frozen teacher; this
        # implementation assumes a single, fixed-size head (same logits dim)
        # across tasks. Enforce class-IL with unified label space.
        if cl_setting != "class_il" or not config.label_space.get("unified", False):
            raise ValueError(
                "LwF requires cl_setting: class_il and label_space.unified: true to keep logits dimensions stable."
            )
    # Note: GEM, EWC, and AGEM work with both task-IL and class-IL
    # They don't inherently require unified label space - they just need memory/importance weights


def _save_cl_artifacts(output_dir: Path, results: Dict[str, Any], config: ExperimentConfig, logger: Any) -> None:
    try:
        cl_out_path = output_dir / "cl_results.json"
        with open(cl_out_path, "w") as fp:
            json.dump(results, fp, indent=2)

        acc_csv = output_dir / "accuracy_matrix.csv"
        with open(acc_csv, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["after_task\\on_task"] + results["task_names"])
            for i, row in enumerate(results["accuracy_matrix"]):
                writer.writerow([results["task_names"][i]] + [f"{v:.6f}" for v in row])

        logger.info(
            f"CL training completed. Metrics: ACC={results['cl_metrics']['ACC']:.4f}, "
            f"BWT={results['cl_metrics']['BWT']:.4f}, FWT={results['cl_metrics']['FWT']:.4f}, "
            f"AAA={results['cl_metrics']['AAA']:.4f}, Forgetting={results['cl_metrics']['Forgetting']:.4f}"
        )
        logger.info(f"Saved CL results to: {cl_out_path}")
    except Exception as er:
        logger.warning(f"Failed to persist CL results: {er}")

    try:
        aaa_png = output_dir / "aaa_curve.png"
        save_aaa_curve_plot(results["cl_metrics"]["AAA_curve"], results["task_names"], str(aaa_png))
        report_md = output_dir / "CL_REPORT.md"
        with open(report_md, "w") as fp:
            fp.write(f"# Continual Learning Report - {config.experiment_name}\n\n")
            fp.write("## Summary Metrics\n")
            fp.write(f"- ACC: {results['cl_metrics']['ACC']:.4f}\n")
            fp.write(f"- BWT: {results['cl_metrics']['BWT']:.4f}\n")
            fp.write(f"- FWT: {results['cl_metrics']['FWT']:.4f}\n")
            fp.write(f"- AAA: {results['cl_metrics']['AAA']:.4f}\n")
            fp.write(f"- Forgetting: {results['cl_metrics']['Forgetting']:.4f}\n\n")
            fp.write("## Task Order\n")
            fp.write("- " + " → ".join(results["task_names"]) + "\n\n")
            fp.write("## AAA Curve\n")
            fp.write("![AAA Curve](aaa_curve.png)\n\n")
            fp.write("## Artifacts\n")
            fp.write("- `cl_results.json`\n")
            fp.write("- `accuracy_matrix.csv`\n")
            fp.write("- `aaa_curve.png`\n")
        logger.info(f"Wrote CL report and AAA plot to: {report_md}")
    except Exception as er:
        logger.warning(f"Failed to save AAA plot/report: {er}")


def _final_eval_and_save(trainer, tasks: List[Dict[str, Any]], output_dir: Path, logger: Any) -> None:
    eval_summary: Dict[str, Any] = {}
    # Use the appropriate evaluation depending on CL setting
    for tt in tasks:
        name = tt["name"]
        if getattr(trainer, "cl_setting", "task_il") == "task_il":
            # Activate per-task head and metrics to avoid label-id mismatches
            lbl_list = tt.get("label_list") or trainer.metrics.label_list
            id2label = tt.get("id2label") or {i: l for i, l in enumerate(lbl_list)}
            metrics = trainer.evaluate_with_head(tt["eval_loader"], name, lbl_list, id2label)
        else:
            # Single head (class-IL): current head is global; evaluate directly
            metrics = trainer.evaluate(tt["eval_loader"])
        logger.info(f"Final eval ({name}): {metrics}")
        eval_summary[name] = metrics
    final_model_path = output_dir / "final_model"
    trainer.model.save_pretrained(str(final_model_path))
    logger.info(f"Final model saved to: {final_model_path}")


def main():
    hf_logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description="Continual Learning training for LayoutLM")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    config, output_dir, logger = _setup_experiment(args)
    logger.info(f"Starting CL experiment: {config.experiment_name}")
    logger.info(f"CL strategy: {config.cl_strategy.name}")

    tasks_result, first_num_labels, cl_setting = _build_tasks(config)
    # Handle true joint training case where we get (training_tasks, eval_tasks) tuple
    if isinstance(tasks_result, tuple):
        tasks_for_training, tasks_for_eval = tasks_result
        is_true_joint = True
        logger.info("Using TRUE JOINT TRAINING: all datasets combined in one training run")
    else:
        tasks_for_training = tasks_result
        tasks_for_eval = tasks_result
        is_true_joint = False

    if first_num_labels == -1:
        raise ValueError("Could not infer first task label count.")

    # Initialize model and trainer.
    if cl_setting == "class_il" and config.label_space.get("unified", False):
        config = config.replace_model_num_labels(len(UNIFIED_LABEL_LIST))
    else:
        config = config.replace_model_num_labels(first_num_labels)

    # Initialize model and trainer.
    model = get_model(config)
    strategy = get_strategy(config)
    _validate_strategy(config, tasks_for_training, cl_setting)
    trainer = create_continual_trainer(
        model=model,
        config=config,
        label_list=tasks_for_training[0]["label_list"],
        id2label=tasks_for_training[0]["id2label"],
        strategy=strategy,
    )

    # Train using tasks_for_training, but evaluate on tasks_for_eval
    if is_true_joint:
        # For true joint, pass both training and eval tasks
        results = trainer.train(tasks_for_training, eval_tasks=tasks_for_eval)
    else:
        # For other strategies, use the same task list for both
        results = trainer.train(tasks_for_training)

    _save_cl_artifacts(output_dir, results, config, logger)
    _final_eval_and_save(trainer, tasks_for_eval, output_dir, logger)


if __name__ == "__main__":
    main()
