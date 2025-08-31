#!/usr/bin/env python3
"""
Continual Learning training script for LayoutLM-based IE.

Runs a sequence of tasks using a chosen CL strategy.
"""
import argparse
import copy
import csv
import json
import warnings as _warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

from torch.utils.data import ConcatDataset, DataLoader
from transformers.utils import logging as hf_logging

from src.data.label_space import UNIFIED_LABEL2ID, UNIFIED_LABEL_LIST
from src.data.layoutlm_datasets import LayoutLMDataset, get_dataset_loader
from src.models.layoutlm_models import get_model
from src.training.cl_metrics import save_aaa_curve_plot
from src.training.continual_trainer import (create_continual_trainer,
                                            get_strategy)
from src.utils import load_config, setup_logging

_warnings.filterwarnings(
    "ignore",
    message=r".*`device` argument is deprecated.*",
    category=FutureWarning,
    module=r".*transformers.*",
)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for kk, vv in updates.items():
        if isinstance(vv, dict) and isinstance(out.get(kk), dict):
            out[kk] = deep_update(out[kk], vv)
        else:
            out[kk] = vv
    return out


def _setup_experiment(args: argparse.Namespace) -> Tuple[Dict[str, Any], Path, Any]:
    config = load_config(args.config)
    output_dir = Path(args.output_dir) / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output_dir)
    logger = setup_logging(str(output_dir / "logs"), f"{config['experiment_name']}_cl")
    return config, output_dir, logger


def _make_loader_from_dataset(ds, config: Dict[str, Any], is_training: bool) -> DataLoader:
    workers = int(config["training"].get("num_workers", 4))
    dl_kwargs = {
        "batch_size": config["training"]["batch_size"],
        "shuffle": is_training and config.get("data_processing", {}).get("shuffle_train", False),
        "num_workers": workers,
        "pin_memory": True,
    }
    if workers > 0:
        if "persistent_workers" in config["training"]:
            dl_kwargs["persistent_workers"] = bool(config["training"]["persistent_workers"])
        if "prefetch_factor" in config["training"]:
            dl_kwargs["prefetch_factor"] = int(config["training"]["prefetch_factor"])
    return DataLoader(ds, **dl_kwargs)


def _build_tasks(config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int, str]:
    tasks_cfg: List[Dict[str, Any]] = config.get("tasks", [{}])
    cl_setting = (config.get("cl_setting") or "task_il").lower()
    tasks: List[Dict[str, Any]] = []
    first_num_labels: int = -1

    global_labels: List[str] = []
    global_label2id: Dict[str, int] = {}

    strat_name = (config.get("cl_strategy", {}).get("name") or "none").lower()
    is_joint = strat_name == "joint"

    for idx, task_overrides in enumerate(tasks_cfg):
        task_name = task_overrides.get("name", f"task{idx}")
        task_config = deep_update(config, task_overrides)
        if cl_setting == "class_il" and config.get("label_space", {}).get("unified", False):
            task_config = deep_update(task_config, {"label_space": {"unified": True}})

        dataset_loader = get_dataset_loader(task_config)
        train_dataset, test_dataset, val_dataset = dataset_loader.load_data()

        if cl_setting == "class_il" and config.get("label_space", {}).get("unified", False):
            label_list_use = list(UNIFIED_LABEL_LIST)
            label2id_use = dict(UNIFIED_LABEL2ID)
        elif cl_setting == "class_il":
            for lab in list(dataset_loader.label_list):
                if lab not in global_label2id:
                    global_label2id[lab] = len(global_labels)
                    global_labels.append(lab)
            label_list_use = list(global_labels)
            label2id_use = dict(global_label2id)
        else:
            label_list_use = list(dataset_loader.label_list)
            label2id_use = {l: i for i, l in enumerate(label_list_use)}

        if first_num_labels == -1:
            first_num_labels = len(label_list_use)

        # Build per-task datasets so we can optionally concat for joint training
        train_ds = LayoutLMDataset(
            dataset_loader=dataset_loader,
            hf_dataset=train_dataset,
            tokenizer=dataset_loader.tokenizer,
            label2id=label2id_use,
            max_seq_length=task_config["dataset"]["preprocessing"]["max_seq_length"],
        )
        eval_dataset = val_dataset if val_dataset else test_dataset
        eval_ds = LayoutLMDataset(
            dataset_loader=dataset_loader,
            hf_dataset=eval_dataset,
            tokenizer=dataset_loader.tokenizer,
            label2id=label2id_use,
            max_seq_length=task_config["dataset"]["preprocessing"]["max_seq_length"],
        )

        # For joint, cumulatively concatenate training datasets (Class-IL + unified only)
        if is_joint:
            if not (
                config.get("cl_setting", "class_il").lower() == "class_il"
                and config.get("label_space", {}).get("unified", False)
            ):
                raise ValueError(
                    "Joint training baseline requires class_il with label_space.unified: true"
                )
            if idx == 0:
                cum_train_ds = train_ds
            else:
                # Concat with previous cumulative dataset from last task entry
                prev = tasks[-1]["_cum_train_ds"]
                cum_train_ds = ConcatDataset([prev, train_ds])
            train_loader = _make_loader_from_dataset(cum_train_ds, task_config, is_training=True)
        else:
            train_loader = _make_loader_from_dataset(train_ds, task_config, is_training=True)

        eval_loader = _make_loader_from_dataset(eval_ds, task_config, is_training=False)
        id2label_use = {ii: ll for ii, ll in enumerate(label_list_use)}
        tasks.append({
            "name": task_name,
            "train_loader": train_loader,
            "eval_loader": eval_loader,
            "label_list": label_list_use,
            "id2label": id2label_use,
            # keep references to datasets for joint accumulation
            "_train_ds": train_ds,
            "_eval_ds": eval_ds,
            "_cum_train_ds": cum_train_ds if is_joint else train_ds,
        })
    if not tasks:
        raise ValueError("No tasks configured for CL training.")
    return tasks, first_num_labels, cl_setting


def _validate_strategy(config: Dict[str, Any], tasks: List[Dict[str, Any]], cl_setting: str) -> None:
    strat_name = (config.get("cl_strategy", {}).get("name") or "none").lower()
    if strat_name not in {"none", "sequential"} and cl_setting != "class_il":
        first = set(tasks[0]["label_list"])
        for tt in tasks[1:]:
            if set(tt["label_list"]) != first:
                raise ValueError(f"CL strategy {strat_name} requires a shared label space across tasks; tasks have "
                                 "mismatched label sets.")


def _save_cl_artifacts(output_dir: Path, results: Dict[str, Any], config: Dict[str, Any], logger: Any) -> None:
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
            fp.write(f"# Continual Learning Report - {config['experiment_name']}\n\n")
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
    for tt in tasks:
        name = tt["name"]
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
    logger.info(f"Starting CL experiment: {config['experiment_name']}")
    logger.info(f"CL strategy: {config.get('cl_strategy', {}).get('name', 'none')}")

    tasks, first_num_labels, cl_setting = _build_tasks(config)
    if first_num_labels == -1:
        raise ValueError("Could not infer first task label count.")

    # Initialize model and trainer.
    # if cl_setting == "class_il" and config.get("label_space", {}).get("unified", False):
    #     config["model"]["config"]["num_labels"] = len(UNIFIED_LABEL_LIST)
    # else:
    #     config["model"]["config"]["num_labels"] = int(first_num_labels)
    config["model"]["config"]["num_labels"] = int(first_num_labels)
    model = get_model(config)
    strategy = get_strategy(config)
    _validate_strategy(config, tasks, cl_setting)
    trainer = create_continual_trainer(
        model=model,
        config=config,
        label_list=tasks[0]["label_list"],
        id2label=tasks[0]["id2label"],
        strategy=strategy,
    )

    results = trainer.train(tasks)
    _save_cl_artifacts(output_dir, results, config, logger)
    _final_eval_and_save(trainer, tasks, output_dir, logger)


if __name__ == "__main__":
    main()
