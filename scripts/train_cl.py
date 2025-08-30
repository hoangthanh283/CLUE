#!/usr/bin/env python3
"""
Continual Learning training script for LayoutLM-based IE.

Runs a sequence of tasks using a chosen CL strategy.
"""
import argparse
import copy
import warnings as _warnings
from pathlib import Path
from typing import Any, Dict, List

from transformers.utils import logging as hf_logging

from src.data.label_space import UNIFIED_LABEL2ID, UNIFIED_LABEL_LIST
from src.data.layoutlm_datasets import create_data_loader, get_dataset_loader
from src.models.layoutlm_models import get_model
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


def main():
    hf_logging.set_verbosity_error()
    parser = argparse.ArgumentParser(description="Continual Learning training for LayoutLM")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir) / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    config["output_dir"] = str(output_dir)

    logger = setup_logging(str(output_dir / "logs"), f"{config['experiment_name']}_cl")
    logger.info(f"Starting CL experiment: {config['experiment_name']}")
    logger.info(f"CL strategy: {config.get('cl_strategy', {}).get('name', 'none')}")

    # Build tasks list from config; if not provided, use a single task from base dataset config.
    tasks_cfg: List[Dict[str, Any]] = config.get("tasks", [{}])
    # Unified label space (single head across tasks).
    label_list = UNIFIED_LABEL_LIST
    label2id = UNIFIED_LABEL2ID
    id2label = {ii: ll for ii, ll in enumerate(label_list)}

    # Initialize model now (num_labels will be set after inferring labels from first task)
    # Prepare per-task dataloaders.
    tasks: List[Dict[str, Any]] = []
    for idx, task_overrides in enumerate(tasks_cfg):
        # Merge overrides into a per-task config copy
        task_name = task_overrides.get("name", f"task{idx}")
        task_config = deep_update(config, task_overrides)

        # Build dataset loader for this task
        dataset_loader = get_dataset_loader(task_config)
        train_dataset, test_dataset, val_dataset = dataset_loader.load_data()

        # Use unified label list for all tasks
        train_loader = create_data_loader(
            config=task_config,
            is_training=True,
            dataset_loader=dataset_loader,
            hf_dataset=train_dataset,
            tokenizer=dataset_loader.tokenizer,
            label2id=label2id,
        )
        eval_dataset = val_dataset if val_dataset else test_dataset
        eval_loader = create_data_loader(
            config=task_config,
            is_training=False,
            dataset_loader=dataset_loader,
            hf_dataset=eval_dataset,
            tokenizer=dataset_loader.tokenizer,
            label2id=label2id,
        )
        tasks.append({
            "name": task_name,
            "train_loader": train_loader,
            "eval_loader": eval_loader,
            "label_list": label_list,
        })

    if not tasks:
        raise ValueError("No tasks configured for CL training.")

    # Initialize model once, using first task label count.
    config["model"]["config"]["num_labels"] = len(label_list)
    model = get_model(config)

    # Build strategy & trainer.
    strategy = get_strategy(config)
    strat_name = (config.get("cl_strategy", {}).get("name") or "none").lower()
    if strat_name not in {"none", "sequential"}:
        # If strategy requires a shared label space, ensure tasks agree.
        first_label_list = tasks[0]["label_list"]
        first = set(first_label_list)
        for tt in tasks[1:]:
            if set(tt["label_list"]) != first:
                raise ValueError(f"CL strategy '{strat_name}' requires a shared label space across tasks; tasks have "
                                 "mismatched label sets.")

    trainer = create_continual_trainer(
        model=model,
        config=config,
        label_list=label_list,
        id2label=id2label,
        strategy=strategy,
    )
    results = trainer.train(tasks)
    logger.info(f"CL training completed. Per-task results: {results}")

    # Evaluate each task with the final backbone and the unified head.
    eval_summary: Dict[str, Any] = {}
    for tt in tasks:
        name = tt["name"]
        metrics = trainer.evaluate(tt["eval_loader"])
        logger.info(f"Final eval ({name}): {metrics}")
        eval_summary[name] = metrics

    # Save final model.
    final_model_path = output_dir / "final_model"
    trainer.model.save_pretrained(str(final_model_path))
    logger.info(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
