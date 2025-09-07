"""
Continual Learning trainer wiring CL strategies into the LayoutLM pipeline.
"""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.cl_strategies import (AGEM, EWC, GEM, BaseCLStrategy,
                               ExperienceReplay, LwF, SequentialFineTuning)
from src.models.layoutlm_models import BaseLayoutLMModel, LayoutLMMetrics
from src.training.cl_metrics import compute_cl_metrics

logger = logging.getLogger(__name__)


STRATEGY_MAP = {
    "none": SequentialFineTuning,
    "sequential": SequentialFineTuning,
    "joint": SequentialFineTuning,
    "er": ExperienceReplay,
    "experience_replay": ExperienceReplay,
    "ewc": EWC,
    "lwf": LwF,
    "agem": AGEM,
    "a-gem": AGEM,
    "gem": GEM,
}


def get_strategy(config: Dict[str, Any]) -> BaseCLStrategy:
    name = (config.get("cl_strategy", {}).get("name") or "none").lower()
    if name not in STRATEGY_MAP:
        raise ValueError(f"Unknown CL strategy '{name}'. Available: {list(STRATEGY_MAP.keys())}")
    strategy_cls = STRATEGY_MAP[name]
    return strategy_cls(config)


class ContinualLayoutLMTrainer:
    """Continual training over a sequence of tasks with a chosen CL strategy."""

    def __init__(
        self,
        model: BaseLayoutLMModel,
        config: Dict[str, Any],
        label_list: List[str],
        id2label: Dict[int, str],
        strategy: Optional[BaseCLStrategy] = None,
    ):
        self.model = model
        self.config = config
        self.training_config = config["training"]
        self.strategy = strategy or get_strategy(config)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Metrics (set per task as label spaces can change under sequential FT)
        self.metrics = LayoutLMMetrics(label_list, id2label)

        # Per-task classifier head states (state_dicts), keyed by task name
        self.head_states: Dict[str, Dict[str, torch.Tensor]] = {}

        # Optimizer (single across tasks)
        self.optimizer = self._setup_optimizer()

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Output directory
        self.output_dir = Path(config.get("output_dir", "results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        self.early_stopping_patience = int(self.training_config.get("early_stopping_patience", 10))

        # Logging
        self.log_steps = int(self.training_config.get("log_steps", 100))

        # CL head setting: 'task_il' (multi-head) or 'class_il' (single head)
        self.cl_setting = (self.config.get("cl_setting") or "task_il").lower()
        if self.cl_setting not in {"task_il", "class_il"}:
            logger.warning(f"Unknown cl_setting '{self.cl_setting}', defaulting to 'task_il'")
            self.cl_setting = "task_il"

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        optimizer_name = self.training_config.get("optimizer", "adamw").lower()
        learning_rate = self.training_config["learning_rate"]
        weight_decay = self.training_config.get("weight_decay", 0.01)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if optimizer_name == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        logger.info(f"Setup {optimizer_name} optimizer with lr={learning_rate}")
        return optimizer

    def _setup_scheduler(self, num_training_steps: int, num_warmup_steps: int):
        scheduler_name = self.training_config.get("scheduler", "linear").lower()
        if scheduler_name == "none":
            return None
        if scheduler_name == "linear":
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        if scheduler_name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=max(1, num_training_steps - num_warmup_steps))
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        num_batches = 0
        all_predictions = []
        all_labels = []
        all_attention_masks = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # Do not pass labels to the model during evaluation to avoid
                # class-range mismatches when heads grow (class-IL).
                model_inputs = {k: v for k, v in batch.items() if k != "labels"}
                outputs = self.model(**model_inputs)
                all_predictions.append(outputs["logits"])
                all_labels.append(batch["labels"])
                all_attention_masks.append(batch["attention_mask"])
                num_batches += 1

        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)
        metrics = self.metrics.compute_metrics(all_predictions, all_labels, all_attention_masks)
        # Loss is omitted in evaluation due to potential class-range mismatches
        # when growing heads; keep metric keys consistent without eval_loss.
        return metrics

    def _activate_head(self, task_name: str, num_labels: int):
        """Ensure classifier matches the requested head for this task.

        If a saved head exists for task_name and matches num_labels, load it.
        Otherwise, reset classifier to num_labels.
        """
        # Load existing head if available and compatible
        state = self.head_states.get(task_name)
        if state is not None:
            # Check output dimension compatibility
            # Infer out_features from any weight tensor in state
            w = state.get('weight')
            if w is not None and w.size(0) == num_labels:
                self.model.reset_classifier(num_labels)
                self.model.classifier.load_state_dict(copy.deepcopy(state))
                return
        # Otherwise reset to requested size
        self.model.reset_classifier(num_labels)
        # After resetting classifier, refresh optimizer so new params are optimized
        self._refresh_optimizer_params()

    def _save_active_head_state(self, task_name: str):
        self.head_states[task_name] = copy.deepcopy(self.model.classifier.state_dict())

    def _refresh_optimizer_params(self):
        """Rebuild optimizer param groups to include current model params.

        This is necessary after swapping the classifier head (task-IL setting).
        """
        optimizer_name = self.training_config.get("optimizer", "adamw").lower()
        lr = self.training_config["learning_rate"]
        weight_decay = self.training_config.get("weight_decay", 0.01)

        # Preserve epsilon if using AdamW
        eps = 1e-8
        if hasattr(self.optimizer, "param_groups"):
            for g in self.optimizer.param_groups:
                if "eps" in g:
                    eps = g["eps"]
                    break

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if optimizer_name == "adamw":
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
        else:
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
            logger.warning(f"Optimizer '{optimizer_name}' not explicitly supported for refresh; using AdamW.")

    def train_task(self, train_loader: DataLoader, eval_loader: Optional[DataLoader], task_id: int,
                   save_tag: str = "task", *, label_list: Optional[List[str]] = None,
                   id2label: Optional[Dict[int, str]] = None) -> Dict[str, float]:
        # If provided, refresh metrics and classifier for this task
        if self.cl_setting == "task_il":
            if label_list is not None:
                # Build id2label if not provided
                id2label = id2label or {i: l for i, l in enumerate(label_list)}
                # Update metrics and activate (or create) the per-task head
                self.metrics = LayoutLMMetrics(label_list, id2label)
                self._activate_head(save_tag, num_labels=len(label_list))
        else:  # class_il
            if label_list is not None:
                id2label = id2label or {i: l for i, l in enumerate(label_list)}
                # Expand classifier if new labels have been introduced
                if len(label_list) > self.model.num_labels:
                    self.model.expand_classifier(len(label_list))
                    self._refresh_optimizer_params()
                # Always keep metrics in sync with the current global label list
                self.metrics = LayoutLMMetrics(label_list, id2label)
        # Determine effective gradient accumulation
        ga_steps_cfg = int(self.training_config.get("gradient_accumulation_steps", 1))
        if isinstance(self.strategy, (AGEM, GEM)) and ga_steps_cfg != 1:
            logger.warning("Overriding gradient_accumulation_steps to 1 for (A-)GEM to ensure correct projection.")
            ga_steps = 1
        else:
            ga_steps = ga_steps_cfg

        num_epochs = int(self.training_config.get("num_epochs", 5))

        # Scheduler per task
        steps_per_epoch = len(train_loader)
        total_steps = max(1, (steps_per_epoch * num_epochs) // max(1, ga_steps))
        warmup_ratio = float(self.training_config.get("warmup_ratio", 0.1))
        warmup_steps = int(warmup_ratio * total_steps)
        scheduler = self._setup_scheduler(total_steps, warmup_steps)

        best_metric = -1e9
        best_path = self.output_dir / f"{save_tag}_{task_id}_best_model"
        early_stop_counter = 0

        self.strategy.before_task(self.model, task_id, train_loader)

        for epoch in range(num_epochs):
            self.epoch = epoch
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            progress = tqdm(train_loader, desc=f"Task {task_id} Epoch {epoch + 1}/{num_epochs}")

            for step, batch in enumerate(progress):
                micro = step % ga_steps
                if micro == 0:
                    self.model.zero_grad(set_to_none=True)

                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = self.strategy.compute_loss(self.model, batch, outputs)

                # Scale for accumulation
                loss_scaled = loss / ga_steps

                # Let strategy project/compute grads if needed
                self.strategy.on_before_backward(self.model, loss_scaled)

                # If no grads were set by strategy, backprop now
                if not any(p.grad is not None for p in self.model.parameters()):
                    loss_scaled.backward()

                # Step if end of accumulation window
                if micro == ga_steps - 1:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                    # Update memory after optimization step
                    self.strategy.update_memory(batch)

                total_loss += loss.item()
                num_batches += 1
                progress.set_postfix({"loss": f"{(total_loss / num_batches):.4f}"})

            # Evaluate end of epoch
            eval_metrics = {}
            if eval_loader is not None:
                eval_metrics = self.evaluate(eval_loader)
                metric_key = self.training_config.get("metric_for_best_model", "eval_f1")
                current = float(eval_metrics.get(metric_key, -1e9))
                if current > best_metric:
                    best_metric = current
                    early_stop_counter = 0
                    if self.training_config.get("save_best_model", True):
                        self.model.save_pretrained(str(best_path))
                        # Cache the best head state for this task
                        self._save_active_head_state(save_tag)
                else:
                    early_stop_counter += 1

                if early_stop_counter >= int(self.training_config.get("early_stopping_patience", 10)):
                    logger.info(f"Early stopping on task {task_id} at epoch {epoch + 1}")
                    break

        # Optionally load best at end
        if self.training_config.get("load_best_model_at_end", True) and best_metric > -1e8 and best_path.exists():
            # Reuse logic to load backbone & head and move to device
            # We cannot import circularly; manually perform a lightweight load
            self.model.load_pretrained(str(best_path))
            self.model.to(self.device)
            # Ensure head state registry holds the final best for this task
            self._save_active_head_state(save_tag)

        self.strategy.after_task(self.model, task_id, train_loader)
        return {"best_metric": best_metric}

    def train(self, tasks: List[Dict[str, Any]]):
        """Train across tasks and collect CL evaluation metrics.

        Returns a dictionary including per-task training summaries, the
        accuracy matrix, and aggregated CL metrics (ACC, BWT, FWT, AAA,
        Forgetting).
        """
        per_task_results: List[Dict[str, float]] = []
        task_names = [t.get("name", f"task{i}") for i, t in enumerate(tasks)]
        T = len(tasks)

        # Pre-training evaluation on each task (R0)
        pre_accuracy: List[float] = []
        for j in range(T):
            if self.cl_setting == "task_il":
                lbl_list = tasks[j].get("label_list") or self.metrics.label_list
                id2label = {i: l for i, l in enumerate(lbl_list)}
                m = self.evaluate_with_head(tasks[j]["eval_loader"], task_names[j], lbl_list, id2label)
            else:
                # Class-IL: choose an id2label that covers BOTH current head and task j
                task_lbl_list = tasks[j].get("label_list") or self.metrics.label_list
                task_id2label = tasks[j].get("id2label") or {i: l for i, l in enumerate(task_lbl_list)}
                head_id2label = getattr(self.metrics, "id2label", task_id2label)
                # Prefer the larger mapping to avoid KeyError in entity F1
                use_id2label = task_id2label if len(task_id2label) >= len(head_id2label) else head_id2label
                use_lbl_list = [use_id2label[i] for i in range(len(use_id2label))]
                self.metrics = LayoutLMMetrics(use_lbl_list, use_id2label)
                m = self.evaluate(tasks[j]["eval_loader"])  # single head evaluation
            pre_accuracy.append(float(m.get("accuracy", 0.0)))

        # Accuracy matrix R[i][j] after training task i, evaluated on task j
        acc_matrix: List[List[float]] = [[0.0 for _ in range(T)] for _ in range(T)]

        for i, task in enumerate(tasks):
            train_loader: DataLoader = task["train_loader"]
            eval_loader: Optional[DataLoader] = task.get("eval_loader")

            res = self.train_task(
                train_loader,
                eval_loader,
                i,
                save_tag=task.get("name", "task"),
                label_list=task.get("label_list"),
                id2label=task.get("id2label"),
            )
            per_task_results.append(res)

            # Evaluate on all tasks with a mapping that covers predictions and labels
            for j in range(T):
                if self.cl_setting == "task_il":
                    lbl_list = tasks[j].get("label_list") or self.metrics.label_list
                    id2label = {i: l for i, l in enumerate(lbl_list)}
                    metrics = self.evaluate_with_head(tasks[j]["eval_loader"], task_names[j], lbl_list, id2label)
                else:
                    # Class-IL: choose an id2label that covers BOTH current head and task j
                    task_lbl_list = tasks[j].get("label_list") or self.metrics.label_list
                    task_id2label = tasks[j].get("id2label") or {i: l for i, l in enumerate(task_lbl_list)}
                    head_id2label = getattr(self.metrics, "id2label", task_id2label)
                    # Prefer the larger mapping to avoid KeyError in entity F1
                    use_id2label = task_id2label if len(task_id2label) >= len(head_id2label) else head_id2label
                    use_lbl_list = [use_id2label[i] for i in range(len(use_id2label))]
                    self.metrics = LayoutLMMetrics(use_lbl_list, use_id2label)
                    metrics = self.evaluate(tasks[j]["eval_loader"])  # single head evaluation
                acc_matrix[i][j] = float(metrics.get("accuracy", 0.0))

            # Save checkpoint after finishing task i (for later offline evaluation)
            try:
                self._save_checkpoint_after_task(i, task_names, tasks, acc_matrix[i])
            except Exception as e:
                logger.warning(f"Failed to save checkpoint after task {i}: {e}")

        # Compute CL metrics
        cl_metrics = compute_cl_metrics(acc_matrix, pre_accuracy, task_names)

        return {
            "per_task": per_task_results,
            "task_names": task_names,
            "pre_accuracy": pre_accuracy,
            "accuracy_matrix": acc_matrix,
            "cl_metrics": cl_metrics,
        }

    def _save_checkpoint_after_task(
        self,
        task_idx: int,
        task_names: List[str],
        tasks: List[Dict[str, Any]],
        acc_row: List[float],
    ) -> None:
        """Persist model snapshot and metadata after finishing a task.

        Creates a directory `checkpoints/after_task_{i}_{name}` with:
          - HF backbone weights and current classifier (classifier.pt)
          - class-IL: label_list.json describing the global label space
          - task-IL: all per-task heads so far under `heads/*.pt` and heads_meta.json
          - acc_row.json: accuracies on all tasks evaluated after finishing this task
        """
        ckpt_root = self.output_dir / "checkpoints"
        ckpt_root.mkdir(parents=True, exist_ok=True)
        ckpt_dir = ckpt_root / f"after_task_{task_idx}_{task_names[task_idx]}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save backbone + current active classifier
        self.model.save_pretrained(str(ckpt_dir))

        # Save accuracy row for this checkpoint
        acc_payload = {
            "after_task_index": task_idx,
            "after_task_name": task_names[task_idx],
            "task_names": list(task_names),
            "accuracy_row": list(map(float, acc_row)),
            "cl_setting": self.cl_setting,
        }
        with open(ckpt_dir / "acc_row.json", "w") as f:
            json.dump(acc_payload, f, indent=2)

        if self.cl_setting == "class_il":
            # Persist the current global label space for this checkpoint
            label_list = self.metrics.label_list
            id2label = self.metrics.id2label
            with open(ckpt_dir / "label_list.json", "w") as f:
                json.dump({
                    "label_list": list(label_list),
                    "id2label": {int(k): v for k, v in id2label.items()},
                }, f, indent=2)
        else:  # task_il
            # Save all heads we have so far under heads/
            heads_dir = ckpt_dir / "heads"
            heads_dir.mkdir(parents=True, exist_ok=True)
            # Map task name -> label_list for metadata
            labels_by_task = {t["name"]: t.get("label_list") for t in tasks}
            heads_meta = {}
            for head_name, state in self.head_states.items():
                torch.save(state, heads_dir / f"{head_name}.pt")
                heads_meta[head_name] = {
                    "num_labels": int(state.get("weight").shape[0]) if state.get("weight") is not None else None,
                    "label_list": labels_by_task.get(head_name),
                }
            with open(ckpt_dir / "heads_meta.json", "w") as f:
                json.dump(heads_meta, f, indent=2)

    def evaluate_with_head(self, dataloader: DataLoader, task_name: str, label_list: List[str], id2label: Dict[int, str]
                           ) -> Dict[str, float]:
        # Activate task-specific head and update metrics
        self._activate_head(task_name, num_labels=len(label_list))
        self.metrics = LayoutLMMetrics(label_list, id2label)
        return self.evaluate(dataloader)


def create_continual_trainer(
    model: BaseLayoutLMModel,
    config: Dict[str, Any],
    label_list: List[str],
    id2label: Dict[int, str],
    strategy: Optional[BaseCLStrategy] = None,
) -> ContinualLayoutLMTrainer:
    return ContinualLayoutLMTrainer(model=model, config=config, label_list=label_list, id2label=id2label,
                                    strategy=strategy)
