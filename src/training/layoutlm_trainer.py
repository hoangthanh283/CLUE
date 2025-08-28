"""
Training procedures for LayoutLM models
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import neptune
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from src.models.layoutlm_models import BaseLayoutLMModel, LayoutLMMetrics

logger = logging.getLogger(__name__)


class LayoutLMTrainer:
    """Trainer for LayoutLM models"""

    def __init__(
        self,
        model: BaseLayoutLMModel,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        config: Dict[str, Any],
        label_list: List[str],
        id2label: Dict[int, str]
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.training_config = config["training"]

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup metrics
        self.metrics = LayoutLMMetrics(label_list, id2label)

        # Gradient accumulation.
        self.gradient_accumulation_steps = self.training_config.get("gradient_accumulation_steps", 1)

        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = 0.0
        self.best_model_path = None

        # Setup output directory
        self.output_dir = Path(config.get("output_dir", "results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Early stopping
        self.early_stopping_patience = self.training_config.get("early_stopping_patience", 10)
        self.early_stopping_counter = 0

        # Logging
        self.log_steps = self.training_config.get("log_steps", 100)
        self.eval_steps = self.training_config.get("eval_steps", None)
        if self.eval_steps is not None:
            self.eval_steps = int(self.eval_steps)

        # Setup Neptune if configured
        self.neptune_run = None
        neptune_config = config.get("neptune", {})
        if neptune_config.get("use_neptune", False):
            self.neptune_run = neptune.init_run(
                project=neptune_config.get("neptune_project", "cl4ie"),
                name=config["experiment_name"],
                tags=neptune_config.get("tags", []),
                api_token=neptune_config.get("neptune_api_token")
            )
            self.neptune_run["config"] = config

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer"""
        optimizer_name = self.training_config.get("optimizer", "adamw").lower()
        learning_rate = self.training_config["learning_rate"]
        weight_decay = self.training_config.get("weight_decay", 0.01)

        # No decay for bias and LayerNorm.
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
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                eps=1e-8
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        logger.info(f"Setup {optimizer_name} optimizer with lr={learning_rate}")
        return optimizer

    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        scheduler_name = self.training_config.get("scheduler", "linear").lower()

        if scheduler_name == "none":
            return None

        num_epochs = self.training_config["num_epochs"]
        num_training_steps = len(self.train_dataloader) * num_epochs // self.gradient_accumulation_steps
        warmup_ratio = self.training_config.get("warmup_ratio", 0.1)
        num_warmup_steps = int(warmup_ratio * num_training_steps)

        if scheduler_name == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_name == "cosine":
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - num_warmup_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

        logger.info(f"Setup {scheduler_name} scheduler with {num_warmup_steps} warmup steps")
        return scheduler

    def train(self) -> Dict[str, float]:
        """Main training loop"""
        logger.info("Starting training...")

        num_epochs = self.training_config["num_epochs"]

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Training
            train_metrics = self._train_epoch()
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train: {train_metrics}")

            # Evaluation
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                logger.info(f"Epoch {epoch + 1}/{num_epochs} - Eval: {eval_metrics}")

                # Check for improvement
                current_metric = eval_metrics[self.training_config.get("metric_for_best_model", "eval_f1")]

                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.early_stopping_counter = 0

                    # Save best model
                    if self.training_config.get("save_best_model", True):
                        self.best_model_path = self.output_dir / "best_model"
                        self.save_model(self.best_model_path)
                        logger.info(f"New best model saved: {current_metric:.4f}")
                else:
                    self.early_stopping_counter += 1

                # Early stopping
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        logger.info("Training completed!")
        if self.best_model_path and self.training_config.get("load_best_model_at_end", True):
            logger.info(f"Loading best model from {self.best_model_path}")
            self.load_model(self.best_model_path)

        # Close Neptune run
        if self.neptune_run is not None:
            self.neptune_run.stop()

        return {"best_metric": self.best_metric}

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch + 1}")
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            outputs = self.model(**batch)
            loss = outputs["loss"]

            # Scale loss for gradient accumulation
            if self.gradient_accumulation_steps > 1:
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            # Logging
            if self.global_step % self.log_steps == 0:
                self._log_metrics({
                    "train_loss": avg_loss,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "epoch": self.epoch,
                    "global_step": self.global_step
                })

        return {"train_loss": total_loss / num_batches}

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model"""
        if self.eval_dataloader is None:
            return {}

        logger.info("Running evaluation...")
        self.model.eval()

        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_attention_masks = []

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                outputs = self.model(**batch)

                # Accumulate results
                if outputs["loss"] is not None:
                    total_loss += outputs["loss"].item()

                all_predictions.append(outputs["logits"])
                all_labels.append(batch["labels"])
                all_attention_masks.append(batch["attention_mask"])

        # Concatenate all results
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)

        # Compute metrics
        metrics = self.metrics.compute_metrics(all_predictions, all_labels, all_attention_masks)

        # Add loss
        if total_loss > 0:
            metrics["eval_loss"] = total_loss / len(self.eval_dataloader)

        # Add eval_ prefix
        eval_metrics = {f"eval_{k}": v for k, v in metrics.items()}

        self._log_metrics(eval_metrics)

        return eval_metrics

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to Neptune if available"""
        if self.neptune_run is not None:
            for key, value in metrics.items():
                self.neptune_run[f"metrics/{key}"].log(value, step=self.global_step)

    def save_model(self, save_path: Path):
        """Save model and tokenizer"""
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(str(save_path))

        # Save training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_metric": self.best_metric,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if self.scheduler is not None:
            training_state["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(training_state, save_path / "training_state.pt")

        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: Path):
        """Load model and training state"""
        # Load model
        self.model.load_pretrained(str(load_path))

        # Load training state if available
        training_state_path = load_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location=self.device)

            self.global_step = training_state["global_step"]
            self.epoch = training_state["epoch"]
            self.best_metric = training_state["best_metric"]
            self.optimizer.load_state_dict(training_state["optimizer_state_dict"])

            if self.scheduler is not None and "scheduler_state_dict" in training_state:
                self.scheduler.load_state_dict(training_state["scheduler_state_dict"])

        logger.info(f"Model loaded from {load_path}")


def create_trainer(
    model: BaseLayoutLMModel,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader],
    config: Dict[str, Any],
    label_list: List[str],
    id2label: Dict[int, str]
) -> LayoutLMTrainer:
    """Factory function to create trainer"""
    return LayoutLMTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config,
        label_list=label_list,
        id2label=id2label
    )
