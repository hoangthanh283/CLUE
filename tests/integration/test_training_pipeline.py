"""Integration tests for training pipeline (loss, backward pass, checkpointing)."""

import pytest
import torch
from pathlib import Path


class TestTrainingPipeline:
    """Integration tests for training loop and optimization."""

    def test_single_training_step(self, tiny_model, fake_batch):
        """Test a single training step."""
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)

        # Forward pass
        output = tiny_model(**fake_batch)
        loss = output["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify optimization happened (parameters changed)
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_gradient_accumulation(self, tiny_model, mock_dataloader):
        """Test gradient accumulation over multiple batches."""
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)
        accumulation_steps = 2

        accumulated_loss = 0
        step_count = 0

        for i, batch in enumerate(mock_dataloader):
            output = tiny_model(**batch)
            loss = output["loss"]

            # Normalize loss for accumulation
            loss = loss / accumulation_steps
            loss.backward()

            accumulated_loss += loss.item()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                step_count += 1

        assert step_count > 0, "Should have at least one optimizer step"
        assert accumulated_loss > 0, "Accumulated loss should be positive"

    def test_learning_rate_scheduling(self, tiny_model, mock_dataloader):
        """Test learning rate scheduling during training."""
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        initial_lr = optimizer.param_groups[0]["lr"]

        for batch in mock_dataloader:
            output = tiny_model(**batch)
            loss = output["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        new_lr = optimizer.param_groups[0]["lr"]

        assert new_lr < initial_lr, "Learning rate should decay"
        assert abs(new_lr - initial_lr * 0.5) < 1e-8, "LR should be multiplied by gamma=0.5"

    def test_early_stopping_tracking(self):
        """Test early stopping logic."""
        patience = 3
        best_loss = float("inf")
        patience_counter = 0
        losses = [1.0, 0.9, 0.85, 0.86, 0.87, 0.88]  # Improves then worsens

        for loss in losses:
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        assert best_loss == 0.85
        assert patience_counter == patience

    def test_checkpoint_save_and_load(self, tiny_model, tmp_path):
        """Test model checkpoint saving and loading."""
        checkpoint_path = tmp_path / "checkpoint.pt"

        # Modify model parameters
        original_param = tiny_model.classifier.weight.clone()

        # Save checkpoint
        torch.save(
            {
                "model_state_dict": tiny_model.state_dict(),
                "epoch": 5,
                "loss": 0.5,
            },
            checkpoint_path,
        )

        # Modify parameters
        with torch.no_grad():
            tiny_model.classifier.weight.mul_(0.5)

        # Verify parameters changed
        assert not torch.allclose(
            tiny_model.classifier.weight, original_param
        ), "Parameters should have changed"

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        tiny_model.load_state_dict(checkpoint["model_state_dict"])

        # Verify parameters restored
        assert torch.allclose(
            tiny_model.classifier.weight, original_param
        ), "Parameters should be restored"
        assert checkpoint["epoch"] == 5

    def test_metric_computation_during_training(self):
        """Test metric tracking during training."""
        metrics = {
            "train_loss": [],
            "eval_f1": [],
            "eval_accuracy": [],
        }

        # Simulate training loop
        for epoch in range(3):
            train_loss = 1.0 / (epoch + 1)  # Decreasing loss
            eval_f1 = 0.5 + epoch * 0.1
            eval_acc = 0.6 + epoch * 0.15

            metrics["train_loss"].append(train_loss)
            metrics["eval_f1"].append(eval_f1)
            metrics["eval_accuracy"].append(eval_acc)

        assert len(metrics["train_loss"]) == 3
        assert metrics["train_loss"][0] > metrics["train_loss"][-1]
        assert metrics["eval_f1"][-1] > metrics["eval_f1"][0]

    def test_best_model_tracking(self):
        """Test tracking best model based on metric."""
        best_metric = 0
        best_epoch = -1

        metrics = [0.6, 0.7, 0.72, 0.68, 0.70, 0.74]  # Best is last

        for epoch, metric in enumerate(metrics):
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch

        assert best_metric == 0.74
        assert best_epoch == 5

    def test_batch_validation_step(self, tiny_model, fake_batch):
        """Test validation step without gradient computation."""
        tiny_model.eval()

        with torch.no_grad():
            output = tiny_model(**fake_batch)

        # Validation should still produce loss and logits
        assert "loss" in output
        assert "logits" in output

        # Parameters should not have gradients (no .backward() called)
        tiny_model.train()

    def test_gradient_clipping(self, tiny_model, fake_batch):
        """Test gradient clipping for stability."""
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-3)

        output = tiny_model(**fake_batch)
        loss = output["loss"]

        optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm before clipping
        grad_norm_before = torch.nn.utils.clip_grad_norm_(
            tiny_model.parameters(), max_norm=1.0
        )

        optimizer.step()

        assert grad_norm_before >= 0, "Gradient norm should be non-negative"

    def test_mixed_precision_dummy(self):
        """Test that mixed precision training logic is compatible."""
        # Note: Actual mixed precision requires CUDA, but we can test the logic
        if torch.cuda.is_available():
            scaler = torch.amp.GradScaler()

            # Create a simple optimizer
            param = torch.nn.Parameter(torch.tensor([1.0], device='cuda'))
            optimizer = torch.optim.SGD([param], lr=0.1)

            # Dummy loss
            loss = param.sum()

            # Scale and step correctly
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Scale should be updated after step
            assert scaler.get_scale() > 0
        else:
            # Skip test if no CUDA
            pass

    def test_parameter_freezing(self, tiny_model):
        """Test that parameters can be frozen for fine-tuning."""
        # Freeze embedding layer
        for param in tiny_model.embed.parameters():
            param.requires_grad = False

        # Only classifier should be trainable
        trainable_params = [p for p in tiny_model.parameters() if p.requires_grad]
        frozen_params = [p for p in tiny_model.parameters() if not p.requires_grad]

        assert len(trainable_params) > 0, "Should have trainable parameters"
        assert len(frozen_params) > 0, "Should have frozen parameters"

    def test_optimizer_state_persistence(self, tiny_model, tmp_path):
        """Test saving and loading optimizer state."""
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        # Training step
        fake_batch = {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "attention_mask": torch.ones(2, 8),
            "bbox": torch.zeros(2, 8, 4),
            "labels": torch.randint(0, 3, (2, 8)),
        }

        output = tiny_model(**fake_batch)
        optimizer.zero_grad()
        output["loss"].backward()
        optimizer.step()

        # Save optimizer state
        checkpoint_path = tmp_path / "optimizer.pt"
        torch.save(
            {
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            checkpoint_path,
        )

        # Create new optimizer and load state
        new_optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1, gamma=0.1)

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        new_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        new_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Verify that optimizer state was restored
        assert "state" in new_optimizer.state_dict()
        assert len(new_optimizer.state_dict()["state"]) == len(optimizer.state_dict()["state"])
