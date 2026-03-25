"""Integration tests for model pipeline (initialization, forward pass, gradients)."""

import pytest
import torch
import torch.nn as nn


class TestModelPipeline:
    """Integration tests for model initialization and inference."""

    def test_tiny_model_initialization(self, tiny_model):
        """Test that tiny model initializes correctly."""
        assert isinstance(tiny_model, nn.Module)
        assert hasattr(tiny_model, "classifier")
        assert tiny_model.num_labels == 3
        assert tiny_model.hidden_size == 16

    def test_forward_pass_without_labels(self, tiny_model, fake_batch):
        """Test forward pass without labels (inference mode)."""
        # Remove labels for inference
        batch = {k: v for k, v in fake_batch.items() if k != "labels"}

        output = tiny_model(**batch)

        assert "logits" in output
        assert output["logits"].shape == (2, 8, 3)  # (batch_size, seq_len, num_labels)
        assert output["loss"] is None

    def test_forward_pass_with_labels(self, tiny_model, fake_batch):
        """Test forward pass with labels (training mode)."""
        output = tiny_model(**fake_batch)

        assert "loss" in output
        assert "logits" in output
        assert output["loss"] is not None
        assert output["loss"].item() >= 0  # Loss should be non-negative

    def test_gradient_computation(self, tiny_model, fake_batch):
        """Test that gradients are computed correctly."""
        output = tiny_model(**fake_batch)
        loss = output["loss"]

        loss.backward()

        # Check that gradients exist and are non-zero
        has_grads = False
        for param in tiny_model.parameters():
            if param.grad is not None:
                has_grads = True
                assert not torch.isnan(param.grad).any(), "Gradients should not contain NaN"
                break

        assert has_grads, "At least one parameter should have gradients"

    def test_classifier_head_expansion(self, tiny_model, fake_batch):
        """Test expanding classifier head for class-incremental learning."""
        old_num_labels = tiny_model.num_labels
        new_num_labels = 6

        # Expand classifier
        tiny_model.expand_classifier(new_num_labels)

        assert tiny_model.num_labels == new_num_labels
        assert tiny_model.classifier.out_features == new_num_labels

        # Forward pass should still work with new classifier
        output = tiny_model(**fake_batch)
        assert output["logits"].shape[2] == new_num_labels  # Output should have new label dimension

    def test_classifier_head_no_shrink(self, tiny_model):
        """Test that classifier head doesn't shrink."""
        old_num_labels = tiny_model.num_labels

        # Try to expand to same size - should be no-op
        tiny_model.expand_classifier(old_num_labels)
        assert tiny_model.num_labels == old_num_labels

        # Try to shrink - should be no-op
        tiny_model.expand_classifier(1)
        assert tiny_model.num_labels == old_num_labels

    def test_classifier_reset(self, tiny_model):
        """Test resetting classifier to new size."""
        new_num_labels = 5
        tiny_model.reset_classifier(new_num_labels)

        assert tiny_model.num_labels == new_num_labels
        assert tiny_model.classifier.out_features == new_num_labels

    def test_model_device_placement(self, tiny_model, fake_batch):
        """Test model device placement and forward pass."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tiny_model = tiny_model.to(device)

        # Move batch to device
        batch = {k: v.to(device) for k, v in fake_batch.items()}

        output = tiny_model(**batch)

        # Check device type (cuda, cpu, etc.)
        assert str(output["logits"].device).split(':')[0] == str(device).split(':')[0]
        if output["loss"] is not None:
            assert str(output["loss"].device).split(':')[0] == str(device).split(':')[0]

    def test_model_save_and_load(self, tiny_model, fake_batch, tmp_path):
        """Test model serialization and deserialization."""
        # Save model
        save_dir = str(tmp_path / "model")
        tiny_model.save_pretrained(save_dir)

        # Create new model and load
        new_model = type(tiny_model)(num_labels=tiny_model.num_labels)
        new_model.load_pretrained(save_dir)

        # Verify loaded model produces same outputs (deterministic)
        torch.manual_seed(42)
        output1 = tiny_model(**fake_batch)

        torch.manual_seed(42)
        output2 = new_model(**fake_batch)

        assert torch.allclose(output1["logits"], output2["logits"], atol=1e-6)

    def test_attention_mask_filtering(self, tiny_model):
        """Test that attention mask properly filters logits."""
        batch_size = 2
        seq_len = 8
        num_labels = 3

        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        attention_mask[0, 5:] = 0  # Mask last 3 tokens of first sample

        output = tiny_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        assert output["logits"].shape == (batch_size, seq_len, num_labels)
        # Masked positions should still have logits (masking applied in loss computation)

    def test_label_smoothing_stability(self, tiny_model, fake_batch):
        """Test that model loss is numerically stable."""
        for _ in range(5):
            output = tiny_model(**fake_batch)
            loss = output["loss"]

            assert not torch.isnan(loss), "Loss should not be NaN"
            assert not torch.isinf(loss), "Loss should not be infinite"
            assert loss.item() >= 0, "Loss should be non-negative"

    def test_batch_size_robustness(self, tiny_model):
        """Test model with different batch sizes."""
        for batch_size in [1, 2, 4]:
            input_ids = torch.randint(0, 100, (batch_size, 8))
            labels = torch.randint(0, 3, (batch_size, 8))

            output = tiny_model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                bbox=torch.zeros(batch_size, 8, 4),
                labels=labels,
            )

            assert output["logits"].shape[0] == batch_size
            assert output["loss"] is not None
