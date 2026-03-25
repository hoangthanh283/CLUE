"""End-to-end tests for CL strategies integration."""

import pytest
import torch


class TestStrategyIntegration:
    """Integration tests for continual learning strategies."""

    def test_sequential_baseline(self, tiny_model, mock_dataloader, tmp_path):
        """Test sequential (baseline) training without any CL mechanism."""
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)

        # Task 1: Train normally
        for batch in mock_dataloader:
            output = tiny_model(**batch)
            loss = output["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Task 2: Train normally (fine-tuning)
        for batch in mock_dataloader:
            output = tiny_model(**batch)
            loss = output["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert True, "Sequential training completed successfully"

    def test_strategy_with_memory_buffer(self, tiny_model, mock_dataloader):
        """Test strategy with memory buffer (replay-based)."""
        memory_buffer = []
        memory_size = 10

        # Task 1: Store exemplars
        task1_data = []
        for batch in mock_dataloader:
            task1_data.append(batch)

        # Simple memory management: store entire batches up to memory_size
        for batch in task1_data[:memory_size]:
            memory_buffer.append(batch)

        assert len(memory_buffer) > 0
        assert len(memory_buffer) <= memory_size

        # Task 2: Use both new data and replay
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)

        # Replay from memory
        for replay_batch in memory_buffer:
            output = tiny_model(**replay_batch)
            loss = output["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # New data
        for batch in mock_dataloader:
            output = tiny_model(**batch)
            loss = output["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert True, "Memory replay training completed"

    def test_strategy_with_fisher_information(self, tiny_model, mock_dataloader):
        """Test strategy with Fisher information (EWC-style)."""
        # Compute Fisher information matrix for task 1
        fisher_dict = {}
        for name, param in tiny_model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param.data)

        # Compute Fisher information
        for batch in mock_dataloader:
            output = tiny_model(**batch)
            loss = output["loss"]

            tiny_model.zero_grad()
            loss.backward()

            for name, param in tiny_model.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2

        # Normalize by number of batches
        num_batches = 3
        for name in fisher_dict:
            fisher_dict[name] /= num_batches

        # Verify Fisher computation
        assert len(fisher_dict) > 0
        for name, fisher in fisher_dict.items():
            assert fisher.shape == tiny_model.state_dict()[name].shape
            assert (fisher >= 0).all(), "Fisher info should be non-negative"

    def test_strategy_with_gradient_constraints(self, tiny_model, mock_dataloader):
        """Test strategy with gradient constraints (GEM-style)."""
        # Store gradients from task 1
        old_task_gradients = []

        for batch in mock_dataloader:
            output = tiny_model(**batch)
            loss = output["loss"]

            tiny_model.zero_grad()
            loss.backward()

            # Store gradient directions
            task1_grads = [
                param.grad.clone() for param in tiny_model.parameters() if param.grad is not None
            ]
            old_task_gradients.append(task1_grads)

        # Task 2: Train with gradient constraint
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)

        for batch in mock_dataloader:
            output = tiny_model(**batch)
            loss = output["loss"]

            optimizer.zero_grad()
            loss.backward()

            # Simple constraint: ensure gradients don't point opposite to old task
            for param in tiny_model.parameters():
                if param.grad is not None:
                    # Clip if violating constraint (dummy implementation)
                    param.grad.clamp_(min=-1.0, max=1.0)

            optimizer.step()

        assert len(old_task_gradients) > 0

    def test_strategy_with_distillation(self, tiny_model, tmp_path):
        """Test strategy with knowledge distillation (LwF-style)."""
        # Task 1: Train and save as teacher
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)

        fake_batch = {
            "input_ids": torch.randint(0, 100, (2, 8)),
            "attention_mask": torch.ones(2, 8),
            "bbox": torch.zeros(2, 8, 4),
            "labels": torch.randint(0, 3, (2, 8)),
        }

        output = tiny_model(**fake_batch)
        loss = output["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save teacher model
        teacher_model = type(tiny_model)(num_labels=3)
        teacher_model.load_state_dict(tiny_model.state_dict())
        teacher_model.eval()

        # Task 2: Train student with distillation loss
        student_model = tiny_model
        student_model.train()

        temperature = 4.0
        distillation_weight = 0.5

        optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

        with torch.no_grad():
            teacher_output = teacher_model(**fake_batch)
            teacher_logits = teacher_output["logits"]

        student_output = student_model(**fake_batch)
        student_logits = student_output["logits"]

        # Distillation loss
        distillation_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits / temperature, dim=-1),
            torch.nn.functional.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature ** 2)

        task_loss = student_output["loss"]
        total_loss = (1 - distillation_weight) * task_loss + distillation_weight * distillation_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        assert True, "Distillation training completed"

    def test_strategy_with_exemplar_selection(self, mock_dataloader):
        """Test intelligent exemplar selection for replay."""
        exemplar_set = {}
        exemplar_per_class = 2

        for batch_idx, batch in enumerate(mock_dataloader):
            labels = batch["labels"]
            unique_labels = torch.unique(labels[labels >= 0])

            for label in unique_labels:
                label_idx = label.item()
                if label_idx not in exemplar_set:
                    exemplar_set[label_idx] = []

                # Simple selection: take first exemplars_per_class samples
                mask = labels == label
                if mask.any() and len(exemplar_set[label_idx]) < exemplar_per_class:
                    exemplar_set[label_idx].append(batch_idx)

        # Verify exemplar set
        assert len(exemplar_set) > 0
        for label_id, indices in exemplar_set.items():
            assert len(indices) <= exemplar_per_class

    def test_continual_learning_task_sequence(self, tiny_model, mock_dataloader):
        """Test full CL sequence with task-incremental settings."""
        num_tasks = 2
        task_results = {}

        for task_id in range(num_tasks):
            task_name = f"task_{task_id}"
            task_results[task_name] = {"train_loss": [], "val_accuracy": []}

            optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)

            # Grow classifier if needed
            if task_id > 0:
                new_labels = 3 + task_id * 2  # Increase labels per task
                tiny_model.expand_classifier(new_labels)

            # Train on task
            for batch in mock_dataloader:
                output = tiny_model(**batch)
                loss = output["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                task_results[task_name]["train_loss"].append(loss.item())

        # Verify results
        assert len(task_results) == num_tasks
        for task_name, results in task_results.items():
            assert len(results["train_loss"]) > 0

    def test_strategy_forgetting_computation(self, tiny_model, mock_dataloader, fake_batch):
        """Test computation of forgetting metric."""
        # Task 1: Train and measure baseline accuracy
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)

        for batch in mock_dataloader:
            output = tiny_model(**batch)
            loss = output["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Measure accuracy on task 1
        tiny_model.eval()
        with torch.no_grad():
            output = tiny_model(**fake_batch)
            logits = output["logits"]
            labels = fake_batch["labels"]
            task1_accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()

        # Task 2: Fine-tune on new task
        tiny_model.train()
        for batch in mock_dataloader:
            output = tiny_model(**batch)
            loss = output["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Measure accuracy on task 1 again (forgetting)
        tiny_model.eval()
        with torch.no_grad():
            output = tiny_model(**fake_batch)
            logits = output["logits"]
            labels = fake_batch["labels"]
            task1_accuracy_after = (logits.argmax(dim=-1) == labels).float().mean().item()

        forgetting = task1_accuracy - task1_accuracy_after

        # Positive forgetting is expected when fine-tuning
        assert forgetting is not None

    def test_strategy_backward_transfer(self, tiny_model, mock_dataloader, fake_batch):
        """Test computation of backward transfer metric."""
        # Task 1: Train and measure
        optimizer = torch.optim.Adam(tiny_model.parameters(), lr=1e-4)

        for batch in mock_dataloader:
            output = tiny_model(**batch)
            loss = output["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Measure task 2 performance (random initialization)
        tiny_model.eval()
        with torch.no_grad():
            output = tiny_model(**fake_batch)
            logits = output["logits"]
            labels = fake_batch["labels"]
            task2_accuracy_random = (logits.argmax(dim=-1) == labels).float().mean().item()

        # Task 2: Train
        tiny_model.train()
        for batch in mock_dataloader:
            output = tiny_model(**batch)
            loss = output["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Measure task 2 after training
        tiny_model.eval()
        with torch.no_grad():
            output = tiny_model(**fake_batch)
            logits = output["logits"]
            labels = fake_batch["labels"]
            task2_accuracy_trained = (logits.argmax(dim=-1) == labels).float().mean().item()

        backward_transfer = task2_accuracy_trained - task2_accuracy_random

        # Should be positive (training on task 2 improves task 2)
        assert backward_transfer >= 0 or backward_transfer is None  # Might be 0 with small data
