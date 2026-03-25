"""Shared fixtures for cl4ie tests."""

import types
from pathlib import Path
from typing import Dict, List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Tiny model – mimics BaseLayoutLMModel interface without heavy deps
# ---------------------------------------------------------------------------

HIDDEN = 16
SEQ_LEN = 8
BATCH = 2
NUM_LABELS = 3
LABEL_LIST = ["O", "B-X", "I-X"]
ID2LABEL = {0: "O", 1: "B-X", 2: "I-X"}


class TinyModel(nn.Module):
    """Lightweight model that satisfies the BaseLayoutLMModel duck-type contract."""

    def __init__(self, num_labels: int = NUM_LABELS):
        super().__init__()
        self.num_labels = num_labels
        self.hidden_size = HIDDEN
        self.embed = nn.Embedding(200, HIDDEN)
        self.classifier = nn.Linear(HIDDEN, num_labels)
        # Fake backbone for code that reads model.backbone.config.hidden_size
        self.backbone = types.SimpleNamespace(
            config=types.SimpleNamespace(hidden_size=HIDDEN, model_type="fake")
        )

    def forward(
        self,
        input_ids: torch.Tensor = None,
        bbox: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        h = self.embed(input_ids)           # (B, S, HIDDEN)
        logits = self.classifier(h)         # (B, S, num_labels)

        loss = None
        if labels is not None:
            mask = labels != -100
            if mask.any():
                loss = F.cross_entropy(
                    logits.view(-1, self.num_labels)[mask.view(-1)],
                    labels.view(-1)[mask.view(-1)],
                )
            else:
                loss = logits.mean() * 0.0
        return {"loss": loss, "logits": logits}

    def save_pretrained(self, save_directory: str) -> None:
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), f"{save_directory}/classifier.pt")

    def load_pretrained(self, load_directory: str) -> None:
        state = torch.load(
            f"{load_directory}/classifier.pt",
            map_location="cpu",
            weights_only=True,
        )
        self.load_state_dict(state, strict=False)

    def expand_classifier(self, new_num_labels: int) -> None:
        if new_num_labels <= self.num_labels:
            return
        device = next(self.parameters()).device
        old = self.classifier
        new_clf = nn.Linear(HIDDEN, new_num_labels).to(device)
        nn.init.normal_(new_clf.weight, 0.0, 0.02)
        nn.init.zeros_(new_clf.bias)
        with torch.no_grad():
            new_clf.weight[: self.num_labels].copy_(old.weight)
            new_clf.bias[: self.num_labels].copy_(old.bias)
        self.classifier = new_clf
        self.num_labels = new_num_labels

    def reset_classifier(self, num_labels: int) -> None:
        device = next(self.parameters()).device
        self.classifier = nn.Linear(HIDDEN, num_labels).to(device)
        self.num_labels = num_labels


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_model() -> TinyModel:
    return TinyModel(num_labels=NUM_LABELS)


@pytest.fixture
def fake_batch() -> Dict[str, torch.Tensor]:
    """Minimal batch dict matching what trainers and strategies expect."""
    return {
        "input_ids": torch.randint(0, 100, (BATCH, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.long),
        "bbox": torch.zeros(BATCH, SEQ_LEN, 4, dtype=torch.long),
        "labels": torch.randint(0, NUM_LABELS, (BATCH, SEQ_LEN)),
    }


@pytest.fixture
def fake_batch_with_ignored(fake_batch) -> Dict[str, torch.Tensor]:
    """Batch with some -100 ignore tokens."""
    b = {k: v.clone() for k, v in fake_batch.items()}
    b["labels"][:, 0] = -100  # first token of each sample ignored
    return b


@pytest.fixture
def mock_dataloader(fake_batch) -> DataLoader:
    """DataLoader yielding 3 identical mini-batches."""
    ds = TensorDataset(
        fake_batch["input_ids"],
        fake_batch["attention_mask"],
        fake_batch["bbox"],
        fake_batch["labels"],
    )

    class _DictDataset:
        def __init__(self, n_repeat=3):
            self.n = n_repeat

        def __len__(self):
            return self.n

        def __iter__(self):
            for _ in range(self.n):
                yield {k: v.clone() for k, v in fake_batch.items()}

    return _DictDataset()


@pytest.fixture
def base_config_dict(tmp_path) -> dict:
    """Minimal valid ExperimentConfig dict with temp output_dir."""
    return {
        "experiment_name": "unit_test",
        "output_dir": str(tmp_path / "output"),
        "cl_setting": "class_il",
        "model": {
            "name": "tiny",
            "model_type": "layoutlmv3",
            "pretrained_model_name": "microsoft/layoutlmv3-base",
            "config": {"num_labels": NUM_LABELS},
        },
        "training": {
            "batch_size": BATCH,
            "learning_rate": 1e-4,
            "num_epochs": 1,
            "scheduler": "none",
            "early_stopping_patience": 100,
            "metric_for_best_model": "eval_f1",
            "save_best_model": False,
            "load_best_model_at_end": False,
            "log_steps": 9999,
            "num_workers": 0,
        },
        "cl_strategy": {"name": "sequential"},
        "wandb": {"use_wandb": False},
    }
