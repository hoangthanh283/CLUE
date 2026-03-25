"""Tests for LayoutLMTrainer (single-task training)."""

from unittest.mock import MagicMock, patch

import torch
import pytest

from src.config import ExperimentConfig
from src.training.layoutlm_trainer import LayoutLMTrainer
from src.models.layoutlm_models import LayoutLMMetrics
from tests.conftest import TinyModel, BATCH, SEQ_LEN, NUM_LABELS, LABEL_LIST, ID2LABEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch():
    return {
        "input_ids": torch.randint(0, 100, (BATCH, SEQ_LEN)),
        "attention_mask": torch.ones(BATCH, SEQ_LEN, dtype=torch.long),
        "bbox": torch.zeros(BATCH, SEQ_LEN, 4, dtype=torch.long),
        "labels": torch.randint(0, NUM_LABELS, (BATCH, SEQ_LEN)),
    }


class _FakeDataLoader:
    def __init__(self, n=3):
        self.n = n

    def __len__(self):
        return self.n

    def __iter__(self):
        for _ in range(self.n):
            yield _make_batch()


@pytest.fixture
def config(base_config_dict):
    return ExperimentConfig.from_dict(base_config_dict)


@pytest.fixture
def trainer(config, tmp_path):
    model = TinyModel()
    train_dl = _FakeDataLoader(3)
    eval_dl = _FakeDataLoader(2)
    return LayoutLMTrainer(
        model=model,
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_trainer_initial_state(trainer):
    assert trainer.global_step == 0
    assert trainer.epoch == 0
    assert trainer.best_metric == 0.0


def test_trainer_optimizer_setup(trainer):
    from torch.optim import AdamW
    assert isinstance(trainer.optimizer, AdamW)


def test_trainer_scheduler_none_when_disabled(base_config_dict, tmp_path):
    base_config_dict["training"]["scheduler"] = "none"
    cfg = ExperimentConfig.from_dict(base_config_dict)
    model = TinyModel()
    t = LayoutLMTrainer(
        model=model,
        train_dataloader=_FakeDataLoader(2),
        eval_dataloader=None,
        config=cfg,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    assert t.scheduler is None


def test_trainer_linear_scheduler(base_config_dict):
    base_config_dict["training"]["scheduler"] = "linear"
    cfg = ExperimentConfig.from_dict(base_config_dict)
    model = TinyModel()
    t = LayoutLMTrainer(
        model=model,
        train_dataloader=_FakeDataLoader(3),
        eval_dataloader=None,
        config=cfg,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    assert t.scheduler is not None


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------


def test_train_runs_without_error(trainer):
    result = trainer.train()
    assert "best_metric" in result


def test_train_returns_best_metric_when_no_eval(base_config_dict, tmp_path):
    base_config_dict["training"]["num_epochs"] = 1
    cfg = ExperimentConfig.from_dict(base_config_dict)
    model = TinyModel()
    t = LayoutLMTrainer(
        model=model,
        train_dataloader=_FakeDataLoader(2),
        eval_dataloader=None,
        config=cfg,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    result = t.train()
    assert result["best_metric"] == 0.0  # no eval → best_metric stays 0


def test_train_increments_global_step(trainer):
    trainer.train()
    assert trainer.global_step > 0


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


def test_evaluate_returns_f1_key(trainer):
    metrics = trainer.evaluate()
    assert "eval_f1" in metrics or "f1" in {k.replace("eval_", "") for k in metrics}


def test_evaluate_returns_empty_without_eval_loader(base_config_dict):
    cfg = ExperimentConfig.from_dict(base_config_dict)
    model = TinyModel()
    t = LayoutLMTrainer(
        model=model,
        train_dataloader=_FakeDataLoader(2),
        eval_dataloader=None,
        config=cfg,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    result = t.evaluate()
    assert result == {}


# ---------------------------------------------------------------------------
# save_model / load_model round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip(trainer, tmp_path):
    save_path = tmp_path / "saved_model"
    # Modify a known weight before saving
    with torch.no_grad():
        trainer.model.classifier.weight.fill_(1.23)
    trainer.save_model(save_path)

    # Reset model weights
    with torch.no_grad():
        trainer.model.classifier.weight.fill_(0.0)

    trainer.load_model(save_path)
    assert torch.allclose(trainer.model.classifier.weight, torch.ones_like(trainer.model.classifier.weight) * 1.23)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


def test_early_stopping_triggers(base_config_dict, tmp_path):
    """Verify early stopping counter fires after patience epochs."""
    base_config_dict["training"]["num_epochs"] = 10
    base_config_dict["training"]["early_stopping_patience"] = 2
    cfg = ExperimentConfig.from_dict(base_config_dict)
    model = TinyModel()

    # Use a mock evaluator that always returns the same (non-improving) metric
    dl = _FakeDataLoader(2)
    t = LayoutLMTrainer(
        model=model,
        train_dataloader=dl,
        eval_dataloader=dl,
        config=cfg,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    t.train()
    # Should have stopped before epoch 10
    assert t.epoch < 9


# ---------------------------------------------------------------------------
# Dict config path (line 37)
# ---------------------------------------------------------------------------


def test_trainer_accepts_dict_config(base_config_dict, tmp_path):
    """LayoutLMTrainer accepts legacy dict config (line 37)."""
    t = LayoutLMTrainer(
        model=TinyModel(),
        train_dataloader=_FakeDataLoader(1),
        eval_dataloader=None,
        config=base_config_dict,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    assert t.config is not None


# ---------------------------------------------------------------------------
# Unsupported optimizer raises (line 86)
# ---------------------------------------------------------------------------


def test_unsupported_optimizer_raises(base_config_dict, tmp_path):
    """Unsupported optimizer name raises ValueError (line 86)."""
    base_config_dict["training"]["optimizer"] = "sgd"
    cfg = ExperimentConfig.from_dict(base_config_dict)
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        LayoutLMTrainer(
            model=TinyModel(),
            train_dataloader=_FakeDataLoader(1),
            eval_dataloader=None,
            config=cfg,
            label_list=LABEL_LIST,
            id2label=ID2LABEL,
        )


# ---------------------------------------------------------------------------
# Cosine scheduler (lines 106-112)
# ---------------------------------------------------------------------------


def test_cosine_scheduler_created(base_config_dict, tmp_path):
    """Cosine scheduler is created when specified (lines 106-112)."""
    base_config_dict["training"]["scheduler"] = "cosine"
    cfg = ExperimentConfig.from_dict(base_config_dict)
    t = LayoutLMTrainer(
        model=TinyModel(),
        train_dataloader=_FakeDataLoader(2),
        eval_dataloader=None,
        config=cfg,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    assert t.scheduler is not None


def test_unsupported_scheduler_raises(base_config_dict, tmp_path):
    """Unsupported scheduler name raises ValueError."""
    base_config_dict["training"]["scheduler"] = "polynomial"
    cfg = ExperimentConfig.from_dict(base_config_dict)
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        LayoutLMTrainer(
            model=TinyModel(),
            train_dataloader=_FakeDataLoader(2),
            eval_dataloader=None,
            config=cfg,
            label_list=LABEL_LIST,
            id2label=ID2LABEL,
        )


# ---------------------------------------------------------------------------
# save_best_model path and load_best_model_at_end (lines 142-144, 155-156)
# ---------------------------------------------------------------------------


def test_save_best_model_and_load_at_end(base_config_dict, tmp_path):
    """save_best_model=True triggers model save; load_best_model_at_end loads it (lines 142-156)."""
    base_config_dict["training"]["save_best_model"] = True
    base_config_dict["training"]["load_best_model_at_end"] = True
    base_config_dict["output_dir"] = str(tmp_path / "output")
    cfg = ExperimentConfig.from_dict(base_config_dict)
    t = LayoutLMTrainer(
        model=TinyModel(),
        train_dataloader=_FakeDataLoader(2),
        eval_dataloader=_FakeDataLoader(2),
        config=cfg,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    result = t.train()
    assert "best_metric" in result


# ---------------------------------------------------------------------------
# gradient_accumulation_steps > 1 loss scaling (line 176)
# ---------------------------------------------------------------------------


def test_gradient_accumulation_steps(base_config_dict, tmp_path):
    """ga_steps=2 exercises loss scaling (line 176) and logging path (192-193)."""
    base_config_dict["training"]["gradient_accumulation_steps"] = 2
    base_config_dict["training"]["log_steps"] = 1  # forces log every step
    cfg = ExperimentConfig.from_dict(base_config_dict)
    t = LayoutLMTrainer(
        model=TinyModel(),
        train_dataloader=_FakeDataLoader(4),
        eval_dataloader=None,
        config=cfg,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    t.train()
    assert t.global_step > 0


# ---------------------------------------------------------------------------
# evaluate with increment_step=True (lines 264-265)
# ---------------------------------------------------------------------------


def test_evaluate_increment_step(trainer):
    """increment_step=True increments global_step after eval (lines 264-265)."""
    step_before = trainer.global_step
    trainer.evaluate(increment_step=True)
    assert trainer.global_step == step_before + 1


# ---------------------------------------------------------------------------
# scheduler save/load in training state (lines 295, 319)
# ---------------------------------------------------------------------------


def test_save_load_with_scheduler(base_config_dict, tmp_path):
    """Save/load training state with a scheduler active (lines 295, 319)."""
    base_config_dict["training"]["scheduler"] = "linear"
    base_config_dict["output_dir"] = str(tmp_path / "output")
    cfg = ExperimentConfig.from_dict(base_config_dict)
    save_path = tmp_path / "ckpt"

    t = LayoutLMTrainer(
        model=TinyModel(),
        train_dataloader=_FakeDataLoader(2),
        eval_dataloader=None,
        config=cfg,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    t.save_model(save_path)  # exercises line 295 (scheduler state dict saved)
    t.load_model(save_path)  # exercises line 319 (scheduler state dict loaded)


# ---------------------------------------------------------------------------
# create_trainer factory (lines 339-346)
# ---------------------------------------------------------------------------


def test_create_trainer_factory(base_config_dict, tmp_path):
    """create_trainer returns a LayoutLMTrainer (lines 339-346)."""
    from src.training.layoutlm_trainer import create_trainer
    cfg = ExperimentConfig.from_dict(base_config_dict)
    t = create_trainer(
        model=TinyModel(),
        train_dataloader=_FakeDataLoader(1),
        eval_dataloader=None,
        config=cfg,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    assert isinstance(t, LayoutLMTrainer)


# ---------------------------------------------------------------------------
# scheduler.step() during training (line 186)
# ---------------------------------------------------------------------------


def test_linear_scheduler_steps_during_train(base_config_dict, tmp_path):
    """Linear scheduler step() is called each optimizer step (line 186)."""
    base_config_dict["training"]["scheduler"] = "linear"
    base_config_dict["training"]["num_epochs"] = 1
    cfg = ExperimentConfig.from_dict(base_config_dict)
    t = LayoutLMTrainer(
        model=TinyModel(),
        train_dataloader=_FakeDataLoader(3),
        eval_dataloader=None,
        config=cfg,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    t.train()
    assert t.global_step > 0


# ---------------------------------------------------------------------------
# cleanup with wandb = None (lines 325-327 NOT triggered; just smoke test)
# ---------------------------------------------------------------------------


def test_cleanup_no_wandb(trainer):
    """cleanup() does nothing when wandb_run is None (line 325 is False)."""
    assert trainer.wandb_run is None
    trainer.cleanup()  # should not raise


# ---------------------------------------------------------------------------
# wandb logging in _log_metrics (lines 275-277) + cleanup (lines 326-327)
# ---------------------------------------------------------------------------


def test_log_metrics_with_wandb(trainer):
    """When wandb_run is set, _log_metrics logs to it (lines 275-277)."""
    mock_run = MagicMock()
    trainer.wandb_run = mock_run
    with patch('src.training.layoutlm_trainer.wandb') as mock_wandb:
        trainer._log_metrics({"eval_f1": 0.85})
        mock_wandb.log.assert_called()


def test_cleanup_with_wandb(trainer):
    """cleanup() finishes wandb run when wandb_run is set (lines 326-327)."""
    mock_run = MagicMock()
    trainer.wandb_run = mock_run
    trainer.cleanup()
    mock_run.finish.assert_called_once()
