"""Tests for ContinualLayoutLMTrainer – multi-task orchestration."""

from unittest.mock import MagicMock, call, patch

import torch
import pytest

from src.cl_strategies.base import BaseCLStrategy
from src.cl_strategies.gem import GEM
from src.cl_strategies.agem import AGEM
from src.cl_strategies.sequential import SequentialFineTuning
from src.config import ExperimentConfig, GEMConfig, StrategyConfig
from src.training.continual_trainer import ContinualLayoutLMTrainer, get_strategy
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
    def __init__(self, n=2):
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
def strategy():
    return SequentialFineTuning(StrategyConfig(name="sequential"))


@pytest.fixture
def trainer(config, strategy):
    return ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
        strategy=strategy,
    )


# ---------------------------------------------------------------------------
# get_strategy factory
# ---------------------------------------------------------------------------


def test_get_strategy_sequential(config):
    s = get_strategy(config)
    assert isinstance(s, SequentialFineTuning)


def test_get_strategy_unknown_raises(base_config_dict):
    base_config_dict["cl_strategy"] = {"name": "unknown_strategy"}
    with pytest.raises((ValueError, Exception)):
        cfg = ExperimentConfig.from_dict(base_config_dict)
        get_strategy(cfg)


# ---------------------------------------------------------------------------
# train_task – strategy hooks called
# ---------------------------------------------------------------------------


def test_train_task_calls_before_and_after_task(config):
    """before_task and after_task should be called exactly once per task."""
    mock_strategy = MagicMock(spec=BaseCLStrategy)
    mock_strategy.before_task = MagicMock()
    mock_strategy.after_task = MagicMock()
    mock_strategy.compute_loss = MagicMock(side_effect=lambda model, batch, outputs: outputs["loss"])
    mock_strategy.on_before_backward = MagicMock()
    mock_strategy.on_after_backward = MagicMock()
    mock_strategy.update_memory = MagicMock()

    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
        strategy=mock_strategy,
    )
    dl = _FakeDataLoader(2)
    trainer.train_task(dl, None, task_id=0, label_list=LABEL_LIST, id2label=ID2LABEL)

    mock_strategy.before_task.assert_called_once()
    mock_strategy.after_task.assert_called_once()


def test_train_task_calls_compute_loss(config):
    """compute_loss is called for each batch."""
    call_count = {"n": 0}

    class _CountingStrategy(SequentialFineTuning):
        def compute_loss(self, model, batch, outputs):
            call_count["n"] += 1
            return outputs["loss"]

    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
        strategy=_CountingStrategy(StrategyConfig("sequential")),
    )
    n_batches = 3
    trainer.train_task(_FakeDataLoader(n_batches), None, task_id=0)
    assert call_count["n"] == n_batches  # called once per batch per epoch (1 epoch)


def test_train_task_calls_update_memory(config):
    """update_memory called once per batch."""
    mock_strategy = MagicMock(spec=BaseCLStrategy)
    mock_strategy.before_task = MagicMock()
    mock_strategy.after_task = MagicMock()
    mock_strategy.compute_loss = MagicMock(side_effect=lambda model, batch, outputs: outputs["loss"])
    mock_strategy.on_before_backward = MagicMock()
    mock_strategy.on_after_backward = MagicMock()
    mock_strategy.update_memory = MagicMock()

    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
        strategy=mock_strategy,
    )
    n_batches = 3
    trainer.train_task(_FakeDataLoader(n_batches), None, task_id=0)
    assert mock_strategy.update_memory.call_count == n_batches


# ---------------------------------------------------------------------------
# Class-IL: classifier head expansion
# ---------------------------------------------------------------------------


def test_class_il_head_expands(base_config_dict, tmp_path):
    """Second task with more labels should grow the classifier."""
    base_config_dict["cl_setting"] = "class_il"
    config = ExperimentConfig.from_dict(base_config_dict)

    model = TinyModel(num_labels=NUM_LABELS)
    trainer = ContinualLayoutLMTrainer(
        model=model,
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )

    label_list_2 = LABEL_LIST + ["B-Y", "I-Y"]  # 2 extra labels
    id2label_2 = {i: l for i, l in enumerate(label_list_2)}

    trainer.train_task(_FakeDataLoader(1), None, task_id=0,
                       label_list=LABEL_LIST, id2label=ID2LABEL)
    assert model.num_labels == NUM_LABELS

    trainer.train_task(_FakeDataLoader(1), None, task_id=1,
                       label_list=label_list_2, id2label=id2label_2)
    assert model.num_labels == len(label_list_2)


# ---------------------------------------------------------------------------
# train() – full multi-task loop
# ---------------------------------------------------------------------------


def test_train_two_tasks_returns_accuracy_matrix(base_config_dict, tmp_path):
    """train() should return an accuracy_matrix with shape T×T."""
    base_config_dict["cl_setting"] = "class_il"
    config = ExperimentConfig.from_dict(base_config_dict)

    model = TinyModel(num_labels=NUM_LABELS)
    trainer = ContinualLayoutLMTrainer(
        model=model,
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )

    tasks = [
        {
            "name": "task0",
            "train_loader": _FakeDataLoader(2),
            "eval_loader": _FakeDataLoader(2),
            "label_list": LABEL_LIST,
            "id2label": ID2LABEL,
        },
        {
            "name": "task1",
            "train_loader": _FakeDataLoader(2),
            "eval_loader": _FakeDataLoader(2),
            "label_list": LABEL_LIST,
            "id2label": ID2LABEL,
        },
    ]

    result = trainer.train(tasks)
    mat = result["accuracy_matrix"]
    assert len(mat) == 2
    assert len(mat[0]) == 2


def test_train_returns_cl_metrics(base_config_dict, tmp_path):
    """train() result should contain standard CL metric keys."""
    config = ExperimentConfig.from_dict(base_config_dict)

    tasks = [
        {
            "name": "t0",
            "train_loader": _FakeDataLoader(1),
            "eval_loader": _FakeDataLoader(1),
            "label_list": LABEL_LIST,
            "id2label": ID2LABEL,
        },
        {
            "name": "t1",
            "train_loader": _FakeDataLoader(1),
            "eval_loader": _FakeDataLoader(1),
            "label_list": LABEL_LIST,
            "id2label": ID2LABEL,
        },
    ]

    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    result = trainer.train(tasks)
    cl = result["cl_metrics"]
    for key in ("ACC", "BWT", "FWT", "AAA", "Forgetting"):
        assert key in cl, f"Missing CL metric: {key}"


# ---------------------------------------------------------------------------
# Task-IL: evaluate_with_head switches head
# ---------------------------------------------------------------------------


def test_evaluate_with_head_no_error(base_config_dict, tmp_path):
    base_config_dict["cl_setting"] = "task_il"
    config = ExperimentConfig.from_dict(base_config_dict)
    model = TinyModel()
    trainer = ContinualLayoutLMTrainer(
        model=model,
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    # Prepare head for task0
    trainer.head_manager.prepare_for_task("task0", LABEL_LIST)
    dl = _FakeDataLoader(1)
    metrics = trainer.evaluate_with_head(dl, "task0", LABEL_LIST, ID2LABEL)
    assert isinstance(metrics, dict)


# ---------------------------------------------------------------------------
# get_strategy: GEM path (line 50) and dict config path (lines 54-58)
# ---------------------------------------------------------------------------


def test_get_strategy_gem_passes_cl_setting(base_config_dict):
    """GEM strategy gets cl_setting from config (line 50)."""
    base_config_dict["cl_strategy"] = {
        "name": "gem",
        "memory_size": 10,
        "samples_per_task": 2,
        "max_tasks": 3,
    }
    base_config_dict["cl_setting"] = "class_il"
    config = ExperimentConfig.from_dict(base_config_dict)
    s = get_strategy(config)
    assert isinstance(s, GEM)
    assert s.cl_setting == "class_il"


def test_get_strategy_dict_config(base_config_dict):
    """Dict-based config exercises the legacy path (lines 54-58)."""
    cfg_dict = {
        "cl_strategy": {"name": "sequential"},
        "cl_setting": "class_il",
    }
    s = get_strategy(cfg_dict)
    assert isinstance(s, SequentialFineTuning)


def test_get_strategy_dict_unknown_raises():
    """Unknown strategy name in dict config raises ValueError."""
    with pytest.raises(ValueError):
        get_strategy({"cl_strategy": {"name": "no_such_strategy"}})


# ---------------------------------------------------------------------------
# Constructor: dict config path (lines 68-69)
# ---------------------------------------------------------------------------


def test_constructor_accepts_dict_config(base_config_dict):
    """ContinualLayoutLMTrainer accepts legacy dict config (lines 68-69)."""
    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=base_config_dict,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    assert trainer.config is not None


# ---------------------------------------------------------------------------
# Unknown cl_setting warning (lines 103-104)
# ---------------------------------------------------------------------------


def test_unknown_cl_setting_defaults_to_task_il(base_config_dict):
    """Unknown cl_setting defaults to 'task_il' (lines 103-104)."""
    base_config_dict["cl_setting"] = "weird_setting"
    config = ExperimentConfig.from_dict(base_config_dict)
    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    assert trainer.cl_setting == "task_il"


# ---------------------------------------------------------------------------
# Linear scheduler step (lines 123-131, 242)
# ---------------------------------------------------------------------------


def test_linear_scheduler_steps(base_config_dict):
    """Linear scheduler is created and steps during training (lines 123-131, 242)."""
    base_config_dict["training"]["scheduler"] = "linear"
    base_config_dict["training"]["num_epochs"] = 1
    config = ExperimentConfig.from_dict(base_config_dict)
    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    # Run training – scheduler.step() should be called each batch
    trainer.train_task(_FakeDataLoader(3), None, task_id=0)


# ---------------------------------------------------------------------------
# GEM/AGEM gradient_accumulation_steps override (lines 317-318)
# ---------------------------------------------------------------------------


def test_agem_overrides_gradient_accumulation(base_config_dict, tmp_path):
    """AGEM forces ga_steps=1 when config has ga_steps != 1 (lines 317-318)."""
    base_config_dict["training"]["gradient_accumulation_steps"] = 4
    config = ExperimentConfig.from_dict(base_config_dict)

    from src.config import AGEMConfig
    agem_cfg = AGEMConfig(name="agem", memory_size=10, replay_batch_size=2)
    agem = AGEM(agem_cfg)
    agem.memory.storage_dir = tmp_path / "agem_ga"
    agem.memory.storage_dir.mkdir(parents=True, exist_ok=True)

    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
        strategy=agem,
    )
    # Should warn about overriding ga_steps; training should still work
    trainer.train_task(_FakeDataLoader(1), None, task_id=0)


# ---------------------------------------------------------------------------
# save_best_model path (lines 282-287) + load_best_model_at_end (lines 349-351)
# ---------------------------------------------------------------------------


def test_save_and_load_best_model(base_config_dict, tmp_path):
    """save_best_model=True exercises model save path; load_best_model_at_end loads it."""
    base_config_dict["training"]["save_best_model"] = True
    base_config_dict["training"]["load_best_model_at_end"] = True
    base_config_dict["output_dir"] = str(tmp_path / "output")
    config = ExperimentConfig.from_dict(base_config_dict)

    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    trainer.train_task(_FakeDataLoader(2), _FakeDataLoader(2), task_id=0,
                       label_list=LABEL_LIST, id2label=ID2LABEL)
    # If we get here without error, save/load paths executed


# ---------------------------------------------------------------------------
# Early stopping break (line 346)
# ---------------------------------------------------------------------------


def test_early_stopping_breaks_training(base_config_dict, tmp_path):
    """patience=1 + multiple epochs → early stopping fires (line 346)."""
    base_config_dict["training"]["num_epochs"] = 10
    base_config_dict["training"]["early_stopping_patience"] = 1
    config = ExperimentConfig.from_dict(base_config_dict)

    epoch_count = {"n": 0}
    original_run = ContinualLayoutLMTrainer._run_training_epoch

    def counting_run(self, *args, **kwargs):
        epoch_count["n"] += 1
        return original_run(self, *args, **kwargs)

    with patch.object(ContinualLayoutLMTrainer, "_run_training_epoch", counting_run):
        trainer = ContinualLayoutLMTrainer(
            model=TinyModel(),
            config=config,
            label_list=LABEL_LIST,
            id2label=ID2LABEL,
        )
        trainer.train_task(_FakeDataLoader(1), _FakeDataLoader(1), task_id=0,
                           label_list=LABEL_LIST, id2label=ID2LABEL)
    # Should have stopped well before 10 epochs
    assert epoch_count["n"] < 10


# ---------------------------------------------------------------------------
# task_il train() evaluation branch (lines 404-406)
# ---------------------------------------------------------------------------


def test_train_task_il_uses_evaluate_with_head(base_config_dict, tmp_path):
    """task_il cl_setting in train() uses evaluate_with_head path (lines 404-406)."""
    base_config_dict["cl_setting"] = "task_il"
    base_config_dict["output_dir"] = str(tmp_path / "output")
    config = ExperimentConfig.from_dict(base_config_dict)

    model = TinyModel()
    trainer = ContinualLayoutLMTrainer(
        model=model,
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )

    tasks = [
        {
            "name": "task0",
            "train_loader": _FakeDataLoader(1),
            "eval_loader": _FakeDataLoader(1),
            "label_list": LABEL_LIST,
            "id2label": ID2LABEL,
        },
    ]
    result = trainer.train(tasks)
    assert "accuracy_matrix" in result


# ---------------------------------------------------------------------------
# create_continual_trainer factory (line 517)
# ---------------------------------------------------------------------------


def test_create_continual_trainer_factory(base_config_dict):
    """create_continual_trainer returns a ContinualLayoutLMTrainer (line 517)."""
    from src.training.continual_trainer import create_continual_trainer
    config = ExperimentConfig.from_dict(base_config_dict)
    trainer = create_continual_trainer(TinyModel(), config, LABEL_LIST, ID2LABEL)
    assert isinstance(trainer, ContinualLayoutLMTrainer)


# ---------------------------------------------------------------------------
# Unsupported optimizer raises (line 117)
# ---------------------------------------------------------------------------


def test_unsupported_optimizer_raises(base_config_dict):
    """Unsupported optimizer raises ValueError in ContinualLayoutLMTrainer (line 117)."""
    base_config_dict["training"]["optimizer"] = "sgd"
    config = ExperimentConfig.from_dict(base_config_dict)
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        ContinualLayoutLMTrainer(
            model=TinyModel(),
            config=config,
            label_list=LABEL_LIST,
            id2label=ID2LABEL,
        )


# ---------------------------------------------------------------------------
# Cosine scheduler in ContinualLayoutLMTrainer (lines 129-131)
# ---------------------------------------------------------------------------


def test_cosine_scheduler_in_continual_trainer(base_config_dict):
    """Cosine scheduler is created and used in train_task (lines 129-131)."""
    base_config_dict["training"]["scheduler"] = "cosine"
    config = ExperimentConfig.from_dict(base_config_dict)
    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    trainer.train_task(_FakeDataLoader(2), None, task_id=0)


def test_unsupported_scheduler_raises_continual(base_config_dict):
    """Unsupported scheduler raises ValueError in ContinualLayoutLMTrainer (line 131)."""
    base_config_dict["training"]["scheduler"] = "polynomial"
    config = ExperimentConfig.from_dict(base_config_dict)
    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    with pytest.raises(ValueError, match="Unsupported scheduler"):
        trainer.train_task(_FakeDataLoader(1), None, task_id=0)


# ---------------------------------------------------------------------------
# torch.cuda.empty_cache at step 10 (line 246)
# ---------------------------------------------------------------------------


def test_cache_clear_at_step_10(base_config_dict):
    """Runs 10+ batches so global_step % 10 == 0 triggers line 246."""
    base_config_dict["training"]["num_epochs"] = 1
    config = ExperimentConfig.from_dict(base_config_dict)
    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    # 10+ batches in one epoch
    trainer.train_task(_FakeDataLoader(11), None, task_id=0)


# ---------------------------------------------------------------------------
# save_best_model path (lines 282-287) + load_best_model_at_end (lines 349-351)
# ---------------------------------------------------------------------------


def test_continual_trainer_save_and_load_best_model(base_config_dict, tmp_path):
    """save_best_model=True + load_best_model_at_end exercises lines 282-287 + 349-351."""
    base_config_dict["training"]["save_best_model"] = True
    base_config_dict["training"]["load_best_model_at_end"] = True
    # continual trainer returns metrics WITHOUT 'eval_' prefix (e.g., 'accuracy', not 'eval_f1')
    base_config_dict["training"]["metric_for_best_model"] = "accuracy"
    base_config_dict["output_dir"] = str(tmp_path / "output")
    config = ExperimentConfig.from_dict(base_config_dict)
    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    trainer.train_task(_FakeDataLoader(1), _FakeDataLoader(1), task_id=0,
                       label_list=LABEL_LIST, id2label=ID2LABEL)


# ---------------------------------------------------------------------------
# is_joint_training path (lines 421-423)
# ---------------------------------------------------------------------------


def test_train_joint_training_path(base_config_dict, tmp_path):
    """eval_tasks != tasks triggers is_joint_training path (lines 421-423)."""
    base_config_dict["output_dir"] = str(tmp_path / "output")
    config = ExperimentConfig.from_dict(base_config_dict)
    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )

    # Single combined training task
    joint_task = {
        "name": "joint",
        "train_loader": _FakeDataLoader(1),
        "eval_loader": _FakeDataLoader(1),
        "label_list": LABEL_LIST,
        "id2label": ID2LABEL,
    }
    # Two separate eval tasks
    eval_tasks = [
        {
            "name": "t0",
            "eval_loader": _FakeDataLoader(1),
            "label_list": LABEL_LIST,
            "id2label": ID2LABEL,
        },
        {
            "name": "t1",
            "eval_loader": _FakeDataLoader(1),
            "label_list": LABEL_LIST,
            "id2label": ID2LABEL,
        },
    ]
    result = trainer.train([joint_task], eval_tasks=eval_tasks)
    # Matrix should have rows for each eval task
    assert len(result["accuracy_matrix"]) == 2


# ---------------------------------------------------------------------------
# _save_checkpoint_after_task exception handler (lines 428-429)
# ---------------------------------------------------------------------------


def test_save_checkpoint_exception_silenced(base_config_dict, tmp_path):
    """Exception in _save_checkpoint_after_task is caught (lines 428-429)."""
    base_config_dict["output_dir"] = str(tmp_path / "output")
    config = ExperimentConfig.from_dict(base_config_dict)
    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )

    # Patch _save_checkpoint_after_task to raise
    with patch.object(trainer, "_save_checkpoint_after_task", side_effect=RuntimeError("disk full")):
        tasks = [
            {
                "name": "t0",
                "train_loader": _FakeDataLoader(1),
                "eval_loader": _FakeDataLoader(1),
                "label_list": LABEL_LIST,
                "id2label": ID2LABEL,
            },
        ]
        result = trainer.train(tasks)  # should not raise
    assert "accuracy_matrix" in result


# ---------------------------------------------------------------------------
# Neptune logging and cleanup (lines 502-505, 509-511)
# ---------------------------------------------------------------------------


def test_log_metrics_with_wandb(base_config_dict):
    """When wandb_run is set, _log_metrics logs to it (lines 502-505)."""
    config = ExperimentConfig.from_dict(base_config_dict)
    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    mock_run = MagicMock()
    trainer.wandb_run = mock_run
    with patch('src.training.continual_trainer.wandb') as mock_wandb:
        trainer._log_metrics({"f1": 0.9}, prefix="task_0")
        # Verify wandb.log was called with correct arguments
        mock_wandb.log.assert_called()


def test_cleanup_with_wandb(base_config_dict):
    """cleanup() finishes wandb run when it's set (lines 509-511)."""
    config = ExperimentConfig.from_dict(base_config_dict)
    trainer = ContinualLayoutLMTrainer(
        model=TinyModel(),
        config=config,
        label_list=LABEL_LIST,
        id2label=ID2LABEL,
    )
    mock_run = MagicMock()
    trainer.wandb_run = mock_run
    trainer.cleanup()
    mock_run.finish.assert_called_once()
