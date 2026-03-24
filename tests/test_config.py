"""Tests for src/config.py – ExperimentConfig and strategy config factories."""

import dataclasses

import pytest

from src.config import (
    AGEMConfig,
    ERConfig,
    EWCConfig,
    ExperimentConfig,
    GEMConfig,
    LwFConfig,
    ModelConfig,
    ModelInnerConfig,
    StrategyConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# ModelInnerConfig
# ---------------------------------------------------------------------------


def test_model_inner_config_positive():
    cfg = ModelInnerConfig(num_labels=5)
    assert cfg.num_labels == 5
    assert cfg.hidden_dropout_prob == 0.1


def test_model_inner_config_defaults():
    cfg = ModelInnerConfig.from_dict({"num_labels": 3})
    assert cfg.hidden_dropout_prob == 0.1
    assert cfg.attention_probs_dropout_prob == 0.1
    assert cfg.classifier_dropout == 0.1


def test_model_inner_config_zero_labels_raises():
    with pytest.raises(ValueError, match="num_labels must be positive"):
        ModelInnerConfig(num_labels=0)


def test_model_inner_config_negative_labels_raises():
    with pytest.raises(ValueError):
        ModelInnerConfig(num_labels=-1)


# ---------------------------------------------------------------------------
# StrategyConfig factory dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected_cls",
    [
        ("sequential", StrategyConfig),
        ("none", StrategyConfig),
        ("er", ERConfig),
        ("experience_replay", ERConfig),
        ("ewc", EWCConfig),
        ("gem", GEMConfig),
        ("agem", AGEMConfig),
        ("a-gem", AGEMConfig),
    ],
)
def test_strategy_config_dispatch(name, expected_cls):
    cfg = StrategyConfig.from_dict({"name": name})
    assert isinstance(cfg, expected_cls)
    assert cfg.name == name


def test_strategy_config_lwf_with_unified():
    cfg = StrategyConfig.from_dict({"name": "lwf"}, unified_label_space=True)
    assert isinstance(cfg, LwFConfig)
    assert cfg.unified_label_space is True


def test_lwf_config_raises_without_unified():
    with pytest.raises(ValueError, match="unified label space"):
        LwFConfig(unified_label_space=False)


def test_lwf_config_raises_via_factory():
    with pytest.raises(ValueError):
        StrategyConfig.from_dict({"name": "lwf"}, unified_label_space=False)


# ---------------------------------------------------------------------------
# ERConfig defaults
# ---------------------------------------------------------------------------


def test_er_config_defaults():
    cfg = ERConfig.from_dict({"name": "er"})
    assert cfg.memory_size == 2000
    assert cfg.replay_batch_size == 32
    assert cfg.replay_weight == 1.0


def test_er_config_custom():
    cfg = ERConfig.from_dict({"name": "er", "memory_size": 500, "replay_weight": 0.5})
    assert cfg.memory_size == 500
    assert cfg.replay_weight == 0.5


# ---------------------------------------------------------------------------
# EWCConfig
# ---------------------------------------------------------------------------


def test_ewc_config_defaults():
    cfg = EWCConfig.from_dict({"name": "ewc"})
    assert cfg.ewc_lambda == 0.4
    assert cfg.store_fishers_on_cpu is True
    assert cfg.n_fisher_samples is None


def test_ewc_config_custom_n_samples():
    cfg = EWCConfig.from_dict({"name": "ewc", "n_fisher_samples": 100})
    assert cfg.n_fisher_samples == 100


# ---------------------------------------------------------------------------
# GEMConfig / AGEMConfig
# ---------------------------------------------------------------------------


def test_gem_config_defaults():
    cfg = GEMConfig.from_dict({"name": "gem"})
    assert cfg.memory_size == 500
    assert cfg.samples_per_task == 5
    assert cfg.margin == 0.5


def test_agem_config_defaults():
    cfg = AGEMConfig.from_dict({"name": "agem"})
    assert cfg.memory_size == 1000
    assert cfg.constraint_threshold == pytest.approx(-1e-6)


# ---------------------------------------------------------------------------
# TrainingConfig
# ---------------------------------------------------------------------------


def test_training_config_defaults():
    cfg = TrainingConfig.from_dict({"batch_size": 4, "learning_rate": 1e-4, "num_epochs": 3})
    assert cfg.weight_decay == 0.01
    assert cfg.optimizer == "adamw"
    assert cfg.scheduler == "linear"
    assert cfg.early_stopping_patience == 10
    assert cfg.eval_steps is None
    assert cfg.persistent_workers is None


# ---------------------------------------------------------------------------
# ExperimentConfig round-trip
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_config_dict():
    return {
        "experiment_name": "test_exp",
        "output_dir": "/tmp/test",
        "cl_setting": "class_il",
        "model": {
            "name": "test-model",
            "model_type": "layoutlmv3",
            "pretrained_model_name": "microsoft/layoutlmv3-base",
            "config": {"num_labels": 5},
        },
        "training": {
            "batch_size": 2,
            "learning_rate": 5e-5,
            "num_epochs": 2,
        },
        "cl_strategy": {"name": "sequential"},
    }


def test_experiment_config_round_trip(minimal_config_dict):
    cfg = ExperimentConfig.from_dict(minimal_config_dict)
    assert cfg.experiment_name == "test_exp"
    assert cfg.cl_setting == "class_il"
    assert cfg.model.model_type == "layoutlmv3"
    assert cfg.model.config.num_labels == 5
    assert cfg.training.learning_rate == pytest.approx(5e-5)
    assert isinstance(cfg.cl_strategy, StrategyConfig)


def test_experiment_config_with_output_dir(minimal_config_dict):
    cfg = ExperimentConfig.from_dict(minimal_config_dict)
    new_cfg = cfg.with_output_dir("/new/path")
    assert new_cfg.output_dir == "/new/path"
    assert cfg.output_dir == "/tmp/test"  # original unchanged


def test_experiment_config_replace_model_num_labels(minimal_config_dict):
    cfg = ExperimentConfig.from_dict(minimal_config_dict)
    new_cfg = cfg.replace_model_num_labels(10)
    assert new_cfg.model.config.num_labels == 10
    assert cfg.model.config.num_labels == 5  # original unchanged


def test_experiment_config_with_task_overrides(minimal_config_dict):
    cfg = ExperimentConfig.from_dict(minimal_config_dict)
    overrides = {"training": {"num_epochs": 5}, "output_dir": "/override"}
    new_cfg = cfg.with_task_overrides(overrides)
    assert new_cfg.training.num_epochs == 5
    assert new_cfg.output_dir == "/override"
    assert cfg.training.num_epochs == 2  # original unchanged


def test_experiment_config_empty_overrides(minimal_config_dict):
    cfg = ExperimentConfig.from_dict(minimal_config_dict)
    same = cfg.with_task_overrides({})
    assert same is cfg  # identity when no overrides


def test_experiment_config_defaults_label_space(minimal_config_dict):
    cfg = ExperimentConfig.from_dict(minimal_config_dict)
    assert cfg.label_space == {}
    assert cfg.neptune == {}
    assert cfg.tasks == []


def test_experiment_config_cl_setting_lowercase(minimal_config_dict):
    minimal_config_dict["cl_setting"] = "CLASS_IL"
    cfg = ExperimentConfig.from_dict(minimal_config_dict)
    assert cfg.cl_setting == "class_il"


# ---------------------------------------------------------------------------
# LwFConfig.from_dict (line 239)
# ---------------------------------------------------------------------------


def test_lwf_config_from_dict():
    """LwFConfig.from_dict with unified_label_space=True (line 239)."""
    d = {"name": "lwf", "lwf_alpha": 0.7, "lwf_temperature": 3.0}
    cfg = LwFConfig.from_dict(d, unified_label_space=True)
    assert cfg.lwf_alpha == pytest.approx(0.7)
    assert cfg.lwf_temperature == pytest.approx(3.0)
    assert cfg.unified_label_space is True


# ---------------------------------------------------------------------------
# with_task_overrides: model field and dict field merging (lines 335-364)
# ---------------------------------------------------------------------------


def test_with_task_overrides_model_field(minimal_config_dict):
    """Model overrides in with_task_overrides exercise lines 335-337."""
    cfg = ExperimentConfig.from_dict(minimal_config_dict)
    overrides = {"model": {"config": {"num_labels": 7}}}
    new_cfg = cfg.with_task_overrides(overrides)
    assert new_cfg.model.config.num_labels == 7
    assert cfg.model.config.num_labels == 5  # original unchanged


def test_with_task_overrides_dict_field(minimal_config_dict):
    """Dict field (label_space, neptune) merging exercises lines 340-346."""
    cfg = ExperimentConfig.from_dict(minimal_config_dict)
    overrides = {"label_space": {"unified": True}, "neptune": {"use_neptune": False}}
    new_cfg = cfg.with_task_overrides(overrides)
    assert new_cfg.label_space.get("unified") is True


def test_deep_merge_dicts():
    """_deep_merge_dicts exercises lines 358-364 (nested dict merging)."""
    from src.config import _deep_merge_dicts

    base = {"a": {"x": 1, "y": 2}, "b": 3}
    updates = {"a": {"y": 99, "z": 10}, "b": 4}
    result = _deep_merge_dicts(base, updates)
    assert result["a"]["x"] == 1
    assert result["a"]["y"] == 99
    assert result["a"]["z"] == 10
    assert result["b"] == 4


def test_with_task_overrides_non_dict_value(minimal_config_dict):
    """Non-dict override for a dict field exercises the else branch (line 346)."""
    cfg = ExperimentConfig.from_dict(minimal_config_dict)
    # "neptune" is normally a dict; override it with a non-dict to hit else branch
    overrides = {"neptune": "disabled"}
    new_cfg = cfg.with_task_overrides(overrides)
    assert new_cfg.neptune == "disabled"
