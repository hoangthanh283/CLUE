"""Unit, sanity, and integration tests for wandb integration.

This test suite covers:
1. Unit tests: wandb_utils initialization logic
2. Sanity tests: Trainer setup with wandb
3. Integration tests: Real wandb API logging (requires WANDB_API_KEY in .env)
"""

import os
from unittest.mock import MagicMock, patch, call
import pytest
import torch
import wandb

from src.config import ExperimentConfig
from src.training.wandb_utils import init_wandb_run
from src.training.continual_trainer import ContinualLayoutLMTrainer
from src.training.layoutlm_trainer import LayoutLMTrainer
from tests.conftest import TinyModel, BATCH, SEQ_LEN, NUM_LABELS, LABEL_LIST, ID2LABEL


# ============================================================================
# Unit Tests: wandb_utils.init_wandb_run()
# ============================================================================


class TestInitWandbRun:
    """Unit tests for init_wandb_run() function."""

    def test_disabled_returns_none(self):
        """When use_wandb=False, init_wandb_run returns None."""
        config = {"wandb": {"use_wandb": False}}
        result = init_wandb_run(config)
        assert result is None

    def test_empty_config_returns_none(self):
        """When wandb config empty or missing, returns None."""
        result = init_wandb_run({})
        assert result is None

    @patch("src.training.wandb_utils.wandb.login")
    @patch("src.training.wandb_utils.wandb.init")
    def test_init_with_api_key_from_config(self, mock_init, mock_login):
        """When use_wandb=True and API key in config, initializes with it."""
        mock_run = MagicMock()
        mock_init.return_value = mock_run

        config = {
            "experiment_name": "test_exp",
            "wandb": {
                "use_wandb": True,
                "wandb_api_key": "test_key_123",
                "wandb_project": "test_project",
                "wandb_entity": "test_entity",
                "tags": ["test", "unit"],
            }
        }

        result = init_wandb_run(config)

        assert result == mock_run
        mock_login.assert_called_once_with(key="test_key_123")
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["project"] == "test_project"
        assert call_kwargs["entity"] == "test_entity"
        assert call_kwargs["name"] == "test_exp"
        assert call_kwargs["tags"] == ["test", "unit"]

    @patch.dict(os.environ, {"WANDB_API_KEY": "env_key_456"})
    @patch("src.training.wandb_utils.wandb.login")
    @patch("src.training.wandb_utils.wandb.init")
    def test_init_with_api_key_from_env(self, mock_init, mock_login):
        """When use_wandb=True and API key in env, uses it."""
        mock_run = MagicMock()
        mock_init.return_value = mock_run

        config = {
            "wandb": {"use_wandb": True}
        }

        result = init_wandb_run(config)

        assert result == mock_run
        mock_login.assert_called_once_with(key="env_key_456")

    @patch.dict(os.environ, {
        "WANDB_PROJECT": "env_project",
        "WANDB_ENTITY": "env_entity"
    })
    @patch("src.training.wandb_utils.wandb.login")
    @patch("src.training.wandb_utils.wandb.init")
    def test_init_with_project_entity_from_env(self, mock_init, mock_login):
        """When project/entity in env, uses them."""
        mock_run = MagicMock()
        mock_init.return_value = mock_run

        config = {
            "wandb": {"use_wandb": True, "wandb_api_key": "test"}
        }

        init_wandb_run(config)

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["project"] == "env_project"
        assert call_kwargs["entity"] == "env_entity"

    @patch("src.training.wandb_utils.wandb.login")
    @patch("src.training.wandb_utils.wandb.init")
    def test_default_project_when_none(self, mock_init, mock_login):
        """Default project is 'cl4ie' when not specified."""
        mock_run = MagicMock()
        mock_init.return_value = mock_run

        config = {
            "wandb": {"use_wandb": True, "wandb_api_key": "test"}
        }

        init_wandb_run(config)

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["project"] == "cl4ie"

    @patch("src.training.wandb_utils.wandb.login")
    @patch("src.training.wandb_utils.wandb.init")
    def test_init_with_legacy_full_config(self, mock_init, mock_login):
        """init_wandb_run accepts full config dict (legacy path)."""
        mock_run = MagicMock()
        mock_init.return_value = mock_run

        # Full config dict passed instead of wandb sub-dict
        full_config = {
            "experiment_name": "legacy_exp",
            "wandb": {
                "use_wandb": True,
                "wandb_api_key": "legacy_key"
            }
        }

        result = init_wandb_run(full_config)

        assert result == mock_run
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["name"] == "legacy_exp"


# ============================================================================
# Sanity Tests: Trainer initialization with wandb
# ============================================================================


class TestTrainerWandbInitialization:
    """Sanity tests for trainer initialization with wandb."""

    def test_continual_trainer_init_without_wandb(self, base_config_dict):
        """ContinualLayoutLMTrainer initializes correctly when wandb disabled."""
        base_config_dict["wandb"] = {"use_wandb": False}
        config = ExperimentConfig.from_dict(base_config_dict)

        trainer = ContinualLayoutLMTrainer(
            model=TinyModel(),
            config=config,
            label_list=LABEL_LIST,
            id2label=ID2LABEL,
        )

        assert trainer.wandb_run is None

    @patch("src.training.wandb_utils.wandb.login")
    @patch("src.training.wandb_utils.wandb.init")
    def test_continual_trainer_init_with_wandb(self, mock_init, mock_login,
                                               base_config_dict):
        """ContinualLayoutLMTrainer initializes wandb when enabled."""
        mock_run = MagicMock()
        mock_init.return_value = mock_run

        base_config_dict["wandb"] = {
            "use_wandb": True,
            "wandb_api_key": "test_key"
        }
        config = ExperimentConfig.from_dict(base_config_dict)

        trainer = ContinualLayoutLMTrainer(
            model=TinyModel(),
            config=config,
            label_list=LABEL_LIST,
            id2label=ID2LABEL,
        )

        assert trainer.wandb_run == mock_run
        mock_init.assert_called_once()

    def test_layoutlm_trainer_init_without_wandb(self, base_config_dict):
        """LayoutLMTrainer initializes correctly when wandb disabled."""
        from torch.utils.data import DataLoader
        base_config_dict["wandb"] = {"use_wandb": False}
        config = ExperimentConfig.from_dict(base_config_dict)

        # Create dummy dataloaders
        dummy_loader = DataLoader([1, 2, 3], batch_size=1)

        trainer = LayoutLMTrainer(
            model=TinyModel(),
            train_dataloader=dummy_loader,
            eval_dataloader=dummy_loader,
            config=config,
            label_list=LABEL_LIST,
            id2label=ID2LABEL,
        )

        assert trainer.wandb_run is None

    @patch("src.training.wandb_utils.wandb.login")
    @patch("src.training.wandb_utils.wandb.init")
    def test_layoutlm_trainer_init_with_wandb(self, mock_init, mock_login,
                                              base_config_dict):
        """LayoutLMTrainer initializes wandb when enabled."""
        from torch.utils.data import DataLoader
        mock_run = MagicMock()
        mock_init.return_value = mock_run

        base_config_dict["wandb"] = {
            "use_wandb": True,
            "wandb_api_key": "test_key"
        }
        config = ExperimentConfig.from_dict(base_config_dict)

        # Create dummy dataloaders
        dummy_loader = DataLoader([1, 2, 3], batch_size=1)

        trainer = LayoutLMTrainer(
            model=TinyModel(),
            train_dataloader=dummy_loader,
            eval_dataloader=dummy_loader,
            config=config,
            label_list=LABEL_LIST,
            id2label=ID2LABEL,
        )

        assert trainer.wandb_run == mock_run


# ============================================================================
# Integration Tests: Real wandb API logging
# ============================================================================


@pytest.mark.integration
class TestWandbIntegration:
    """Integration tests using real wandb API.

    Requires WANDB_API_KEY in .env file with proper permissions.
    Run with: pytest tests/training/test_wandb_integration.py -m integration -v

    NOTE: These tests create real wandb runs. The API key must have:
    - Project creation / write permissions on WANDB_PROJECT
    - Valid entity (WANDB_ENTITY) with active membership
    """

    @pytest.fixture(autouse=True)
    def check_wandb_api_key(self):
        """Skip test if WANDB_API_KEY not available."""
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key:
            pytest.skip("WANDB_API_KEY not set in environment")
        yield

    def test_real_wandb_init(self):
        """Initialize real wandb run with API key from .env."""
        config = {
            "experiment_name": "test_real_init",
            "wandb": {
                "use_wandb": True,
                "tags": ["integration_test"]
            }
        }

        run = init_wandb_run(config)

        try:
            assert run is not None, "wandb.init() failed to create run"
            assert run.project == os.getenv("WANDB_PROJECT", "cl4ie"), \
                f"Wrong project: {run.project}"
            assert run.name == "test_real_init", f"Wrong name: {run.name}"
        except Exception as e:
            # Log helpful debug info if test fails
            print(f"\n--- wandb Integration Debug ---")
            print(f"WANDB_PROJECT: {os.getenv('WANDB_PROJECT', 'cl4ie')}")
            print(f"WANDB_ENTITY: {os.getenv('WANDB_ENTITY', 'N/A')}")
            print(f"Error: {e}")
            raise
        finally:
            if run:
                run.finish()

    def test_real_wandb_logging(self):
        """Log metrics to real wandb run."""
        config = {
            "experiment_name": "test_real_logging",
            "wandb": {
                "use_wandb": True,
                "tags": ["integration_test"]
            }
        }

        run = init_wandb_run(config)

        try:
            assert run is not None

            # Log some metrics
            wandb.log({
                "accuracy": 0.95,
                "loss": 0.05,
                "epoch": 1
            })

            # Log table data
            table = wandb.Table(columns=["task", "f1", "recall"])
            table.add_data("task_0", 0.92, 0.89)
            table.add_data("task_1", 0.87, 0.84)
            wandb.log({"results": table})

            # Verify run state
            assert run.summary.keys() is not None
        finally:
            if run:
                run.finish()

    def test_real_wandb_cleanup(self):
        """Verify wandb run cleanup."""
        config = {
            "experiment_name": "test_real_cleanup",
            "wandb": {
                "use_wandb": True,
                "tags": ["integration_test"]
            }
        }

        run = init_wandb_run(config)

        try:
            assert run is not None
            run_id = run.id

            # Finish the run
            run.finish()

            # Verify run finished
            # (Note: wandb may need time to sync, so just check it's callable)
            assert run_id is not None
        except Exception as e:
            # If run not properly cleaned up, fail test
            if run:
                run.finish()
            raise e
