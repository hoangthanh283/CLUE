"""Unit tests for agent model switching and fallback mechanisms."""

import pytest
from typing import List, Dict, Any


class ModelRegistry:
    """Registry for available models with fallback chains."""

    AVAILABLE_MODELS = {
        "claude-opus-4-6": {"available": True, "tier": "premium", "cost": 3},
        "claude-sonnet-4-6": {"available": True, "tier": "standard", "cost": 1},
        "claude-haiku-4-5-20251001": {"available": True, "tier": "fast", "cost": 0.5},
    }

    FALLBACK_CHAIN = {
        "claude-opus-4-6": ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        "claude-sonnet-4-6": ["claude-haiku-4-5-20251001"],
        "claude-haiku-4-5-20251001": [],
    }

    @classmethod
    def is_available(cls, model: str) -> bool:
        """Check if model is available."""
        return model in cls.AVAILABLE_MODELS and cls.AVAILABLE_MODELS[model]["available"]

    @classmethod
    def get_fallback(cls, model: str) -> str:
        """Get fallback model for given model."""
        if cls.is_available(model):
            return model
        fallbacks = cls.FALLBACK_CHAIN.get(model, [])
        for fallback in fallbacks:
            if cls.is_available(fallback):
                return fallback
        return None

    @classmethod
    def get_cost(cls, model: str) -> float:
        """Get cost of model."""
        if model in cls.AVAILABLE_MODELS:
            return cls.AVAILABLE_MODELS[model]["cost"]
        return None

    @classmethod
    def set_availability(cls, model: str, available: bool):
        """Set model availability (for testing)."""
        if model in cls.AVAILABLE_MODELS:
            cls.AVAILABLE_MODELS[model]["available"] = available


class AgentWithModelSwitching:
    """Agent with model switching and fallback."""

    def __init__(self, name: str, preferred_model: str = "claude-opus-4-6"):
        self.name = name
        self.preferred_model = preferred_model
        self.current_model = None
        self.model_history = []
        self._initialize_model()

    def _initialize_model(self):
        """Initialize model with fallback."""
        self.current_model = ModelRegistry.get_fallback(self.preferred_model)
        self.model_history.append(self.current_model)

    def switch_model(self, new_model: str) -> bool:
        """Switch to new model with fallback."""
        target_model = ModelRegistry.get_fallback(new_model)
        if target_model:
            self.current_model = target_model
            self.model_history.append(target_model)
            return True
        return False

    def get_current_model(self) -> str:
        """Get current active model."""
        return self.current_model

    def get_model_cost(self) -> float:
        """Get cost of current model."""
        return ModelRegistry.get_cost(self.current_model)

    def use_cheapest_model(self) -> str:
        """Switch to cheapest available model."""
        cheapest = "claude-haiku-4-5-20251001"
        self.switch_model(cheapest)
        return self.current_model

    def use_best_model(self) -> str:
        """Switch to best available model."""
        best = "claude-opus-4-6"
        self.switch_model(best)
        return self.current_model

    def get_model_history(self) -> List[str]:
        """Get history of models used."""
        return self.model_history.copy()


class TestModelSwitching:
    """Test model switching and fallback mechanisms."""

    def test_model_initialization_with_preferred(self):
        """Test agent initializes with preferred model."""
        agent = AgentWithModelSwitching("test_agent", "claude-opus-4-6")

        assert agent.current_model == "claude-opus-4-6"
        assert agent.preferred_model == "claude-opus-4-6"

    def test_model_initialization_fallback(self):
        """Test model initialization falls back when preferred unavailable."""
        # Make opus unavailable
        ModelRegistry.set_availability("claude-opus-4-6", False)

        agent = AgentWithModelSwitching("test_agent", "claude-opus-4-6")

        # Should fall back to sonnet
        assert agent.current_model == "claude-sonnet-4-6"

        # Restore
        ModelRegistry.set_availability("claude-opus-4-6", True)

    def test_model_switching(self):
        """Test switching between models."""
        agent = AgentWithModelSwitching("test_agent")
        agent.switch_model("claude-sonnet-4-6")

        assert agent.current_model == "claude-sonnet-4-6"

    def test_model_switching_with_fallback(self):
        """Test model switching with fallback."""
        agent = AgentWithModelSwitching("test_agent")

        # Make sonnet unavailable
        ModelRegistry.set_availability("claude-sonnet-4-6", False)

        # Try to switch to sonnet - should fall back
        agent.switch_model("claude-sonnet-4-6")
        assert agent.current_model == "claude-haiku-4-5-20251001"

        # Restore
        ModelRegistry.set_availability("claude-sonnet-4-6", True)

    def test_model_cost_calculation(self):
        """Test model cost calculation."""
        agent = AgentWithModelSwitching("test_agent", "claude-opus-4-6")

        cost_opus = agent.get_model_cost()
        assert cost_opus == 3

        agent.switch_model("claude-sonnet-4-6")
        cost_sonnet = agent.get_model_cost()
        assert cost_sonnet == 1

        agent.switch_model("claude-haiku-4-5-20251001")
        cost_haiku = agent.get_model_cost()
        assert cost_haiku == 0.5

    def test_use_cheapest_model(self):
        """Test switching to cheapest model."""
        agent = AgentWithModelSwitching("test_agent", "claude-opus-4-6")

        agent.use_cheapest_model()
        assert agent.current_model == "claude-haiku-4-5-20251001"
        assert agent.get_model_cost() == 0.5

    def test_use_best_model(self):
        """Test switching to best model."""
        agent = AgentWithModelSwitching("test_agent", "claude-haiku-4-5-20251001")

        agent.use_best_model()
        assert agent.current_model == "claude-opus-4-6"
        assert agent.get_model_cost() == 3

    def test_model_history_tracking(self):
        """Test tracking of model switching history."""
        agent = AgentWithModelSwitching("test_agent", "claude-opus-4-6")

        history = agent.get_model_history()
        assert history[0] == "claude-opus-4-6"

        agent.switch_model("claude-sonnet-4-6")
        agent.switch_model("claude-haiku-4-5-20251001")

        history = agent.get_model_history()
        assert len(history) == 3
        assert history == [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5-20251001",
        ]

    def test_graceful_degradation(self):
        """Test graceful degradation when models unavailable."""
        # Make all models unavailable
        ModelRegistry.set_availability("claude-opus-4-6", False)
        ModelRegistry.set_availability("claude-sonnet-4-6", False)
        ModelRegistry.set_availability("claude-haiku-4-5-20251001", False)

        agent = AgentWithModelSwitching("test_agent")
        # Should result in None (no available model)
        assert agent.current_model is None

        # Restore
        ModelRegistry.set_availability("claude-opus-4-6", True)
        ModelRegistry.set_availability("claude-sonnet-4-6", True)
        ModelRegistry.set_availability("claude-haiku-4-5-20251001", True)

    def test_fallback_chain_complete(self):
        """Test complete fallback chain."""
        # Disable opus
        ModelRegistry.set_availability("claude-opus-4-6", False)

        agent = AgentWithModelSwitching("test_agent", "claude-opus-4-6")
        assert agent.current_model == "claude-sonnet-4-6"

        # Disable sonnet too
        ModelRegistry.set_availability("claude-sonnet-4-6", False)

        # Switch should fall back further
        agent.switch_model("claude-sonnet-4-6")
        assert agent.current_model == "claude-haiku-4-5-20251001"

        # Restore
        ModelRegistry.set_availability("claude-opus-4-6", True)
        ModelRegistry.set_availability("claude-sonnet-4-6", True)

    def test_multiple_agents_independent_models(self):
        """Test that multiple agents maintain independent model states."""
        agent1 = AgentWithModelSwitching("agent1", "claude-opus-4-6")
        agent2 = AgentWithModelSwitching("agent2", "claude-haiku-4-5-20251001")

        assert agent1.current_model == "claude-opus-4-6"
        assert agent2.current_model == "claude-haiku-4-5-20251001"

        agent1.use_cheapest_model()
        assert agent1.current_model == "claude-haiku-4-5-20251001"
        assert agent2.current_model == "claude-haiku-4-5-20251001"

        agent2.use_best_model()
        assert agent1.current_model == "claude-haiku-4-5-20251001"
        assert agent2.current_model == "claude-opus-4-6"

    def test_model_switch_idempotent(self):
        """Test that switching to same model multiple times is safe."""
        agent = AgentWithModelSwitching("test_agent")

        agent.switch_model("claude-sonnet-4-6")
        agent.switch_model("claude-sonnet-4-6")
        agent.switch_model("claude-sonnet-4-6")

        assert agent.current_model == "claude-sonnet-4-6"
        history = agent.get_model_history()
        # Should have 4 entries (initial + 3 switches)
        assert len(history) == 4
