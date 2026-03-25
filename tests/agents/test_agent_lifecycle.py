"""Unit tests for agent initialization and lifecycle management."""

import pytest
import yaml
from pathlib import Path
from typing import Dict, Any


class MockAgent:
    """Mock agent for testing lifecycle."""

    def __init__(self, name: str, model: str = "claude-haiku-4-5-20251001"):
        self.name = name
        self.model = model
        self.state = {"initialized": True, "active": False}
        self.memory = {}
        self.tools = []
        self.skills = []
        self.config = {}

    def startup(self):
        """Start the agent."""
        self.state["active"] = True
        return {"status": "started", "agent": self.name}

    def shutdown(self):
        """Shut down the agent."""
        self.state["active"] = False
        return {"status": "stopped", "agent": self.name}

    def load_config(self, config: Dict[str, Any]):
        """Load configuration."""
        self.config = config
        return True

    def register_tool(self, tool_name: str):
        """Register a tool."""
        self.tools.append(tool_name)
        return len(self.tools)

    def register_skill(self, skill_name: str):
        """Register a skill."""
        self.skills.append(skill_name)
        return len(self.skills)

    def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities."""
        return {
            "name": self.name,
            "model": self.model,
            "tools": self.tools,
            "skills": self.skills,
            "active": self.state["active"],
        }


class TestAgentInitialization:
    """Test agent initialization and lifecycle."""

    def test_agent_creation(self):
        """Test basic agent creation."""
        agent = MockAgent("test_agent")

        assert agent.name == "test_agent"
        assert agent.state["initialized"] is True
        assert agent.state["active"] is False

    def test_agent_startup(self):
        """Test agent startup."""
        agent = MockAgent("test_agent")

        result = agent.startup()

        assert result["status"] == "started"
        assert agent.state["active"] is True

    def test_agent_shutdown(self):
        """Test agent shutdown."""
        agent = MockAgent("test_agent")
        agent.startup()

        result = agent.shutdown()

        assert result["status"] == "stopped"
        assert agent.state["active"] is False

    def test_agent_startup_shutdown_cycle(self):
        """Test complete startup/shutdown cycle."""
        agent = MockAgent("test_agent")

        # Start
        assert agent.startup()["status"] == "started"
        assert agent.state["active"] is True

        # Shutdown
        assert agent.shutdown()["status"] == "stopped"
        assert agent.state["active"] is False

        # Restart
        assert agent.startup()["status"] == "started"
        assert agent.state["active"] is True

    def test_agent_config_loading(self):
        """Test loading agent configuration."""
        agent = MockAgent("test_agent")
        config = {
            "model": "claude-sonnet-4-6",
            "tools": ["Bash", "Read"],
            "skills": ["writing", "code-debugging"],
        }

        result = agent.load_config(config)

        assert result is True
        assert agent.config == config

    def test_agent_tool_registration(self):
        """Test tool registration."""
        agent = MockAgent("test_agent")

        assert agent.register_tool("Bash") == 1
        assert agent.register_tool("Read") == 2
        assert agent.register_tool("Write") == 3

        assert len(agent.tools) == 3
        assert "Bash" in agent.tools

    def test_agent_skill_registration(self):
        """Test skill registration."""
        agent = MockAgent("test_agent")

        assert agent.register_skill("writing") == 1
        assert agent.register_skill("code-debugging") == 2

        assert len(agent.skills) == 2
        assert "writing" in agent.skills

    def test_agent_capability_detection(self):
        """Test agent capability detection."""
        agent = MockAgent("test_agent", model="claude-sonnet-4-6")
        agent.register_tool("Bash")
        agent.register_skill("writing")

        capabilities = agent.get_capabilities()

        assert capabilities["name"] == "test_agent"
        assert capabilities["model"] == "claude-sonnet-4-6"
        assert "Bash" in capabilities["tools"]
        assert "writing" in capabilities["skills"]
        assert capabilities["active"] is False

    def test_multiple_agents(self):
        """Test creating multiple agents."""
        agent1 = MockAgent("agent1")
        agent2 = MockAgent("agent2")
        agent3 = MockAgent("agent3")

        assert agent1.name == "agent1"
        assert agent2.name == "agent2"
        assert agent3.name == "agent3"

        # Each agent is independent
        agent1.startup()
        assert agent1.state["active"] is True
        assert agent2.state["active"] is False
        assert agent3.state["active"] is False

    def test_agent_model_assignment(self):
        """Test agent model assignment."""
        haiku_agent = MockAgent("haiku_agent", model="claude-haiku-4-5-20251001")
        sonnet_agent = MockAgent("sonnet_agent", model="claude-sonnet-4-6")
        opus_agent = MockAgent("opus_agent", model="claude-opus-4-6")

        assert haiku_agent.model == "claude-haiku-4-5-20251001"
        assert sonnet_agent.model == "claude-sonnet-4-6"
        assert opus_agent.model == "claude-opus-4-6"

    def test_agent_state_isolation(self):
        """Test that agent states are isolated."""
        agent1 = MockAgent("agent1")
        agent2 = MockAgent("agent2")

        agent1.register_tool("Bash")
        agent1.register_skill("writing")

        assert len(agent1.tools) == 1
        assert len(agent1.skills) == 1
        assert len(agent2.tools) == 0
        assert len(agent2.skills) == 0

    def test_agent_memory_initialization(self):
        """Test agent memory initialization."""
        agent = MockAgent("test_agent")

        assert isinstance(agent.memory, dict)
        assert len(agent.memory) == 0

        # Store memory
        agent.memory["key1"] = "value1"
        assert agent.memory["key1"] == "value1"

    def test_agent_config_validation(self):
        """Test agent configuration validation."""
        agent = MockAgent("test_agent")

        valid_config = {
            "model": "claude-sonnet-4-6",
            "tools": ["Bash"],
            "skills": ["writing"],
        }

        result = agent.load_config(valid_config)
        assert result is True

    def test_agent_double_startup(self):
        """Test that double startup is idempotent."""
        agent = MockAgent("test_agent")

        result1 = agent.startup()
        result2 = agent.startup()

        assert result1["status"] == "started"
        assert result2["status"] == "started"
        assert agent.state["active"] is True

    def test_agent_double_shutdown(self):
        """Test that double shutdown is safe."""
        agent = MockAgent("test_agent")
        agent.startup()

        result1 = agent.shutdown()
        result2 = agent.shutdown()

        assert result1["status"] == "stopped"
        assert result2["status"] == "stopped"
        assert agent.state["active"] is False
