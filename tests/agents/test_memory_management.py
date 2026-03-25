"""Unit tests for agent memory management and persistence."""

import pytest
from typing import Dict, Any, List
from datetime import datetime


class AgentMemory:
    """Agent memory management system."""

    def __init__(self, max_size: int = 1000):
        self.storage: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.max_size = max_size
        self.conversation_history = []

    def store(self, key: str, value: Any) -> bool:
        """Store value in memory."""
        if len(self.storage) >= self.max_size and key not in self.storage:
            return False  # Memory full

        self.storage[key] = value
        self.history.append(
            {"action": "store", "key": key, "timestamp": datetime.now().isoformat()}
        )
        return True

    def retrieve(self, key: str) -> Any:
        """Retrieve value from memory."""
        return self.storage.get(key, None)

    def delete(self, key: str) -> bool:
        """Delete value from memory."""
        if key in self.storage:
            del self.storage[key]
            self.history.append(
                {"action": "delete", "key": key, "timestamp": datetime.now().isoformat()}
            )
            return True
        return False

    def search(self, pattern: str) -> Dict[str, Any]:
        """Search for values by key pattern."""
        results = {}
        for key, value in self.storage.items():
            if pattern in key:
                results[key] = value
        return results

    def clear(self) -> int:
        """Clear all memory."""
        count = len(self.storage)
        self.storage.clear()
        self.history.append(
            {"action": "clear", "items": count, "timestamp": datetime.now().isoformat()}
        )
        return count

    def get_size(self) -> int:
        """Get current memory size."""
        return len(self.storage)

    def add_conversation(self, speaker: str, message: str):
        """Add message to conversation history."""
        self.conversation_history.append(
            {"speaker": speaker, "message": message, "timestamp": datetime.now().isoformat()}
        )

    def get_conversation(self, limit: int = None) -> List[Dict[str, str]]:
        """Get conversation history."""
        if limit:
            return self.conversation_history[-limit:]
        return self.conversation_history.copy()

    def get_history(self) -> List[Dict[str, Any]]:
        """Get memory operation history."""
        return self.history.copy()


class TestMemoryManagement:
    """Test agent memory management."""

    def test_memory_store_retrieve(self):
        """Test storing and retrieving values."""
        memory = AgentMemory()

        memory.store("key1", "value1")
        assert memory.retrieve("key1") == "value1"

    def test_memory_store_multiple(self):
        """Test storing multiple values."""
        memory = AgentMemory()

        memory.store("key1", "value1")
        memory.store("key2", "value2")
        memory.store("key3", {"data": "complex"})

        assert memory.get_size() == 3
        assert memory.retrieve("key1") == "value1"
        assert memory.retrieve("key2") == "value2"
        assert memory.retrieve("key3") == {"data": "complex"}

    def test_memory_delete(self):
        """Test deleting values from memory."""
        memory = AgentMemory()

        memory.store("key1", "value1")
        assert memory.get_size() == 1

        memory.delete("key1")
        assert memory.get_size() == 0
        assert memory.retrieve("key1") is None

    def test_memory_clear(self):
        """Test clearing all memory."""
        memory = AgentMemory()

        memory.store("key1", "value1")
        memory.store("key2", "value2")
        assert memory.get_size() == 2

        count = memory.clear()
        assert count == 2
        assert memory.get_size() == 0

    def test_memory_search(self):
        """Test searching memory by pattern."""
        memory = AgentMemory()

        memory.store("user_name", "Alice")
        memory.store("user_email", "alice@example.com")
        memory.store("agent_id", "12345")

        results = memory.search("user_")
        assert len(results) == 2
        assert "user_name" in results
        assert "user_email" in results

    def test_memory_size_limit(self):
        """Test memory size limit."""
        memory = AgentMemory(max_size=5)

        # Fill memory
        for i in range(5):
            result = memory.store(f"key{i}", f"value{i}")
            assert result is True

        # Try to add one more - should fail
        result = memory.store("key5", "value5")
        assert result is False

    def test_memory_size_limit_replacement(self):
        """Test replacing value within size limit."""
        memory = AgentMemory(max_size=5)

        # Fill memory
        for i in range(5):
            memory.store(f"key{i}", f"value{i}")

        # Replace existing key - should succeed
        result = memory.store("key0", "new_value")
        assert result is True
        assert memory.retrieve("key0") == "new_value"

    def test_conversation_history(self):
        """Test conversation history tracking."""
        memory = AgentMemory()

        memory.add_conversation("user", "Hello, how are you?")
        memory.add_conversation("agent", "I'm doing well, thank you!")
        memory.add_conversation("user", "Can you help me?")

        history = memory.get_conversation()
        assert len(history) == 3
        assert history[0]["speaker"] == "user"
        assert history[1]["speaker"] == "agent"

    def test_conversation_limit(self):
        """Test limiting conversation history."""
        memory = AgentMemory()

        for i in range(10):
            memory.add_conversation("user", f"Message {i}")

        # Get last 3
        history = memory.get_conversation(limit=3)
        assert len(history) == 3
        assert "Message 9" in history[-1]["message"]

    def test_memory_operation_history(self):
        """Test tracking memory operations."""
        memory = AgentMemory()

        memory.store("key1", "value1")
        memory.store("key2", "value2")
        memory.delete("key1")

        history = memory.get_history()
        assert len(history) >= 3

        actions = [h["action"] for h in history]
        assert "store" in actions
        assert "delete" in actions

    def test_memory_isolation(self):
        """Test memory isolation between instances."""
        memory1 = AgentMemory()
        memory2 = AgentMemory()

        memory1.store("key", "value1")
        memory2.store("key", "value2")

        assert memory1.retrieve("key") == "value1"
        assert memory2.retrieve("key") == "value2"

    def test_complex_value_storage(self):
        """Test storing complex values."""
        memory = AgentMemory()

        complex_value = {
            "user": {"name": "Alice", "id": 123},
            "tasks": [1, 2, 3],
            "metadata": {"created": "2026-03-25"},
        }

        memory.store("complex", complex_value)
        retrieved = memory.retrieve("complex")

        assert retrieved["user"]["name"] == "Alice"
        assert retrieved["tasks"] == [1, 2, 3]

    def test_memory_consistency(self):
        """Test memory consistency after operations."""
        memory = AgentMemory()

        # Store, modify pattern, verify consistency
        memory.store("prefix_1", "value1")
        memory.store("prefix_2", "value2")
        memory.store("other", "value3")

        # Search for prefix
        results = memory.search("prefix_")
        assert len(results) == 2

        # Delete one
        memory.delete("prefix_1")
        results = memory.search("prefix_")
        assert len(results) == 1

        # Verify consistency
        assert memory.retrieve("prefix_1") is None
        assert memory.retrieve("prefix_2") == "value2"

    def test_nonexistent_key_retrieval(self):
        """Test retrieving nonexistent keys."""
        memory = AgentMemory()

        result = memory.retrieve("nonexistent")
        assert result is None

    def test_nonexistent_key_deletion(self):
        """Test deleting nonexistent keys."""
        memory = AgentMemory()

        result = memory.delete("nonexistent")
        assert result is False

    def test_memory_timestamp_tracking(self):
        """Test that operations are timestamped."""
        memory = AgentMemory()

        memory.store("key1", "value1")

        history = memory.get_history()
        assert len(history) > 0
        assert "timestamp" in history[0]

    def test_conversation_timestamp(self):
        """Test conversation message timestamps."""
        memory = AgentMemory()

        memory.add_conversation("user", "Hello")
        conversation = memory.get_conversation()

        assert len(conversation) == 1
        assert "timestamp" in conversation[0]
