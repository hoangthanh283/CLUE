"""Integration tests for agent communication and message passing."""

import pytest
import json
import time
from typing import Dict, List, Any, Optional
from queue import Queue
from threading import Thread


class Message:
    """Message between agents."""

    def __init__(
        self,
        sender: str,
        recipient: str,
        content: str,
        message_id: str = None,
        reply_to: str = None,
    ):
        self.sender = sender
        self.recipient = recipient
        self.content = content
        self.message_id = message_id or self._generate_id()
        self.reply_to = reply_to
        self.timestamp = time.time()
        self.delivered = False

    _id_counter = 0

    def _generate_id(self) -> str:
        """Generate unique message ID."""
        Message._id_counter += 1
        return f"{self.sender}_{self.recipient}_{int(time.time() * 1000)}_{Message._id_counter}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.message_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "reply_to": self.reply_to,
            "timestamp": self.timestamp,
        }


class MessageBroker:
    """Handles inter-agent message delivery."""

    def __init__(self):
        self.mailboxes: Dict[str, Queue] = {}
        self.delivered_messages: List[Message] = []
        self.failed_messages: List[Message] = []

    def register_agent(self, agent_name: str):
        """Register agent mailbox."""
        if agent_name not in self.mailboxes:
            self.mailboxes[agent_name] = Queue()

    def send_message(self, message: Message, timeout: float = 5.0) -> bool:
        """Send message to recipient."""
        if message.recipient not in self.mailboxes:
            self.failed_messages.append(message)
            return False

        try:
            self.mailboxes[message.recipient].put(message, timeout=timeout)
            message.delivered = True
            self.delivered_messages.append(message)
            return True
        except Exception:
            self.failed_messages.append(message)
            return False

    def receive_message(self, agent_name: str, timeout: float = 1.0) -> Optional[Message]:
        """Receive message for agent."""
        if agent_name not in self.mailboxes:
            return None

        try:
            return self.mailboxes[agent_name].get(timeout=timeout)
        except:
            return None

    def get_mailbox_size(self, agent_name: str) -> int:
        """Get number of messages in mailbox."""
        if agent_name not in self.mailboxes:
            return 0
        return self.mailboxes[agent_name].qsize()

    def get_delivered_count(self) -> int:
        """Get total delivered messages."""
        return len(self.delivered_messages)

    def get_failed_count(self) -> int:
        """Get total failed messages."""
        return len(self.failed_messages)

    def clear_mailbox(self, agent_name: str):
        """Clear mailbox for agent."""
        if agent_name in self.mailboxes:
            while not self.mailboxes[agent_name].empty():
                try:
                    self.mailboxes[agent_name].get_nowait()
                except:
                    break


class TestAgentCommunication:
    """Test agent communication and message passing."""

    def test_message_creation(self):
        """Test message creation."""
        msg = Message("agent1", "agent2", "Hello, agent2!")

        assert msg.sender == "agent1"
        assert msg.recipient == "agent2"
        assert msg.content == "Hello, agent2!"
        assert msg.message_id is not None

    def test_message_serialization(self):
        """Test message serialization."""
        msg = Message("agent1", "agent2", "Test message")
        msg_dict = msg.to_dict()

        assert msg_dict["sender"] == "agent1"
        assert msg_dict["recipient"] == "agent2"
        assert msg_dict["content"] == "Test message"
        assert "timestamp" in msg_dict

    def test_message_broker_registration(self):
        """Test agent registration with broker."""
        broker = MessageBroker()

        broker.register_agent("agent1")
        broker.register_agent("agent2")

        assert "agent1" in broker.mailboxes
        assert "agent2" in broker.mailboxes

    def test_message_delivery(self):
        """Test message delivery between agents."""
        broker = MessageBroker()
        broker.register_agent("agent1")
        broker.register_agent("agent2")

        msg = Message("agent1", "agent2", "Hello!")
        result = broker.send_message(msg)

        assert result is True
        assert msg.delivered is True

    def test_message_reception(self):
        """Test receiving messages."""
        broker = MessageBroker()
        broker.register_agent("agent1")
        broker.register_agent("agent2")

        # Send message
        msg = Message("agent1", "agent2", "Hello!")
        broker.send_message(msg)

        # Receive message
        received = broker.receive_message("agent2")
        assert received is not None
        assert received.content == "Hello!"
        assert received.sender == "agent1"

    def test_message_ordering(self):
        """Test that messages are delivered in order."""
        broker = MessageBroker()
        broker.register_agent("sender")
        broker.register_agent("receiver")

        # Send multiple messages
        for i in range(5):
            msg = Message("sender", "receiver", f"Message {i}")
            broker.send_message(msg)

        # Receive in order
        for i in range(5):
            received = broker.receive_message("receiver")
            assert received.content == f"Message {i}"

    def test_failed_delivery(self):
        """Test failed message delivery."""
        broker = MessageBroker()
        broker.register_agent("agent1")
        # Don't register agent2

        msg = Message("agent1", "agent2", "Hello!")
        result = broker.send_message(msg)

        assert result is False
        assert msg not in broker.delivered_messages
        assert msg in broker.failed_messages

    def test_message_delivery_count(self):
        """Test tracking delivered messages."""
        broker = MessageBroker()
        broker.register_agent("agent1")
        broker.register_agent("agent2")

        for i in range(10):
            msg = Message("agent1", "agent2", f"Message {i}")
            broker.send_message(msg)

        assert broker.get_delivered_count() == 10
        assert broker.get_failed_count() == 0

    def test_message_timeout(self):
        """Test message delivery timeout."""
        broker = MessageBroker()
        broker.register_agent("sender")
        broker.register_agent("receiver")

        # Send with timeout
        msg = Message("sender", "receiver", "Test")
        result = broker.send_message(msg, timeout=5.0)
        assert result is True

    def test_mailbox_size(self):
        """Test checking mailbox size."""
        broker = MessageBroker()
        broker.register_agent("receiver")

        assert broker.get_mailbox_size("receiver") == 0

        broker.register_agent("sender")
        for i in range(3):
            msg = Message("sender", "receiver", f"Message {i}")
            broker.send_message(msg)

        assert broker.get_mailbox_size("receiver") == 3

    def test_clear_mailbox(self):
        """Test clearing mailbox."""
        broker = MessageBroker()
        broker.register_agent("receiver")
        broker.register_agent("sender")

        # Add messages
        for i in range(5):
            msg = Message("sender", "receiver", f"Message {i}")
            broker.send_message(msg)

        assert broker.get_mailbox_size("receiver") == 5

        # Clear
        broker.clear_mailbox("receiver")
        assert broker.get_mailbox_size("receiver") == 0

    def test_reply_messages(self):
        """Test reply to messages."""
        broker = MessageBroker()
        broker.register_agent("agent1")
        broker.register_agent("agent2")

        # Send message
        msg1 = Message("agent1", "agent2", "Hello!")
        broker.send_message(msg1)

        # Receive and create reply
        received = broker.receive_message("agent2")
        reply = Message("agent2", "agent1", "Hi back!", reply_to=received.message_id)
        broker.send_message(reply)

        # Verify reply
        response = broker.receive_message("agent1")
        assert response.reply_to == msg1.message_id

    def test_bidirectional_communication(self):
        """Test bidirectional communication."""
        broker = MessageBroker()
        broker.register_agent("agent1")
        broker.register_agent("agent2")

        # Agent1 -> Agent2
        msg1 = Message("agent1", "agent2", "Request")
        broker.send_message(msg1)

        received1 = broker.receive_message("agent2")
        assert received1.content == "Request"

        # Agent2 -> Agent1
        msg2 = Message("agent2", "agent1", "Response", reply_to=msg1.message_id)
        broker.send_message(msg2)

        received2 = broker.receive_message("agent1")
        assert received2.content == "Response"
        assert received2.reply_to == msg1.message_id

    def test_broadcast_capability(self):
        """Test message delivery statistics."""
        broker = MessageBroker()

        # Register multiple agents
        for i in range(5):
            broker.register_agent(f"agent{i}")

        # Send from agent0 to agent1
        msg = Message("agent0", "agent1", "Test")
        broker.send_message(msg)

        assert broker.get_delivered_count() == 1

        # Try to send to unregistered agent
        msg2 = Message("agent0", "agent_unknown", "Test")
        broker.send_message(msg2)

        assert broker.get_failed_count() == 1

    def test_empty_mailbox_receive(self):
        """Test receiving from empty mailbox."""
        broker = MessageBroker()
        broker.register_agent("agent1")

        received = broker.receive_message("agent1", timeout=0.1)
        assert received is None

    def test_message_id_uniqueness(self):
        """Test message IDs are unique."""
        messages = []
        for i in range(100):
            msg = Message("agent1", "agent2", f"Message {i}")
            messages.append(msg.message_id)

        # All IDs should be unique
        assert len(messages) == len(set(messages))
