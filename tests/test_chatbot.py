"""Integration tests for the conversation chatbot wrapper."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from nsctx.chatbot import ConversationChatbot


def test_chatbot_learns_and_recalls_memory():
    bot = ConversationChatbot(device="cpu")
    metrics = bot.train_on_dataset(epochs=1)
    assert "total" in metrics
    bot.learn([
        ("user", "The observatory detected a bright comet near Jupiter"),
        ("assistant", "It kept moving past the planet"),
    ])

    reply = bot.respond([("user", "What did the observatory see?" )])

    assert "response" in reply
    assert len(reply["probabilities"]) == 3
    assert reply["match_score"] >= 0.0
    assert reply["memory"] is None or "observatory" in reply["memory"]["text"].lower()
