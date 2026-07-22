"""Tests for MemorySummarizer conversation pruning and summary generation."""

from vajra_agent.memory.summaries import MemorySummarizer
from vajra_agent.schemas import Conversation


def test_memory_summarizer():
    conv = Conversation()
    for i in range(15):
        conv.add_user(f"User message {i}")
        conv.add_assistant(f"Assistant response {i}")

    summarizer = MemorySummarizer()
    sum_str = summarizer.summarize_conversation(conv, max_messages=6)

    assert "Past Conversation Summary" in sum_str
    assert "User message 14" in sum_str
