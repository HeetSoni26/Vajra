"""MemorySummarizer generating concise conversation and session summaries."""

from __future__ import annotations

from vajra_agent.reasoners.base import BaseReasoner
from vajra_agent.schemas.messages import Conversation, MessageRole


class MemorySummarizer:
    """Summarizes conversation history when exceeding message thresholds."""

    def __init__(self, reasoner: BaseReasoner | None = None) -> None:
        self.reasoner = reasoner

    def summarize_conversation(self, conversation: Conversation, max_messages: int = 10) -> str:
        if len(conversation.messages) <= max_messages:
            return conversation.format_history_string()

        # Split system messages from history
        sys_msgs = [m for m in conversation.messages if m.role == MessageRole.SYSTEM]
        user_assistant_msgs = [m for m in conversation.messages if m.role != MessageRole.SYSTEM]

        history_str = "\n".join([f"{m.role}: {m.content}" for m in user_assistant_msgs[:-4]])

        if self.reasoner:
            prompt = (
                f"Summarize the following conversation history concise preserving key architectural decisions, "
                f"completed tasks, and findings:\n\n{history_str}"
            )
            summary_text = self.reasoner.generate(prompt=prompt)
        else:
            summary_text = f"Summary of past {len(user_assistant_msgs[:-4])} messages: {history_str[:200]}..."

        # Reconstruct pruned conversation
        new_conv = Conversation()
        for sm in sys_msgs:
            new_conv.add_message(sm)

        new_conv.add_system(f"Past Conversation Summary:\n{summary_text}")

        for recent_m in user_assistant_msgs[-4:]:
            new_conv.add_message(recent_m)

        return new_conv.format_history_string()
