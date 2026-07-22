"""Memory summarization example."""

from vajra_agent import Conversation, MemorySummarizer


def main():
    conv = Conversation()
    for i in range(12):
        conv.add_user(f"Message {i}: What is step {i}?")
        conv.add_assistant(f"Response {i}: Step {i} involves configuration.")

    summarizer = MemorySummarizer()
    summary_prompt = summarizer.summarize_conversation(conv, max_messages=4)

    print("Summarized Conversation Prompt:")
    print(summary_prompt)


if __name__ == "__main__":
    main()
