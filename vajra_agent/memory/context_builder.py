"""ContextBuilder assembling enriched prompts from memory, knowledge graph, and plans."""

from __future__ import annotations

from typing import Any

from vajra_agent.context.manager import ProjectContext
from vajra_agent.memory.knowledge.graph import RepositoryKnowledgeGraph
from vajra_agent.memory.retrieval.engine import RetrievalResult
from vajra_agent.prompts.system import build_system_prompt
from vajra_agent.schemas.state import AgentState


class ContextBuilder:
    """Modular builder assembling comprehensive prompts combining memory, knowledge, and state."""

    @classmethod
    def build_enriched_context(
        cls,
        state: AgentState,
        tool_schemas: list[dict[str, Any]],
        project_context: ProjectContext | None = None,
        knowledge_graph: RepositoryKnowledgeGraph | None = None,
        retrieved_memories: list[RetrievalResult] | None = None,
        custom_system_prompt: str | None = None,
    ) -> tuple[str, str]:
        """Returns tuple of (system_prompt, full_history_prompt)."""
        sys_prompt = build_system_prompt(tool_schemas, custom_prompt=custom_system_prompt)

        sections = []

        # 1. Project Context & Repository Information
        if project_context:
            sections.append(
                f"### Project Context\n"
                f"- Framework: {project_context.repo_context.framework}\n"
                f"- Primary Language: {project_context.repo_context.primary_language}\n"
                f"- Active Branch: {project_context.active_branch}\n"
                f"- Root: {project_context.workspace_root}"
            )

        # 2. Knowledge Graph Summary
        if knowledge_graph:
            summary = knowledge_graph.to_summary_dict()
            sections.append(
                f"### Repository Knowledge Graph\n"
                f"- Indexed Files: {summary['files_count']}\n"
                f"- Symbols (Classes/Functions): {summary['symbols_count']}"
            )

        # 3. Retrieved Long-Term Memories
        if retrieved_memories:
            mem_text = "\n".join(
                [f"- [{m.final_score:.2f}] {m.record.text}" for m in retrieved_memories]
            )
            sections.append(f"### Relevant Retrieved Long-Term Memory\n{mem_text}")

        # 4. Current Execution Plan
        if state.plan_steps:
            plan_text = "\n".join(
                [f"Step {i + 1}: {step}" for i, step in enumerate(state.plan_steps)]
            )
            sections.append(f"### Active Task Plan\n{plan_text}")

        # 5. Reflection Notes
        if state.reflection_notes:
            refl_text = "\n".join(state.reflection_notes[-3:])
            sections.append(f"### Recent Reflection Notes\n{refl_text}")

        context_header = "\n\n".join(sections)
        history_str = state.conversation.format_history_string()

        full_prompt = (
            f"{context_header}\n\n### Conversation History\n{history_str}"
            if context_header
            else history_str
        )
        return sys_prompt, full_prompt
