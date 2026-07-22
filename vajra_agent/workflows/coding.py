"""Specialized coding workflows: BugFix, Feature, Refactor, TestGen, RepoAnalysis."""

from __future__ import annotations

from typing import Any

from vajra_agent.schemas.results import AgentResponse
from vajra_agent.workflows.base import BaseWorkflow


class RepoAnalysisWorkflow(BaseWorkflow):
    """Workflow for analyzing repository structure, framework, and entrypoints."""

    @property
    def name(self) -> str:
        return "repo_analysis"

    def execute(self, directory: str = ".", **kwargs: Any) -> AgentResponse:
        prompt = f"Analyze the repository structure in '{directory}', identify primary language, framework, config files, and summarize the architecture."
        return self.agent.run(prompt)


class BugFixWorkflow(BaseWorkflow):
    """Workflow for reproducing, diagnosing, fixing, and verifying bug reports."""

    @property
    def name(self) -> str:
        return "bug_fix"

    def execute(self, bug_description: str, file_target: str | None = None, **kwargs: Any) -> AgentResponse:
        prompt = f"Diagnose and fix the following bug:\n{bug_description}\nTarget file: {file_target or 'entire repository'}. Verify fix with tests."
        return self.agent.run(prompt)


class FeatureWorkflow(BaseWorkflow):
    """Workflow for implementing new software features end-to-end."""

    @property
    def name(self) -> str:
        return "feature_implementation"

    def execute(self, feature_description: str, **kwargs: Any) -> AgentResponse:
        prompt = f"Plan and implement the following new feature:\n{feature_description}\nEnsure unit tests and linter checks pass."
        return self.agent.run(prompt)


class RefactorWorkflow(BaseWorkflow):
    """Workflow for refactoring code modules while maintaining full test pass rates."""

    @property
    def name(self) -> str:
        return "code_refactor"

    def execute(self, refactor_goal: str, target_file: str, **kwargs: Any) -> AgentResponse:
        prompt = f"Refactor code in '{target_file}' according to: {refactor_goal}. Ensure backward compatibility and pass all tests."
        return self.agent.run(prompt)


class TestGenerationWorkflow(BaseWorkflow):
    """Workflow for generating comprehensive unit test suites."""

    __test__ = False

    @property
    def name(self) -> str:
        return "test_generation"

    def execute(self, source_file: str, test_file: str | None = None, **kwargs: Any) -> AgentResponse:
        prompt = f"Generate pytest unit test suite for source file '{source_file}' in '{test_file or 'tests/'}'."
        return self.agent.run(prompt)
