"""Human Approval System managing permission policies for sensitive tool operations."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Any


class ActionCategory(str, Enum):
    """Categorized actions requiring permission policy evaluation."""

    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    FILE_DELETE = "file_delete"
    SHELL_EXECUTION = "shell_execution"
    GIT_PUSH = "git_push"
    NETWORK_ACCESS = "network_access"
    RECURSIVE_EDIT = "recursive_edit"


class PermissionPolicy(str, Enum):
    """Permission evaluation policies."""

    ALWAYS_ALLOW = "always_allow"
    ALWAYS_DENY = "always_deny"
    ASK_EVERY_TIME = "ask_every_time"


class PermissionManager:
    """Manages action permissions and human interactive approvals."""

    def __init__(
        self,
        default_policy: PermissionPolicy = PermissionPolicy.ALWAYS_ALLOW,
        policy_overrides: dict[ActionCategory, PermissionPolicy] | None = None,
        approval_handler: Callable[[ActionCategory, dict[str, Any]], bool] | None = None,
    ) -> None:
        self.default_policy = default_policy
        self.policy_overrides = policy_overrides or {}
        self.approval_handler = approval_handler

    def set_policy(self, category: ActionCategory, policy: PermissionPolicy) -> None:
        """Override policy for a specific action category."""
        self.policy_overrides[category] = policy

    def check_permission(self, category: ActionCategory, details: dict[str, Any] | None = None) -> bool:
        """Check if an action is permitted under configured policy."""
        policy = self.policy_overrides.get(category, self.default_policy)

        if policy == PermissionPolicy.ALWAYS_ALLOW:
            return True
        elif policy == PermissionPolicy.ALWAYS_DENY:
            return False
        elif policy == PermissionPolicy.ASK_EVERY_TIME:
            if self.approval_handler:
                return self.approval_handler(category, details or {})
            return True  # Fallback to true if no interactive handler registered
        return True
