"""Tests for PermissionManager approval policy evaluation."""

from vajra_agent import ActionCategory, PermissionManager, PermissionPolicy


def test_permission_manager():
    pm = PermissionManager(default_policy=PermissionPolicy.ALWAYS_ALLOW)
    assert pm.check_permission(ActionCategory.FILE_READ) is True

    pm.set_policy(ActionCategory.FILE_DELETE, PermissionPolicy.ALWAYS_DENY)
    assert pm.check_permission(ActionCategory.FILE_DELETE) is False
