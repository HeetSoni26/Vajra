"""Permission approval manager demonstration."""

from vajra_agent import ActionCategory, PermissionManager, PermissionPolicy


def main():
    pm = PermissionManager(default_policy=PermissionPolicy.ALWAYS_ALLOW)
    pm.set_policy(ActionCategory.FILE_DELETE, PermissionPolicy.ALWAYS_DENY)

    print("Permission Check Results:")
    print(f"  File Read: {pm.check_permission(ActionCategory.FILE_READ)}")
    print(f"  File Delete: {pm.check_permission(ActionCategory.FILE_DELETE)}")


if __name__ == "__main__":
    main()
