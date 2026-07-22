"""Security audit review workflow example."""

from vajra_agent import SecurityAuditWorkflow


def main():
    wf = SecurityAuditWorkflow()
    response = wf.execute("api/routes")

    print("Security Audit Output:")
    print(response.output)


if __name__ == "__main__":
    main()
