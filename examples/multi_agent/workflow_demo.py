"""SaaS build workflow demonstration."""

from vajra_agent import SaaSBuildWorkflow


def main():
    wf = SaaSBuildWorkflow()
    response = wf.execute("Build Stripe subscription billing integration with webhooks")

    print("SaaS Workflow Execution Summary:")
    print(response.output)


if __name__ == "__main__":
    main()
