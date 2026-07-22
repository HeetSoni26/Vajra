"""PlanningEngine task decomposition example."""

from vajra_agent import PlanningEngine


def main():
    planner = PlanningEngine()
    plan = planner.create_simple_plan(
        objective="Implement JWT Authentication",
        steps_desc=[
            "Scan repository for auth modules",
            "Generate auth/jwt.py token handler",
            "Add login route to api/main.py",
            "Run pytest to verify authentication tests",
        ],
    )

    print(f"Plan ID: {plan.plan_id}")
    print(f"Objective: {plan.objective}")
    print("Steps:")
    for step in plan.steps:
        print(f"  - [{step.status.value}] {step.description} (Deps: {step.dependencies})")


if __name__ == "__main__":
    main()
