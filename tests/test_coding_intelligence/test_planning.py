"""Tests for PlanningEngine and structured Plan models."""

from vajra_agent.planning import Plan, PlanningEngine, PlanStep, StepStatus


def test_plan_step_lifecycle():
    step = PlanStep(description="Test step")
    assert step.status == StepStatus.PENDING

    step.mark_completed({"output": "ok"})
    assert step.status == StepStatus.COMPLETED
    assert step.result == {"output": "ok"}


def test_plan_next_step_dependency_resolution():
    step1 = PlanStep(description="Step 1")
    step2 = PlanStep(description="Step 2", dependencies=[step1.step_id])

    plan = Plan(objective="Test", steps=[step1, step2])

    # Next step should be step1
    next_step = plan.get_next_pending_step()
    assert next_step == step1

    # Complete step1
    step1.mark_completed("Done")

    # Next step should now be step2
    next_step2 = plan.get_next_pending_step()
    assert next_step2 == step2


def test_planning_engine_simple_plan():
    engine = PlanningEngine()
    plan = engine.create_simple_plan("Analyze repo", ["Step A", "Step B"])

    assert len(plan.steps) == 2
    assert plan.steps[0].description == "Step A"
    assert plan.steps[1].dependencies == [plan.steps[0].step_id]
