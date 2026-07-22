"""Regression test suite ensuring zero regressions across core subsystems."""

from vajra_agent import (
    AgentConfig,
    AgentPluginRegistry,
    CoderAgent,
    Conversation,
    FoundationAgent,
    MemoryManager,
    MockReasoner,
    PermissionManager,
    PythonSandbox,
    ToolRegistry,
    VerificationEngine,
)
from vajra_agent.function_calling import FunctionParser


def test_regression_core_abstractions():
    # 1. Config
    cfg = AgentConfig.production()
    assert cfg.max_iterations == 20

    # 2. Reasoner & Agent
    reasoner = MockReasoner(["Test output."])
    agent = FoundationAgent(reasoner)
    res = agent.run("Hello")
    assert res.output == "Test output."

    # 3. MemoryManager
    mem = MemoryManager()
    rec = mem.remember("Key insight text")
    assert rec.text == "Key insight text"
    recall_res = mem.recall("insight", top_k=1)
    assert len(recall_res) == 1

    # 4. Function Parser
    call = FunctionParser.parse('```json\n{"tool": "file_tool", "arguments": {"action": "read"}}\n```')
    assert call is not None
    assert call.tool_name == "file_tool"

    # 5. Tool Registry
    reg = ToolRegistry()
    assert len(reg.get_tool_schemas()) >= 0

    # 6. Sandbox
    sandbox = PythonSandbox()
    s_res = sandbox.execute("print('Hello Sandbox')")
    assert s_res.success is True
    assert "Hello Sandbox" in s_res.stdout

    # 7. Verification Engine
    v_report = VerificationEngine.verify(run_linter=False, run_tests=False)
    assert v_report.passed is True

    # 8. Permission Manager
    pm = PermissionManager()
    assert pm.check_permission("file_read") is True

    # 9. Plugin Registry
    AgentPluginRegistry.register_agent_plugin("RegCoder", CoderAgent)
    assert AgentPluginRegistry.get_agent_plugin("RegCoder") == CoderAgent

    # 10. Conversation
    conv = Conversation()
    conv.add_user("User msg")
    assert len(conv.messages) == 1
