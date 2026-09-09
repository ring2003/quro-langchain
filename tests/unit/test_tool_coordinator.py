"""Tests for ToolCoordinator — unified tool governance (decisions 40–46).

Covers surface composition (contract / lifecycle / ``StepType.tools``),
default-deny allocation, profile expansion, the standalone ``shell`` grant,
missing-name fail-fast, and invoke routing.
"""

from __future__ import annotations

import pytest

from quro.core.features import Feature
from quro.core.tools import (
    CONTRACT_TOOLS,
    ToolAdapter,
    ToolCoordinator,
    domain_toolsets,
    framework_toolsets,
)
from quro.core.worker_tools import all_worker_tools
from quro.domains.codebase_research import CodebaseResearchDomain
from quro.steps.core import StepSpec


class _OkResp:
    ok = True
    error = None
    explanation = None


class _FakeSession:
    def call(self, op, args):
        return _OkResp()


class _FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name

    def invoke(self, args):
        return f"ran:{self.name}"


def _domain() -> CodebaseResearchDomain:
    return CodebaseResearchDomain()


def _coordinator(**overrides) -> ToolCoordinator:
    session = _FakeSession()
    domain = _domain()
    internal = {
        t.name: t
        for t in ToolAdapter(session).build(
            framework_toolsets() + domain_toolsets(domain)
        )
    }
    external = {t.name: t for t in all_worker_tools()}
    kwargs = dict(internal_tools=internal, external_tools=external)
    kwargs.update(overrides)
    return ToolCoordinator(**kwargs)


def _step(**kw) -> StepSpec:
    return StepSpec(id="s1", **kw)


def _names(tools) -> set[str]:
    return {t.name for t in tools}


def test_contract_tools_always_present():
    c = _coordinator()
    names = _names(c.surface(step=_step(), phase="survey", principal="worker"))
    assert set(CONTRACT_TOOLS) <= names


def test_artifact_tools_always_allocated():
    """get_artifact is unconditionally allocated (any step), not gated by
    depends_on — the L0 visibility fix (bugreport-artifact RC-A)."""
    c = _coordinator()
    names = _names(c.surface(step=_step(), phase="survey", principal="worker"))
    assert "add_artifact" in names
    assert "get_artifact" in names


def test_recovery_feature_injects_recovery_tools():
    c = _coordinator()
    plain = _names(c.surface(step=_step(), phase="survey", principal="worker"))
    assert "commit_clue" not in plain
    assert "continue_step" not in plain
    recovery = _names(
        c.surface(
            step=_step(features=(Feature.RECOVERY.value,)),
            phase="survey",
            principal="worker",
        )
    )
    assert "commit_clue" in recovery
    assert "continue_step" in recovery


def test_lifecycle_separation():
    c = _coordinator()
    worker = _names(c.surface(step=_step(), phase="survey", principal="worker"))
    planner = _names(c.surface(step=_step(), phase="survey", principal="planner"))
    evaluator = _names(c.surface(step=_step(), phase="survey", principal="evaluator"))

    assert "set_plan" not in worker
    assert "set_plan" in planner
    assert "create_step" in planner
    assert "submit_evaluation" not in planner
    assert "submit_evaluation" in evaluator


def test_shell_is_a_standalone_grant():
    c = _coordinator()
    names = _names(
        c.surface(step=_step(tools=["shell"]), phase="survey", principal="worker")
    )
    assert "shell" in names
    assert "read" not in names  # not granted alongside shell


def test_assemble_report_author_tool_via_step_type():
    c = _coordinator()
    step = _step(step_type="assemble_report", tools=["assemble_report"])
    names = _names(c.surface(step=step, phase="assemble_report", principal="worker"))
    assert "assemble_report" in names


def test_missing_tool_fails_fast():
    c = _coordinator()
    with pytest.raises(ValueError, match="INVALID_TOOL"):
        c.surface(step=_step(tools=["nonexistent"]), phase="survey", principal="worker")


def test_invoke_unknown_tool():
    c = _coordinator()
    assert c.invoke("nonexistent", {}) == "Unknown tool: nonexistent"


def test_invoke_routes_local_and_mcp():
    local = _FakeTool("local_tool")
    mcp = _FakeTool("mcp_tool")

    class _McpSession:
        def invoke_tool(self, name, args):
            return f"mcp:{name}"

    c = _coordinator(
        extra_tools=[local],
        mcp_tools=[mcp],
        mcp_session=_McpSession(),
    )
    assert c.invoke("local_tool", {}) == "ran:local_tool"
    assert c.invoke("mcp_tool", {}) == "mcp:mcp_tool"
