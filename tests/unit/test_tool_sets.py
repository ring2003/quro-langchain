"""Tests for the declarative tool framework (resource-scoped toolsets).

Covers ``architecture/evolution.md`` §2.6 and the unified-resource-layer
design doc §8: toolsets declare semantic tools as plain public methods over a
resource kind, and a single ``ToolAdapter`` maps them to runtime tools.
"""

from __future__ import annotations

from quro.core.tools import (
    RESOURCE_KIND_STEP,
    ArtifactTool,
    PlanningTool,
    StepTool,
    ToolAdapter,
)


class _OkResp:
    ok = True
    error = None
    explanation = None


class _FakeSession:
    def call(self, op, args):
        return _OkResp()


def test_toolset_collect_declares_resource_kind():
    specs = StepTool().collect()
    by_name = {s.name: s for s in specs}
    assert "create_step" in by_name
    assert "get_step" in by_name
    assert "update_step" in by_name
    assert all(s.resource_kind == RESOURCE_KIND_STEP for s in specs)


def test_artifact_tool_declares_artifact_kind():
    from quro.core.tools import RESOURCE_KIND_ARTIFACT
    specs = ArtifactTool().collect()
    assert all(s.resource_kind == RESOURCE_KIND_ARTIFACT for s in specs)


def test_update_step_merges_legacy_step_setters():
    names = {s.name for s in StepTool().collect()}
    assert "update_step" in names
    # The four legacy step-setter tools are collapsed into one update_step.
    assert "set_step_depends_on" not in names
    assert "set_step_access" not in names
    assert "set_step_hint_max_chars" not in names
    assert "add_step_field" not in names


def test_tool_adapter_builds_runtime_tools_with_schema():
    tools = ToolAdapter(_FakeSession()).build([StepTool(), PlanningTool()])
    by_name = {t.name: t for t in tools}
    assert "create_step" in by_name
    assert "confirm_understanding" in by_name
    # create_step's args_schema is derived from the method signature.
    schema = by_name["create_step"].args_schema.model_json_schema()
    props = schema["properties"]
    assert {"step_id", "objective", "step_type", "skills"} <= set(props)
    assert "role" not in props
