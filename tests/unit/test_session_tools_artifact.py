"""Tests for artifact content-type handling in session tools."""

from __future__ import annotations

from quro.core.domain import EngineeringWorkflowDomain
from quro.core.tools import ArtifactTool, ToolAdapter
from quro_thinking.kernel import ReasoningSession


def _make_session() -> ReasoningSession:
    domain = EngineeringWorkflowDomain()
    session = ReasoningSession(domain)
    session.call("engineering_workflow_init", {"problem": "test"})
    session.call("set_interpretation", {"text": "a"})
    session.call("set_plan", {"plan": "b"})
    session.call("confirm_understanding", {})
    session.call("create_step", {"step_id": "s1", "objective": "test"})
    session.call("finalize_step", {})
    return session


def _tools(session: ReasoningSession):
    tools = ToolAdapter(session).build([ArtifactTool()])
    return (
        next(t for t in tools if t.name == "add_artifact"),
        next(t for t in tools if t.name == "get_artifact"),
    )


def test_add_artifact_file_kind_requires_existing_path(tmp_path):
    session = _make_session()
    add_artifact, _ = _tools(session)

    result = add_artifact.invoke({"summary": "x", "kind": "file", "body": str(tmp_path / "missing.txt")})
    assert result.startswith("Error:")
    assert "not found" in result
    assert session.state.get("artifacts") == []


def test_add_artifact_file_kind_rejects_empty_body():
    session = _make_session()
    add_artifact, _ = _tools(session)

    result = add_artifact.invoke({"summary": "x", "kind": "file", "body": ""})
    assert result.startswith("Error:")
    assert session.state.get("artifacts") == []


def test_add_artifact_file_kind_succeeds_with_existing_path(tmp_path):
    session = _make_session()
    add_artifact, _ = _tools(session)
    f = tmp_path / "output.txt"
    f.write_text("hello file", encoding="utf-8")

    result = add_artifact.invoke({"summary": "x", "kind": "file", "body": str(f)})
    assert result.startswith("Artifact recorded")
    artifacts = session.state.get("artifacts", [])
    assert len(artifacts) == 1
    assert artifacts[0]["kind"] == "file"
    assert artifacts[0]["body"] == str(f)


def test_add_artifact_other_kind_accepts_inline_body():
    session = _make_session()
    add_artifact, _ = _tools(session)

    result = add_artifact.invoke({"summary": "x", "kind": "analysis", "body": "some notes"})
    assert result.startswith("Artifact recorded")
    artifacts = session.state.get("artifacts", [])
    assert artifacts[0]["kind"] == "analysis"
    assert artifacts[0]["body"] == "some notes"


def test_get_artifact_file_kind_returns_content(tmp_path):
    session = _make_session()
    add_artifact, get_artifact = _tools(session)
    f = tmp_path / "output.txt"
    f.write_text("file content here", encoding="utf-8")

    add_artifact.invoke({"summary": "x", "kind": "file", "body": str(f)})
    artifact_id = session.state["artifacts"][0]["artifact_id"]

    raw = get_artifact.invoke({"artifact_id": artifact_id})
    assert "file content here" in raw
    assert "content" in raw


def test_get_artifact_other_kind_has_no_content():
    session = _make_session()
    add_artifact, get_artifact = _tools(session)

    add_artifact.invoke({"summary": "x", "kind": "analysis", "body": "notes"})
    artifact_id = session.state["artifacts"][0]["artifact_id"]

    raw = get_artifact.invoke({"artifact_id": artifact_id})
    assert "notes" in raw
    assert '"content"' not in raw
