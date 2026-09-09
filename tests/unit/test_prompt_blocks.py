"""Targeted tests for the prompt block scaffolding (blocks).

Covers the core of the prompt-management scaffolding design:
- the cognitive ordering keeps objective last;
- domains declare their phase → block mapping via ``@phase_blocks``.

The coordinator assembly behaviour (ordering + projection grants) is exercised
in ``test_projection_grants.py``.
"""

from __future__ import annotations

from quro.context.blocks import (
    KIND_ORDER,
    phase_blocks,
    registered_blocks,
    render_block,
)


def _steering_artifact_state() -> dict:
    """Session state with own + dependency + unrelated-step artifacts."""
    return {
        "executing_step_id": "step-1",
        "steps": [
            {"step_id": "step-1", "depends_on": ["step-0"], "access": "hints"},
            {"step_id": "step-0", "depends_on": [], "access": None},
            {"step_id": "step-9", "depends_on": [], "access": None},
        ],
        "artifacts": [
            {
                "artifact_id": "art-a",
                "step_id": "step-1",
                "kind": "analysis",
                "summary": "own summary",
                "evidences": ["e-own"],
                "body": "OWN BODY CONTENT",
            },
            {
                "artifact_id": "art-b",
                "step_id": "step-0",
                "kind": "analysis",
                "summary": "dep summary",
                "evidences": [],
                "body": "DEP BODY CONTENT",
            },
            {
                "artifact_id": "art-other",
                "step_id": "step-9",
                "kind": "analysis",
                "summary": "other",
                "body": "OTHER BODY",
            },
        ],
    }


def test_kind_order_objective_is_last():
    assert KIND_ORDER["objective"] > KIND_ORDER["hints"]
    assert KIND_ORDER["objective"] > KIND_ORDER["skills"]


def test_steering_blocks_in_kind_order():
    assert KIND_ORDER["steering_artifacts"] < KIND_ORDER["objective"]
    assert KIND_ORDER["steering_context"] < KIND_ORDER["objective"]
    assert "steering_artifacts" in registered_blocks()


def test_steering_artifacts_renders_full_bodies_own_and_deps():
    """The steering act's artifact view carries FULL body content, not a
    count/summary index — the L1 visibility fix (bugreport-artifact)."""
    text = render_block(
        "steering_artifacts", _steering_artifact_state(),
        phase="", budget=0, principal="steering",
    )
    # Canonical unified view: one heading reports the own + dependency split.
    assert "1 own, 1 dependency (access=hints)" in text
    # Own history: fully rendered, body included.
    assert "art-a" in text
    assert "OWN BODY CONTENT" in text
    # Dependencies (access='hints'): fully rendered, body included.
    assert "art-b" in text
    assert "DEP BODY CONTENT" in text
    # Unrelated step's artifact is excluded.
    assert "art-other" not in text
    assert "OTHER BODY" not in text


def test_steering_artifacts_deps_gated_by_access():
    """Without access='hints', dependency artifacts are NOT exposed."""
    st = _steering_artifact_state()
    st["steps"][0]["access"] = None
    text = render_block(
        "steering_artifacts", st,
        phase="", budget=0, principal="steering",
    )
    assert "art-a" in text
    assert "art-b" not in text


def test_steering_artifacts_empty_without_artifacts():
    st = {
        "executing_step_id": "step-1",
        "steps": [{"step_id": "step-1", "depends_on": [], "access": None}],
        "artifacts": [],
    }
    assert render_block(
        "steering_artifacts", st,
        phase="", budget=0, principal="steering",
    ) == ""


def test_steering_artifacts_respects_budget():
    text = render_block(
        "steering_artifacts", _steering_artifact_state(),
        phase="", budget=16, principal="steering",
    )
    assert "CONTEXT TRUNCATED" in text
    assert "art-a" not in text


def test_domain_declares_phase_blocks():
    from quro.core.domain.workflow_domain import EngineeringWorkflowDomain
    from quro.domains.codebase_research.domain import CodebaseResearchDomain

    research = CodebaseResearchDomain.PHASE_BLOCKS
    assert "survey" in research
    assert research["survey"][-1] == "objective"
    # L0 survey must surface prior artifacts (own_history) and the full
    # artifact store (artifacts) so the RECALL ids from steering surface in
    # the executor prompt (bugreport-artifact RC-B).
    assert "own_history" in research["survey"]
    assert "artifacts" in research["survey"]
    assert research["assemble_report"] == ["problem", "evidence", "report_chunks", "objective"]

    workflow = EngineeringWorkflowDomain.PHASE_BLOCKS
    assert "step_execute" in workflow
    assert workflow["step_execute"][-1] == "objective"


def test_registered_blocks_cover_research_blocks():
    names = set(registered_blocks())
    assert {"problem", "plan", "hints", "evidence", "report_chunks", "module_inventory"} <= names


def test_phase_blocks_decorator_sets_class_attr():
    @phase_blocks({"a": ["problem"]})
    class _Dummy:
        pass

    assert _Dummy.PHASE_BLOCKS == {"a": ["problem"]}
