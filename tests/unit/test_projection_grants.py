"""Tests for projection grants (grants.py + coordinator assembly)."""

from __future__ import annotations

import pytest

from quro.context.coordinator import PromptCoordinator, SystemBlocksHook, UnknownBlockError
from quro.core.resources import (
    DomainStateResolver,
    ProjectionGrant,
    ProjectionGrantRegistry,
    ResourceRef,
    acl_for_state,
    principal_from_state,
)


def _state() -> dict:
    return {
        "steps": [
            {"step_id": "s1", "step_type": "implement", "depends_on": [], "access": None},
            {"step_id": "s2", "step_type": "deep_dive", "depends_on": ["s1"], "access": "hints"},
        ],
        "artifacts": [
            {"artifact_id": "a1", "step_id": "s1", "summary": "evidence", "body": "found X"},
        ],
        "recovery": {},
        "executing_step_id": "s2",
        "plan": "the plan",
        "phase": "step_execute",
    }


def _coordinator(state, grants):
    return PromptCoordinator(
        [SystemBlocksHook("identity")],
        principal="deep_dive",
        phase="step_execute",
        grants=grants,
        acl=acl_for_state(state),
        acl_principal=principal_from_state(state),
        resolver=DomainStateResolver(state),
    )


def test_grant_adds_block_when_source_allowed():
    grants = ProjectionGrantRegistry([
        ProjectionGrant(
            principal="deep_dive",
            blocks=("evidence",),
            source=ResourceRef("artifact", "a1"),
            reason="deep_dive needs upstream evidence",
        ),
    ])
    system, user = _coordinator(_state(), grants).assemble()
    assert "evidence" in user
    assert "found X" in user["evidence"]


def test_grant_denied_source_contributes_nothing():
    grants = ProjectionGrantRegistry([
        ProjectionGrant(
            principal="deep_dive",
            blocks=("plan",),
            source=ResourceRef("plan", "main"),
            reason="deep_dive must align with the plan",
        ),
    ])
    system, user = _coordinator(_state(), grants).assemble()
    # plan:// is denied to a step principal, so the grant contributes nothing.
    assert "plan" not in user


def test_unknown_block_fails_at_assembly():
    grants = ProjectionGrantRegistry([
        ProjectionGrant(
            principal="deep_dive",
            blocks=("not_a_block",),
            source=ResourceRef("artifact", "a1"),
            reason="bad block name",
        ),
    ])
    with pytest.raises(UnknownBlockError):
        _coordinator(_state(), grants).assemble()


def test_unknown_block_fails_fast_even_with_denied_source():
    """Reference integrity is checked before the ACL gate (fail-fast)."""
    grants = ProjectionGrantRegistry([
        ProjectionGrant(
            principal="deep_dive",
            blocks=("not_a_block",),
            source=ResourceRef("plan", "main"),  # denied source
            reason="bad block name + denied source",
        ),
    ])
    with pytest.raises(UnknownBlockError):
        _coordinator(_state(), grants).assemble()


def test_non_matching_principal_is_ignored():
    grants = ProjectionGrantRegistry([
        ProjectionGrant(
            principal="survey_module",
            blocks=("evidence",),
            source=ResourceRef("artifact", "a1"),
            reason="only survey_module",
        ),
    ])
    system, user = _coordinator(_state(), grants).assemble()
    assert "evidence" not in user


def test_manifest_aggregates():
    grants = ProjectionGrantRegistry([
        ProjectionGrant(principal="p1", blocks=("a",), source=ResourceRef("step", "s1"), reason="r1"),
        ProjectionGrant(principal="p2", blocks=("b",), source=ResourceRef("step", "s2"), reason="r2"),
    ])
    manifest = grants.manifest()
    assert [g.principal for g in manifest] == ["p1", "p2"]
    assert [g.reason for g in manifest] == ["r1", "r2"]
