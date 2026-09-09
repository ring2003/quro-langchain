"""Tests for the ACL engine (acl.py)."""

from __future__ import annotations

from quro.core.resources import (
    Decision,
    Principal,
    ResourceRef,
    acl_for_state,
    principal_from_state,
)


def _state() -> dict:
    return {
        "steps": [
            {"step_id": "s1", "step_type": "implement", "depends_on": [], "access": None},
            {"step_id": "s2", "step_type": "implement", "depends_on": ["s1"], "access": "hints"},
        ],
        "artifacts": [
            {"artifact_id": "a1", "step_id": "s1"},
            {"artifact_id": "a2", "step_id": "s2"},
        ],
        "recovery": {},
        "executing_step_id": "s2",
    }


def test_principal_from_state():
    principal = principal_from_state(_state())
    assert principal.step_id == "s2"
    assert principal.step_type == "implement"


def test_ownership_allow_own_artifact():
    engine = acl_for_state(_state())
    principal = principal_from_state(_state())
    verdict = engine.check(principal, ResourceRef("artifact", "a2"), "read")
    assert verdict.decision is Decision.ALLOW


def test_dependency_allow_artifact_with_hints():
    engine = acl_for_state(_state())
    principal = principal_from_state(_state())
    verdict = engine.check(principal, ResourceRef("artifact", "a1"), "read")
    assert verdict.decision is Decision.ALLOW


def test_dependency_allow_upstream_step():
    engine = acl_for_state(_state())
    principal = principal_from_state(_state())
    verdict = engine.check(principal, ResourceRef("step", "s1"), "read")
    assert verdict.decision is Decision.ALLOW


def test_foreign_artifact_denied():
    state = _state()
    state["artifacts"].append({"artifact_id": "a3", "step_id": "s3"})
    engine = acl_for_state(state)
    principal = principal_from_state(state)
    verdict = engine.check(principal, ResourceRef("artifact", "a3"), "read")
    assert verdict.decision is Decision.DENY


def test_write_own_step_allowed():
    engine = acl_for_state(_state())
    principal = principal_from_state(_state())
    verdict = engine.check(principal, ResourceRef("step", "s2"), "write")
    assert verdict.decision is Decision.ALLOW


def test_write_foreign_step_denied():
    engine = acl_for_state(_state())
    principal = principal_from_state(_state())
    verdict = engine.check(principal, ResourceRef("step", "s1"), "write")
    assert verdict.decision is Decision.DENY


def test_plan_read_denied_to_step_principal():
    engine = acl_for_state(_state())
    principal = principal_from_state(_state())
    verdict = engine.check(principal, ResourceRef("plan", "main"), "read")
    assert verdict.decision is Decision.DENY


def test_unset_principal_step_denied():
    """Default-deny holds even with no executing step (empty id ≠ empty id)."""
    state = _state()
    state["executing_step_id"] = None
    engine = acl_for_state(state)
    principal = principal_from_state(state)
    assert principal.step_id == ""
    verdict = engine.check(principal, ResourceRef("step", ""), "write")
    assert verdict.decision is Decision.DENY
