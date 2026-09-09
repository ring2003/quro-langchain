"""Tests for the DomainStateResolver (resolver.py)."""

from __future__ import annotations

import pytest

from quro.core.resources import DomainStateResolver, ResourceNotFound, ResourceRef


def _state() -> dict:
    return {
        "steps": [{"step_id": "s1", "objective": "o", "depends_on": []}],
        "artifacts": [{"artifact_id": "a1", "step_id": "s1", "summary": "x"}],
        "recovery": {"s1": {"clues": [{"text": "c1"}]}},
        "plan": "the plan",
        "interpretation": "the interp",
        "phase": "step_execute",
        "executing_step_id": "s1",
    }


def test_step_resolves():
    data = DomainStateResolver(_state()).read(ResourceRef("step", "s1"))
    assert data["objective"] == "o"


def test_artifact_resolves():
    data = DomainStateResolver(_state()).read(ResourceRef("artifact", "a1"))
    assert data["summary"] == "x"


def test_clue_resolves_subpath():
    data = DomainStateResolver(_state()).read(
        ResourceRef("clue", "s1", ("clues",))
    )
    assert data == [{"text": "c1"}]


def test_plan_resolves():
    assert DomainStateResolver(_state()).read(ResourceRef("plan", "main")) == "the plan"


def test_interpretation_resolves():
    assert DomainStateResolver(_state()).read(
        ResourceRef("interpretation", "main")
    ) == "the interp"


def test_round_resolves_provenance():
    data = DomainStateResolver(_state()).read(ResourceRef("round", "3"))
    assert data["phase"] == "step_execute"
    assert data["round"] == "3"


def test_missing_step_raises():
    with pytest.raises(ResourceNotFound):
        DomainStateResolver(_state()).read(ResourceRef("step", "nope"))


def test_missing_artifact_raises():
    with pytest.raises(ResourceNotFound):
        DomainStateResolver(_state()).read(ResourceRef("artifact", "nope"))


def test_missing_clue_raises():
    with pytest.raises(ResourceNotFound):
        DomainStateResolver(_state()).read(ResourceRef("clue", "nope"))
