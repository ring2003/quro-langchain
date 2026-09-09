"""Tests for the resource graph (graph.py)."""

from __future__ import annotations

from quro.core.resources import DEPENDS_ON, OWNS, DomainStateGraph, ResourceRef


def _state() -> dict:
    return {
        "steps": [
            {"step_id": "s1", "depends_on": []},
            {"step_id": "s2", "depends_on": ["s1"]},
        ],
        "artifacts": [{"artifact_id": "a1", "step_id": "s1"}],
        "recovery": {"s1": {"clues": []}},
    }


def test_nodes():
    graph = DomainStateGraph.from_state(_state())
    nodes = graph.nodes()
    assert ResourceRef("step", "s1") in nodes
    assert ResourceRef("step", "s2") in nodes
    assert ResourceRef("artifact", "a1") in nodes
    assert ResourceRef("clue", "s1") in nodes


def test_depends_on_outgoing():
    graph = DomainStateGraph.from_state(_state())
    assert graph.outgoing(ResourceRef("step", "s2"), DEPENDS_ON) == {
        ResourceRef("step", "s1")
    }


def test_owns_outgoing():
    graph = DomainStateGraph.from_state(_state())
    owned = graph.outgoing(ResourceRef("step", "s1"), OWNS)
    assert ResourceRef("artifact", "a1") in owned
    assert ResourceRef("clue", "s1") in owned


def test_depends_on_incoming():
    graph = DomainStateGraph.from_state(_state())
    assert graph.incoming(ResourceRef("step", "s1"), DEPENDS_ON) == {
        ResourceRef("step", "s2")
    }


def test_owner_of_artifact():
    graph = DomainStateGraph.from_state(_state())
    assert graph.owner_of(ResourceRef("artifact", "a1")) == ResourceRef("step", "s1")


def test_edges_lists_all_relations():
    graph = DomainStateGraph.from_state(_state())
    rels = {rel for _, _, rel in graph.edges()}
    assert DEPENDS_ON in rels
    assert OWNS in rels
