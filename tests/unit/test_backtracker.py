"""Tests for Phase C — the built-in backtracker (§2.7)."""

from __future__ import annotations

from quro.core.resources import DomainStateGraph
from quro.runtime.backtracker import (
    BacktrackReport,
    Backtracker,
    RoundObject,
)


# ---------------------------------------------------------------------------
# RoundObject
# ---------------------------------------------------------------------------


def test_round_object_from_state():
    state = {
        "steps": [{"step_id": "s1", "depends_on": []}],
        "artifacts": [{"artifact_id": "a1", "step_id": "s1"}],
        "executing_step_id": "s1",
        "phase": "step_execute",
    }
    obj = RoundObject.from_state(state)
    assert obj.step_ids() == {"s1"}
    assert obj.artifact_step_ids() == {"s1"}
    assert obj.executing_step_id == "s1"
    assert obj.phase == "step_execute"


def test_round_object_from_none():
    obj = RoundObject.from_state(None)
    assert obj.steps == []
    assert obj.step_ids() == set()


# ---------------------------------------------------------------------------
# Backtracker structural analysis
# ---------------------------------------------------------------------------


def _state(steps, artifacts=None, executing="s2"):
    return {
        "steps": steps,
        "artifacts": artifacts or [],
        "executing_step_id": executing,
        "phase": "step_execute",
    }


def test_broken_deps_detected():
    state = _state([
        {"step_id": "s1", "status": "completed", "depends_on": []},
        {"step_id": "s2", "status": "in_progress", "depends_on": ["s_missing"]},
    ])
    report = Backtracker().rebuild(state, failed_step_id="s2")
    assert report.broken_deps == [{"step_id": "s2", "missing_dep": "s_missing"}]
    assert "s_missing" in report.next_step_suggestion


def test_missing_artifacts_detected():
    state = _state([
        {"step_id": "s1", "status": "completed", "depends_on": []},
    ], artifacts=[], executing="s2")
    report = Backtracker().rebuild(state, failed_step_id="s1")
    assert report.missing_artifacts == ["s1"]
    assert "s1" in report.next_step_suggestion


def test_blocked_and_ready_steps():
    state = _state([
        {"step_id": "s1", "status": "completed", "depends_on": []},
        {"step_id": "s2", "status": "pending", "depends_on": ["s1"]},
        {"step_id": "s3", "status": "pending", "depends_on": ["s2"]},
    ])
    report = Backtracker().rebuild(state, failed_step_id="s1")
    assert report.ready_steps == ["s2"]
    assert report.blocked_steps == ["s3"]


def test_next_step_suggestion_priority_broken_dep_first():
    state = _state([
        {"step_id": "s1", "status": "completed", "depends_on": ["ghost"]},
        {"step_id": "s2", "status": "pending", "depends_on": []},
    ])
    report = Backtracker().rebuild(state, failed_step_id="s1")
    # broken deps take priority over ready_steps
    assert "ghost" in report.next_step_suggestion
    assert report.broken_deps


def test_context_rebuild_includes_topology():
    state = _state([
        {"step_id": "s1", "status": "completed", "depends_on": []},
        {"step_id": "s2", "status": "in_progress", "depends_on": ["s1"]},
    ])
    report = Backtracker().rebuild(state, failed_step_id="s2")
    assert "## Round Rebuild" in report.context_rebuild
    assert "s1 [completed] depends_on=[—]" in report.context_rebuild
    assert "s2 [in_progress] depends_on=[s1]" in report.context_rebuild


def test_report_to_dict_roundtrip():
    report = BacktrackReport(
        failed_step_id="s1",
        broken_deps=[{"step_id": "s1", "missing_dep": "x"}],
        next_step_suggestion="fix",
        context_rebuild="## Round Rebuild",
    )
    d = report.to_dict()
    assert d["failed_step_id"] == "s1"
    assert d["broken_deps"] == [{"step_id": "s1", "missing_dep": "x"}]


def test_backtracker_accepts_injected_graph():
    """Stage 2: the backtracker reads the shared edge table when injected."""
    state = _state([
        {"step_id": "s1", "status": "completed", "depends_on": []},
        {"step_id": "s2", "status": "pending", "depends_on": ["s1"]},
        {"step_id": "s3", "status": "pending", "depends_on": ["s2"]},
    ])
    graph = DomainStateGraph.from_state(state)
    report = Backtracker(graph=graph).rebuild(state, failed_step_id="s1")
    assert report.ready_steps == ["s2"]
    assert report.blocked_steps == ["s3"]
    assert report.broken_deps == []


def test_backtracker_reads_restored_state_from_snapshot(tmp_path):
    """[c10] the graph rebuild reads the restored state, not post-hoc state."""
    from quro.core.resources import FileResourceStore, ResourceRef

    store = FileResourceStore(tmp_path / "session")
    restored = _state([
        {"step_id": "s1", "status": "completed", "depends_on": []},
        {"step_id": "s2", "status": "in_progress", "depends_on": ["s_missing"]},
    ])
    store.save_snapshot(
        ResourceRef("step", "s2"),
        {"step_id": "s2", "domain_state": restored},
    )

    bt = Backtracker.from_snapshot(store, "s2", failed_step_id="s2")
    report = bt.rebuild(None)  # state omitted → restored state is used

    assert report.failed_step_id == "s2"
    assert report.broken_deps == [{"step_id": "s2", "missing_dep": "s_missing"}]
    assert "s_missing" in report.next_step_suggestion
