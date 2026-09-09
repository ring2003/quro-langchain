"""Tests for the phase-2 event replay + resume (stage 5)."""

from __future__ import annotations

from quro.core.resources import (
    FileResourceStore,
    ResourceRef,
    apply_event,
    replay_events,
    resume_domain_state,
)


def _store(tmp_path) -> FileResourceStore:
    return FileResourceStore(tmp_path / "session")


# -- apply_event ------------------------------------------------------------


def test_apply_add_artifact():
    state: dict = {}
    apply_event(state, {"op": "add_artifact", "payload": {"artifact": {"artifact_id": "a1"}}})
    assert state["artifacts"] == [{"artifact_id": "a1"}]


def test_apply_update_step_status():
    state = {"steps": [{"step_id": "s1", "status": "in_progress"}]}
    apply_event(state, {"op": "update_step_status", "payload": {"step_id": "s1", "status": "completed"}})
    assert state["steps"][0]["status"] == "completed"


def test_apply_commit_clue():
    state: dict = {}
    apply_event(state, {"op": "commit_clue", "payload": {"step_id": "s1", "clue": "found x"}})
    assert state["recovery"]["s1"]["clues"] == ["found x"]


def test_apply_scalar_ops():
    state: dict = {}
    apply_event(state, {"op": "set_plan", "payload": {"plan": "P"}})
    apply_event(state, {"op": "set_interpretation", "payload": {"text": "I"}})
    apply_event(state, {"op": "set_phase", "payload": {"phase": "evaluate", "executing_step_id": "s2"}})
    assert state["plan"] == "P"
    assert state["interpretation"] == "I"
    assert state["phase"] == "evaluate"
    assert state["executing_step_id"] == "s2"


def test_apply_unknown_op_is_skipped():
    state = {"steps": []}
    apply_event(state, {"op": "future_op", "payload": {"x": 1}})
    assert state == {"steps": []}


# -- replay_events ----------------------------------------------------------


def test_replay_does_not_mutate_base():
    base = {"artifacts": [{"artifact_id": "a0"}]}
    events = [{"op": "add_artifact", "payload": {"artifact": {"artifact_id": "a1"}}}]
    out = replay_events(base, events)

    assert base == {"artifacts": [{"artifact_id": "a0"}]}  # unchanged
    assert [a["artifact_id"] for a in out["artifacts"]] == ["a0", "a1"]


# -- resume_domain_state (c9) -----------------------------------------------


def test_resume_after_interrupt_yields_identical_state(tmp_path):
    store = _store(tmp_path)

    # Round-start snapshot: the replay anchor.
    base_state = {
        "steps": [{"step_id": "s1", "status": "completed"}],
        "artifacts": [{"artifact_id": "art_a", "step_id": "s1"}],
        "plan": "P",
    }
    store.save_snapshot(
        ResourceRef("round", "0", ("start",)),
        {"round_idx": 0, "goal": "g", "domain_state": base_state},
    )

    # Round operations appended after the snapshot.
    round_events = [
        {"op": "add_artifact", "payload": {"artifact": {"artifact_id": "art_b", "step_id": "s2"}}},
        {"op": "update_step_status", "payload": {"step_id": "s2", "status": "completed"}},
    ]
    for e in round_events:
        store.append_event(ResourceRef("round", "0"), e)

    # The pre-interrupt state is base + applied events.
    expected = replay_events(base_state, round_events)

    # Simulated interrupt → resume from the round-start snapshot + events.
    resumed = resume_domain_state(
        store,
        ResourceRef("round", "0", ("start",)),
        event_ref=ResourceRef("round", "0"),
    )
    assert resumed == expected
    assert [a["artifact_id"] for a in resumed["artifacts"]] == ["art_a", "art_b"]


def test_resume_unknown_snapshot_returns_empty(tmp_path):
    store = _store(tmp_path)
    assert resume_domain_state(store, ResourceRef("round", "99")) == {}


def test_step_snapshot_resume_without_events(tmp_path):
    store = _store(tmp_path)
    store.save_snapshot(
        ResourceRef("step", "s2"),
        {"step_id": "s2", "domain_state": {"steps": [{"step_id": "s1"}]}},
    )
    resumed = resume_domain_state(store, ResourceRef("step", "s2"))
    assert resumed == {"steps": [{"step_id": "s1"}]}
