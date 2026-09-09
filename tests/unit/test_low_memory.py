"""Tests for the phase-2 low-memory payload offload + round semantics (stage 6)."""

from __future__ import annotations

from quro.core.resources import (
    DomainStateResolver,
    FileResourceStore,
    LivePolicy,
    ResourceRef,
    apply_event,
    offload_payload,
    resolve_offloaded,
)


def _store(tmp_path) -> FileResourceStore:
    return FileResourceStore(tmp_path / "session")


# -- payload offload (c11) --------------------------------------------------


def test_offload_keeps_state_bounded(tmp_path):
    store = _store(tmp_path)
    body = "x" * 100_000  # a large payload

    entry = offload_payload(
        store,
        ResourceRef("artifact", "art_a"),
        descriptor={"artifact_id": "art_a", "summary": "big result"},
        payload={"body": body},
    )

    # [c11] DomainState keeps only the descriptor + reference — not the payload.
    assert entry["artifact_id"] == "art_a"
    assert entry["summary"] == "big result"
    assert "ref" in entry
    assert "body" not in entry
    # The payload is on disk, not resident in state.
    assert store.load(ResourceRef("artifact", "art_a"))["body"] == body


def test_resolve_offloaded_loads_payload_on_access(tmp_path):
    store = _store(tmp_path)
    offload_payload(
        store,
        ResourceRef("artifact", "art_a"),
        descriptor={"artifact_id": "art_a", "summary": "s"},
        payload={"body": "the full body"},
    )
    entry = {"artifact_id": "art_a", "summary": "s", "ref": "artifact://art_a"}

    resolved = resolve_offloaded(store, entry)
    assert resolved["body"] == "the full body"
    assert resolved["artifact_id"] == "art_a"


def test_resolver_reads_offloaded_artifact_on_access(tmp_path):
    store = _store(tmp_path)
    offload_payload(
        store,
        ResourceRef("artifact", "art_a"),
        descriptor={"artifact_id": "art_a", "summary": "s"},
        payload={"body": "payload body"},
    )
    state = {
        "artifacts": [{"artifact_id": "art_a", "summary": "s", "ref": "artifact://art_a"}],
    }
    resolver = DomainStateResolver(state, store=store)

    read = resolver.read(ResourceRef("artifact", "art_a"))
    assert read["body"] == "payload body"
    assert read["artifact_id"] == "art_a"


def test_resolve_offloaded_missing_payload_is_dangling_tolerant(tmp_path):
    store = _store(tmp_path)
    entry = {"artifact_id": "art_a", "ref": "artifact://missing"}
    # Weak reference: a dangling ref falls back to the entry (not a crash).
    assert resolve_offloaded(store, entry) == entry


# -- LivePolicy round semantics (c12) ---------------------------------------


def test_live_policy_projects_and_materializes_nothing(tmp_path):
    store = _store(tmp_path)
    state = {"phase": "step_execute", "executing_step_id": "s2", "steps": []}

    before = dict(state)
    projected = LivePolicy.project(state, 3)

    # A read projects the current state and never implicitly writes.
    assert projected == {
        "round": "3",
        "phase": "step_execute",
        "executing_step_id": "s2",
        "provenance": True,
    }
    assert state == before  # no mutation
    assert store.load_snapshot(ResourceRef("round", "3")) is None  # nothing persisted


def test_resolver_round_read_uses_live_projection(tmp_path):
    store = _store(tmp_path)
    resolver = DomainStateResolver(
        {"phase": "evaluate", "executing_step_id": "s1"}, store=store,
    )
    assert resolver.read(ResourceRef("round", "5"))["round"] == "5"
    assert store.load_snapshot(ResourceRef("round", "5")) is None


# -- provenance stamped by runtime, absent in domain apply ------------------


def test_round_provenance_stamped_by_runtime_not_domain():
    # The domain's apply() is round-unaware: applying events never stamps a
    # round index.
    state: dict = {}
    apply_event(state, {"op": "add_artifact", "payload": {"artifact": {"artifact_id": "a1"}}})
    apply_event(state, {"op": "set_plan", "payload": {"plan": "P"}})
    assert "round" not in state
    assert "provenance" not in state

    # Provenance is stamped by the runtime at the round boundary.
    stamped = LivePolicy.project(state, 2)
    assert stamped["round"] == "2"
    assert stamped["provenance"] is True
