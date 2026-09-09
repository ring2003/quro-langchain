"""Tests for the phase-2 descriptor-keyed ``IResourceStore`` / ``FileResourceStore``.

Stage 1 acceptance:
  [c1] ``store.save(ResourceRef("artifact", "art_x"), payload)`` then
       ``store.load(...)`` round-trips; the store never receives a physical
       path.
"""

from __future__ import annotations

from quro.core.resources import FileResourceStore, IResourceStore, ResourceRef


def _store(tmp_path) -> FileResourceStore:
    return FileResourceStore(tmp_path / "session")


def test_store_is_protocol_driven(tmp_path):
    store = _store(tmp_path)
    assert isinstance(store, IResourceStore)


def test_save_load_round_trips(tmp_path):
    store = _store(tmp_path)
    ref = ResourceRef("artifact", "art_x")
    payload = {"artifact_id": "art_x", "summary": "hello", "body": "world"}

    store.save(ref, payload)
    assert store.load(ref) == payload


def test_load_unknown_descriptor_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.load(ResourceRef("artifact", "missing")) is None
    assert store.load(ResourceRef("clue", "missing")) is None


def test_list_returns_all_artifacts_sorted(tmp_path):
    store = _store(tmp_path)
    store.save(ResourceRef("artifact", "art_b"), {"artifact_id": "art_b"})
    store.save(ResourceRef("artifact", "art_a"), {"artifact_id": "art_a"})

    artifacts = store.list(ResourceRef("artifact", ""))
    assert [a["artifact_id"] for a in artifacts] == ["art_a", "art_b"]


def test_list_empty_when_no_resources(tmp_path):
    store = _store(tmp_path)
    assert store.list(ResourceRef("artifact", "")) == []


def test_snapshot_round_trip(tmp_path):
    store = _store(tmp_path)
    ref = ResourceRef("round", "3")
    snapshot = {"round": 3, "phase": "step_execute", "steps": []}

    store.save_snapshot(ref, snapshot)
    assert store.load_snapshot(ref) == snapshot
    assert store.load_snapshot(ResourceRef("round", "9")) is None


def test_artifact_and_state_are_separate_namespaces(tmp_path):
    store = _store(tmp_path)
    store.save(ResourceRef("artifact", "art_x"), {"artifact_id": "art_x"})
    store.save(ResourceRef("clue", "s1"), {"step_id": "s1", "clues": []})

    # artifact:// and clue:// live under different namespaces.
    assert store.load(ResourceRef("artifact", "art_x")) == {"artifact_id": "art_x"}
    assert store.load(ResourceRef("clue", "s1")) == {"step_id": "s1", "clues": []}
    assert store.load(ResourceRef("artifact", "s1")) is None


def test_products_layout_on_disk(tmp_path):
    store = _store(tmp_path)
    store.save(ResourceRef("artifact", "art_x"), {"artifact_id": "art_x"})

    # The physical layout is the store's private business, but the state/
    # products split must be visible on disk (architecture §4).
    assert (tmp_path / "session" / "artifacts" / "art_x.json").exists()
