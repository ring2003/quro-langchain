"""Tests for the mount adapter (phase 4 — the resource half of mount)."""

from __future__ import annotations

from quro.core.resources import (
    FileResourceStore,
    IMountAdapter,
    NaiveCopyMountAdapter,
    NoopMountAdapter,
    ResourceRef,
)


def _store(tmp_path) -> FileResourceStore:
    return FileResourceStore(tmp_path / "session")


def _seed_step(store: FileResourceStore, step_id: str = "s1") -> None:
    store.save_snapshot(
        ResourceRef("step", step_id),
        {
            "step_id": step_id,
            "domain_state": {"steps": [{"step_id": step_id}]},
            "step_config": {"id": step_id},
        },
    )


# -- c3: both adapters satisfy the protocol --------------------------------


def test_protocol_satisfied_by_both_adapters(tmp_path):
    assert isinstance(NoopMountAdapter(), IMountAdapter)
    assert isinstance(NaiveCopyMountAdapter(_store(tmp_path)), IMountAdapter)


# -- c4: no-op adapter ------------------------------------------------------


def test_noop_mount_out_returns_input(tmp_path):
    store = _store(tmp_path)
    ref = ResourceRef("step", "s1")
    assert NoopMountAdapter().mount_out(ref) == ref
    # nothing written anywhere
    assert store.load_snapshot(ref) is None
    assert store.load(ref) is None


def test_noop_mount_back_is_noop(tmp_path):
    store = _store(tmp_path)
    NoopMountAdapter().mount_back(ResourceRef("step", "p1"), ResourceRef("step", "s1"))
    assert store.load_snapshot(ResourceRef("step", "s1")) is None


# -- c5: naive copy uses only IResourceStore -------------------------------


def test_naive_mount_out_copies_snapshot_to_package(tmp_path):
    store = _store(tmp_path)
    _seed_step(store, "s1")
    adapter = NaiveCopyMountAdapter(store)

    package_ref = adapter.mount_out(ResourceRef("step", "s1"))
    assert package_ref == ResourceRef("step", "s1", ("export",))

    package = store.load(package_ref)
    assert package is not None
    assert package["source_step"] == "step://s1"
    assert package["snapshot"]["step_id"] == "s1"
    # source is not mutated
    assert store.load_snapshot(ResourceRef("step", "s1"))["step_id"] == "s1"


def test_naive_mount_out_explicit_target(tmp_path):
    store = _store(tmp_path)
    _seed_step(store, "s1")
    adapter = NaiveCopyMountAdapter(store)

    target = ResourceRef("artifact", "exported_s1")
    assert adapter.mount_out(ResourceRef("step", "s1"), target=target) == target
    assert store.load(target) is not None


def test_naive_mount_back_folds_provenance(tmp_path):
    store = _store(tmp_path)
    _seed_step(store, "s1")
    adapter = NaiveCopyMountAdapter(store)

    package_ref = adapter.mount_out(ResourceRef("step", "s1"))
    # simulate a product produced inside the package
    store.save(
        package_ref,
        {"source_step": "step://s1", "snapshot": {}, "product": "art_x"},
    )
    adapter.mount_back(package_ref, ResourceRef("step", "s1"))

    snapshot = store.load_snapshot(ResourceRef("step", "s1"))
    assert snapshot["provenance"][0]["package"] == "step://s1/export"
    assert snapshot["provenance"][0]["product"] == "art_x"
