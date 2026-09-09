"""IResourceStore — pluggable, descriptor-keyed storage (architecture §4).

Phase 2 stage 1 lands the *stored* half of the unified resource layer: a
storage backend keyed by ``ResourceRef``, never a physical path.  The physical
layout is the store's private business; the organisation semantics (job →
session → round → artifact) are structured and mirrored by the layout — only
the *format* (json vs event file, flat vs tree) is private.

``FileResourceStore`` migrates the phase-1 ``FileArtifactStore`` directly to
descriptor keys and the state/products split::

    {base_dir}/
        artifacts/{id}.json                 # products (result-oriented)
        state/{kind}/{id}.json              # state resources (replay-only)
        state/{kind}/{id}/snapshot.json     # checkpoint snapshots
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from quro.core.resources.refs import ResourceRef


@runtime_checkable
class IResourceStore(Protocol):
    """Pluggable storage keyed by ``ResourceRef`` (architecture §4).

    Callers address resources by logical descriptor only — the concrete
    backend owns the physical layout.
    """

    def save(self, ref: ResourceRef, payload: dict[str, Any]) -> None: ...

    def load(self, ref: ResourceRef) -> dict[str, Any] | None: ...

    def list(self, ref: ResourceRef) -> list[dict[str, Any]]: ...

    def save_snapshot(self, ref: ResourceRef, snapshot: dict[str, Any]) -> None: ...

    def load_snapshot(self, ref: ResourceRef) -> dict[str, Any] | None: ...

    def append_event(self, ref: ResourceRef, event: dict[str, Any]) -> None: ...

    def read_events(self, ref: ResourceRef) -> list[dict[str, Any]]: ...


class FileResourceStore:
    """Descriptor-keyed file store (migrates ``FileArtifactStore``).

    Args:
        base_dir: Root directory for this store — the store's private root.
            Callers key resources by ``ResourceRef`` only; they never pass a
            physical path into ``save`` / ``load``.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base_dir = Path(base_dir)

    @property
    def base_dir(self) -> Path:
        """The store's private root directory (for deriving sibling namespaces)."""
        return self._base_dir

    # -- path mapping (private business) -----------------------------------

    def _resource_path(self, ref: ResourceRef) -> Path:
        if ref.kind == "artifact":
            return self._base_dir / "artifacts" / f"{ref.id}.json"
        return self._base_dir / "state" / ref.kind / f"{ref.id}.json"

    def _snapshot_path(self, ref: ResourceRef) -> Path:
        base = self._base_dir / "state" / ref.kind / ref.id
        if ref.subpath:
            return base / f"{ref.subpath[-1]}.json"
        return base / "snapshot.json"

    def _event_path(self, ref: ResourceRef) -> Path:
        base = self._base_dir / "state" / ref.kind / ref.id
        if ref.subpath:
            return base / f"{ref.subpath[-1]}.events.jsonl"
        return base / "events.jsonl"

    def _list_dir(self, ref: ResourceRef) -> Path:
        if ref.kind == "artifact":
            return self._base_dir / "artifacts"
        return self._base_dir / "state" / ref.kind

    # -- IResourceStore -----------------------------------------------------

    def save(self, ref: ResourceRef, payload: dict[str, Any]) -> None:
        self._write_json(self._resource_path(ref), payload)

    def load(self, ref: ResourceRef) -> dict[str, Any] | None:
        return self._read_json(self._resource_path(ref))

    def list(self, ref: ResourceRef) -> list[dict[str, Any]]:
        """List every resource of *ref*'s kind.

        For ``artifact://`` this returns all stored artifacts; for other
        kinds it returns all stored resources under ``state/{kind}/``.
        Round-aggregated sub-path listing (``round://3/artifacts``) is a
        later stage.
        """
        directory = self._list_dir(ref)
        if not directory.exists():
            return []
        out: list[dict[str, Any]] = []
        for f in sorted(directory.glob("*.json")):
            item = self._read_json(f)
            if item is not None:
                out.append(item)
        return out

    def save_snapshot(self, ref: ResourceRef, snapshot: dict[str, Any]) -> None:
        self._write_json(self._snapshot_path(ref), snapshot)

    def load_snapshot(self, ref: ResourceRef) -> dict[str, Any] | None:
        return self._read_json(self._snapshot_path(ref))

    def append_event(self, ref: ResourceRef, event: dict[str, Any]) -> None:
        """Append one event to the round-scoped event stream (append-only)."""
        p = self._event_path(ref)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def read_events(self, ref: ResourceRef) -> list[dict[str, Any]]:
        """Read the full ordered event list for *ref* (empty when absent)."""
        p = self._event_path(ref)
        if not p.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
