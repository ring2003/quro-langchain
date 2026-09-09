"""Mount adapter — the resource half of "mount" (phase 4).

``IMountAdapter`` is the two-op resource surface for exporting and re-importing
a step's config + artifacts, over ``IResourceStore`` only.  ``mount_out``
packages a step; ``mount_back`` folds the package's product back in with
provenance.  **Neither executes** — execution stays with the runtime
(primitive-step §4: ``f`` executes, ``◁`` inlines).  This is the resource layer
of ``runtime-session.md`` §11's mount word, not the execution layer.

Phase 4 lands the interface + a naive full-copy implementation + a no-op
placeholder.  Real mount semantics (target key allocation, provenance merge,
cross-model run) stay deferred — see the phase-4 plan §7.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from quro.core.resources.refs import (
    InvalidDescriptor,
    ResourceRef,
    parse_descriptor,
)
from quro.core.resources.store import IResourceStore


@runtime_checkable
class IMountAdapter(Protocol):
    """Move/fold a step's resources over ``IResourceStore`` (never executes)."""

    def mount_out(self, step_ref: ResourceRef, *, target: Any = None) -> ResourceRef:
        """Export *step_ref* as a self-contained runnable package.

        Returns the descriptor of the exported package (a new job / pipeline
        spec).  The source step is NOT mutated — the export is a copy.

        Args:
            step_ref: The step to export (its snapshot carries config + state).
            target: Optional explicit package descriptor (``ResourceRef`` or a
                descriptor string).  ``None`` derives a default package ref.
        """
        ...

    def mount_back(self, package_ref: ResourceRef, step_ref: ResourceRef) -> None:
        """Fold the package's product back into *step_ref*.

        Records provenance: the step's artifact was produced in *package_ref*,
        not in-place.  The source package is left intact.
        """
        ...


class NoopMountAdapter:
    """Interface placeholder: ``mount_out`` returns the input, ``mount_back`` no-ops."""

    def mount_out(self, step_ref: ResourceRef, *, target: Any = None) -> ResourceRef:
        return step_ref

    def mount_back(self, package_ref: ResourceRef, step_ref: ResourceRef) -> None:
        return None


class NaiveCopyMountAdapter:
    """Full self-contained copy over ``IResourceStore`` (naive, correct default).

    ``mount_out`` copies the step's snapshot (config + ``domain_state``)
    verbatim into a fresh package ref; ``mount_back`` folds the package's
    product back into the step's snapshot with a provenance record.  It touches
    only ``IResourceStore`` — never a physical path — so a future s3/container
    backend is a drop-in (acceptance c5).

    The default package key (``<kind>://<id>/export``) is a phase-4 placeholder;
    the real target-key allocation (derived job vs independent namespace) is an
    open point (phase-4 plan §7).  Callers may pass an explicit ``target``.
    """

    def __init__(self, store: IResourceStore) -> None:
        self._store = store

    def mount_out(self, step_ref: ResourceRef, *, target: Any = None) -> ResourceRef:
        package_ref = self._resolve_package_ref(step_ref, target)
        snapshot = self._store.load_snapshot(step_ref) or {}
        self._store.save(
            package_ref,
            {"source_step": str(step_ref), "snapshot": snapshot},
        )
        return package_ref

    def mount_back(self, package_ref: ResourceRef, step_ref: ResourceRef) -> None:
        package = self._store.load(package_ref) or {}
        snapshot = self._store.load_snapshot(step_ref) or {}
        provenance = list(snapshot.get("provenance", []))
        provenance.append({
            "package": str(package_ref),
            "product": package.get("product"),
        })
        snapshot["provenance"] = provenance
        self._store.save_snapshot(step_ref, snapshot)

    # -- helpers ------------------------------------------------------------

    def _resolve_package_ref(self, step_ref: ResourceRef, target: Any) -> ResourceRef:
        if target is None:
            # Phase-4 placeholder default: an "export" sibling under the step.
            return ResourceRef(step_ref.kind, step_ref.id, ("export",))
        if isinstance(target, ResourceRef):
            return target
        if isinstance(target, str):
            return parse_descriptor(target)
        raise InvalidDescriptor(f"unsupported mount target {target!r}")
