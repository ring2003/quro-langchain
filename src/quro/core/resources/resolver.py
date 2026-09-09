"""Resource resolver — logical descriptor → data (architecture §3).

Phase 1 is a **pure projection** over ``DomainState`` (no new storage): the
resolver knows how ``step://`` maps to ``state["steps"]``, ``artifact://`` to
``state["artifacts"]``, ``clue://`` to ``state["recovery"]``, ``plan://`` /
``interpretation://`` to the scalar fields, and ``round://`` to current-round
provenance.  ``DomainState`` itself stays a plain JSON-safe dict (design doc
§7.1).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from quro.core.resources.offload import resolve_offloaded
from quro.core.resources.refs import ResourceRef


class ResourceNotFound(KeyError):
    """Raised when a descriptor addresses a resource that does not exist."""


@runtime_checkable
class IResourceResolver(Protocol):
    """Resolve a logical descriptor to data (lazy; raises ResourceNotFound)."""

    def read(self, ref: ResourceRef) -> Any: ...


def _find_step(state: dict[str, Any], step_id: str) -> dict[str, Any] | None:
    for s in state.get("steps", []):
        if isinstance(s, dict) and s.get("step_id") == step_id:
            return s
    return None


def _find_artifact(state: dict[str, Any], artifact_id: str) -> dict[str, Any] | None:
    for a in state.get("artifacts", []):
        if isinstance(a, dict) and a.get("artifact_id") == artifact_id:
            return a
    return None


class DomainStateResolver:
    """Resolve logical descriptors against a ``DomainState`` dict.

    Args:
        state: The ``DomainState`` to resolve against.
        store: Optional ``IResourceStore`` for on-access payload resolution.
            When a state entry carries an offloaded ``ref``, the payload is
            loaded from the store on access (low-memory; architecture §7).
    """

    def __init__(
        self,
        state: dict[str, Any] | None = None,
        *,
        store: Any = None,
    ) -> None:
        self._state: dict[str, Any] = state or {}
        self._store = store

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    def read(self, ref: ResourceRef) -> Any:
        kind = ref.kind
        if kind == "step":
            step = _find_step(self._state, ref.id)
            if step is None:
                raise ResourceNotFound(f"step not found: {ref.id}")
            return step
        if kind == "artifact":
            artifact = _find_artifact(self._state, ref.id)
            if artifact is None:
                raise ResourceNotFound(f"artifact not found: {ref.id}")
            if self._store is not None:
                return resolve_offloaded(self._store, artifact)
            return artifact
        if kind == "clue":
            journal = (self._state.get("recovery") or {}).get(ref.id)
            if journal is None:
                raise ResourceNotFound(f"clue journal not found: {ref.id}")
            if ref.subpath == ("clues",):
                return journal.get("clues", [])
            return journal
        if kind == "plan":
            return self._state.get("plan", "")
        if kind == "interpretation":
            return self._state.get("interpretation", "")
        if kind == "round":
            return self._round_provenance(ref)
        raise ResourceNotFound(f"unsupported kind: {kind}")

    def _round_provenance(self, ref: ResourceRef) -> dict[str, Any]:
        """Round semantics are provenance-only in phase 1 (design doc §5).

        No snapshot materialization exists yet, so a round reference resolves
        against the *current* live projection, stamped with the requested
        round label as provenance.
        """
        return LivePolicy.project(self._state, ref.id)


class LivePolicy:
    """Round read projection — projects the current state, materializes nothing.

    Locked decision 4 (architecture §8): a round read projects the *current*
    ``DomainState`` and never implicitly writes.  Provenance is stamped by the
    runtime at the round boundary, never by the domain (the domain's
    ``apply()`` is round-unaware).
    """

    @staticmethod
    def project(state: dict[str, Any] | None, round_idx: Any) -> dict[str, Any]:
        """Project the live round view of *state* (no materialization)."""
        state = state or {}
        return {
            "round": str(round_idx),
            "phase": state.get("phase", ""),
            "executing_step_id": state.get("executing_step_id"),
            "provenance": True,
        }
