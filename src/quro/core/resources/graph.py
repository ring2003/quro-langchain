"""Resource graph — the edge table (architecture §4).

The edge table is the **single source of truth** for dependency and ownership
relations, extracted from ``DomainState`` and shared by the ACL engine and the
backtracker.  It is derived on demand and deterministic (architecture §10.1,
phase-1 default).

Edges:
    - ``step → step``    relation ``depends_on``   (from ``step.depends_on``)
    - ``step → artifact`` relation ``owns``         (from ``artifact.step_id``)
    - ``step → clue``    relation ``owns``          (from ``recovery[step_id]``)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from quro.core.resources.refs import ResourceRef

DEPENDS_ON = "depends_on"
OWNS = "owns"


@runtime_checkable
class IResourceGraph(Protocol):
    """Queryable resource edge table."""

    def nodes(self) -> set[ResourceRef]: ...

    def edges(self) -> list[tuple[ResourceRef, ResourceRef, str]]: ...

    def outgoing(self, node: ResourceRef, relation: str) -> set[ResourceRef]: ...

    def incoming(self, node: ResourceRef, relation: str) -> set[ResourceRef]: ...


class DomainStateGraph:
    """A deterministic edge table derived from a ``DomainState`` dict."""

    def __init__(self, state: dict[str, Any] | None = None) -> None:
        self._state: dict[str, Any] = state or {}
        self._nodes: set[ResourceRef] | None = None
        self._edges: list[tuple[ResourceRef, ResourceRef, str]] | None = None

    @classmethod
    def from_state(cls, state: dict[str, Any] | None) -> "DomainStateGraph":
        return cls(state)

    def _derive(self) -> None:
        if self._nodes is not None:
            return
        nodes: set[ResourceRef] = set()
        edges: list[tuple[ResourceRef, ResourceRef, str]] = []

        for s in self._state.get("steps", []):
            if not isinstance(s, dict):
                continue
            sid = s.get("step_id")
            if not sid:
                continue
            step_node = ResourceRef("step", str(sid))
            nodes.add(step_node)
            for dep in s.get("depends_on", []):
                if dep:
                    edges.append(
                        (step_node, ResourceRef("step", str(dep)), DEPENDS_ON)
                    )

        for a in self._state.get("artifacts", []):
            if not isinstance(a, dict):
                continue
            aid = a.get("artifact_id")
            if not aid:
                continue
            art_node = ResourceRef("artifact", str(aid))
            nodes.add(art_node)
            owner = a.get("step_id")
            if owner:
                edges.append(
                    (ResourceRef("step", str(owner)), art_node, OWNS)
                )

        for sid in (self._state.get("recovery") or {}):
            if not sid:
                continue
            clue_node = ResourceRef("clue", str(sid))
            nodes.add(clue_node)
            edges.append((ResourceRef("step", str(sid)), clue_node, OWNS))

        self._nodes = nodes
        self._edges = edges

    def nodes(self) -> set[ResourceRef]:
        self._derive()
        return set(self._nodes or ())

    def edges(self) -> list[tuple[ResourceRef, ResourceRef, str]]:
        self._derive()
        return list(self._edges or ())

    def outgoing(self, node: ResourceRef, relation: str) -> set[ResourceRef]:
        self._derive()
        return {
            dst for (src, dst, rel) in self._edges or ()
            if src == node and rel == relation
        }

    def incoming(self, node: ResourceRef, relation: str) -> set[ResourceRef]:
        self._derive()
        return {
            src for (src, dst, rel) in self._edges or ()
            if dst == node and rel == relation
        }

    def owner_of(self, ref: ResourceRef) -> ResourceRef | None:
        """Return the owning step node for an artifact/clue, or None."""
        incoming = self.incoming(ref, OWNS)
        return next(iter(incoming), None)
