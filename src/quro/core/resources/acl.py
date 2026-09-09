"""ACL engine — the Layer 2 data surface (architecture §5).

Having the tool ≠ reading *this* id.  ``IAclEngine`` is checked **at call
time**, before a read/write handler reaches the kernel.  Default is deny; deny
wins over every allow (design doc §9.3).

Phase-1 grants:
    - ``ownership`` — a step reads/writes its own step entry and its own
      artifacts / clues.
    - ``dependency`` — a step reads upstream artifacts/clues where
      ``upstream ∈ step.depends_on`` and ``step.access == "hints"``.
    - ``capability`` — the ``StepType.tools`` tool allocation (Layer 1).
    - ``system`` — runtime components (deferred).

Phase-1 read rules (architecture §5):
    - ``get_step(s)`` — allow iff ``s == principal.step_id`` or
      ``s ∈ depends_on``.
    - ``get_artifact(a)`` — allow iff ``owner(a) == principal.step_id`` or
      (``owner(a) ∈ depends_on`` and ``access == "hints"``).
    - ``get_clue`` — phase 1 grants clue read to the owning step only.
    - ``plan`` / ``interpretation`` / ``round`` — denied to step principals
      (Layer 1 already gates the writer tools to the planner).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from quro.core.resources.graph import IResourceGraph
from quro.core.resources.refs import ResourceRef


@dataclass(frozen=True)
class Principal:
    """The requestor — a step instance (design doc §9.1)."""

    step_id: str
    step_type: str = ""
    capabilities: frozenset[str] = frozenset()


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision is Decision.ALLOW


def allow(reason: str = "") -> Verdict:
    return Verdict(Decision.ALLOW, reason)


def deny(reason: str = "") -> Verdict:
    return Verdict(Decision.DENY, reason)


@runtime_checkable
class IAclEngine(Protocol):
    def check(self, principal: Principal, ref: ResourceRef, action: str) -> Verdict: ...


def principal_from_state(state: dict[str, Any] | None) -> Principal:
    """Derive the current step-instance principal from ``DomainState``."""
    state = state or {}
    step_id = str(state.get("executing_step_id") or "")
    step_type = ""
    for s in state.get("steps", []):
        if isinstance(s, dict) and s.get("step_id") == step_id:
            step_type = str(s.get("step_type") or "")
            break
    return Principal(step_id=step_id, step_type=step_type)


def _step_field(state: dict[str, Any], step_id: str, field: str) -> Any:
    for s in state.get("steps", []):
        if isinstance(s, dict) and s.get("step_id") == step_id:
            return s.get(field)
    return None


class DefaultAclEngine:
    """Default-deny ACL over a resource graph + the step metadata in state."""

    def __init__(
        self,
        graph: IResourceGraph,
        state: dict[str, Any] | None = None,
    ) -> None:
        self._graph = graph
        self._state: dict[str, Any] = state or {}

    def _depends_on(self, step_id: str) -> set[str]:
        return {
            ref.id for ref in self._graph.outgoing(
                ResourceRef("step", step_id), "depends_on"
            )
        }

    def _access(self, step_id: str) -> str | None:
        return _step_field(self._state, step_id, "access")

    def check(self, principal: Principal, ref: ResourceRef, action: str) -> Verdict:
        # Default-deny must hold even with an unset executing step: an empty
        # principal id must never match an empty resource id.
        if not principal.step_id or not ref.id:
            return deny("no executing step — cannot resolve a principal")
        if action in ("write", "edit", "append", "delete"):
            return self._check_write(principal, ref)
        return self._check_read(principal, ref)

    # -- read -------------------------------------------------------------

    def _check_read(self, principal: Principal, ref: ResourceRef) -> Verdict:
        sid = principal.step_id
        if ref.kind == "step":
            if ref.id == sid:
                return allow("own step")
            if ref.id in self._depends_on(sid):
                return allow("dependency step")
            return deny(
                f"step '{ref.id}' is neither the current step nor a dependency"
            )
        if ref.kind == "artifact":
            owner = self._graph.owner_of(ref)
            if owner is None:
                return deny(f"artifact '{ref.id}' has no owning step")
            if owner.id == sid:
                return allow("own artifact")
            if owner.id in self._depends_on(sid) and self._access(sid) == "hints":
                return allow("dependency artifact (access='hints')")
            return deny(f"artifact '{ref.id}' belongs to a non-dependency step")
        if ref.kind == "clue":
            if ref.id == sid:
                return allow("own clue")
            return deny(f"clue '{ref.id}' is not owned by the current step")
        # plan / interpretation / round — no step-level read in phase 1.
        return deny(
            f"{ref.kind}:// reads are denied to step principals (phase 1)"
        )

    # -- write ------------------------------------------------------------

    def _check_write(self, principal: Principal, ref: ResourceRef) -> Verdict:
        # A step may only write to its own step entry (its artifacts/clues).
        if ref.kind == "step" and ref.id == principal.step_id:
            return allow("own step")
        return deny(f"write to {ref} denied: not the current step")


def acl_for_state(state: dict[str, Any] | None) -> DefaultAclEngine:
    """Build a default ACL engine over the graph + metadata of *state*."""
    from quro.core.resources.graph import DomainStateGraph

    state = state or {}
    return DefaultAclEngine(DomainStateGraph.from_state(state), state)
