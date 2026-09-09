"""Projection grants — the block-level exception (architecture §6).

The only phase-1 "compromise" form (design doc §9.4, form B).  A projection
grant is **not** an ACL exemption: it cannot turn a `deny` into an `allow`.
It only adds *block* projection once the `source` resource has already passed
the ACL check — "two doors, neither removed".

Grants are owned by the coordinator, reuse ``@user_block`` / ``@system_block``
names, fail fast on unknown block names, and aggregate into an auditable
``manifest()``.  ``auto_expanding`` defaults to off and is **not** built in
phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from quro.core.resources.refs import ResourceRef


@dataclass(frozen=True)
class ProjectionGrant:
    """A block-level projection exception for one StepType principal."""

    principal: str              # StepType ref (e.g. "deep_dive_symbol")
    blocks: tuple[str, ...]     # prompt-block names (reuse @user_block/@system_block)
    source: ResourceRef         # the extra resource to project — STILL ACL-gated
    reason: str                 # required, auditable


@runtime_checkable
class IProjectionGrant(Protocol):
    def grants_for(self, principal: str, phase: str) -> list[ProjectionGrant]: ...

    def manifest(self) -> list[ProjectionGrant]: ...


class ProjectionGrantRegistry:
    """A flat, ordered registry of declared projection grants."""

    def __init__(self, grants: list[ProjectionGrant] | tuple[ProjectionGrant, ...] = ()) -> None:
        self._grants: list[ProjectionGrant] = list(grants)

    def register(self, grant: ProjectionGrant) -> None:
        self._grants.append(grant)

    def grants_for(self, principal: str, phase: str = "") -> list[ProjectionGrant]:
        """Return the grants matching *principal* (phase reserved for later)."""
        return [g for g in self._grants if g.principal == principal]

    def manifest(self) -> list[ProjectionGrant]:
        """Return every declared grant, in declaration order (diffable)."""
        return list(self._grants)
