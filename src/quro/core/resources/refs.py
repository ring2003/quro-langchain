"""Logical resource descriptors (architecture §3).

A descriptor is a **query predicate**, not an entity: it says *where* a
resource lives, in a uniform vocabulary independent of physical storage.
Phase 1 covers the internal-state resources of ``DomainState``:

    step://s1            artifact://art_ab12cd34
    clue://s1/clues      round://3/step/s1
    plan://main          interpretation://main

The grammar is ``<kind>://<id>[/<subpath>…]``.  Invalid syntax raises
:class:`InvalidDescriptor`; the resolver (``resolver.py``) turns a parsed
ref into data.
"""

from __future__ import annotations

from dataclasses import dataclass

# Phase-1 resource kinds (architecture §3 / design doc §9.5).
RESOURCE_KINDS: frozenset[str] = frozenset({
    "step",
    "artifact",
    "clue",
    "round",
    "plan",
    "interpretation",
})


class InvalidDescriptor(ValueError):
    """Raised when a descriptor string does not parse."""


@dataclass(frozen=True)
class ResourceRef:
    """A parsed logical descriptor.

    Attributes:
        kind: ``step`` / ``artifact`` / ``clue`` / ``round`` / ``plan`` /
            ``interpretation``.
        id: The resource id (e.g. ``"s1"``, ``"art_ab12cd34"``, ``"main"``).
        subpath: Optional path segments below the id (e.g. ``("clues",)``).
    """

    kind: str
    id: str
    subpath: tuple[str, ...] = ()

    def __str__(self) -> str:
        base = f"{self.kind}://{self.id}"
        if self.subpath:
            base += "/" + "/".join(self.subpath)
        return base

    def __repr__(self) -> str:
        return f"ResourceRef({self.kind!r}, {self.id!r}, {self.subpath!r})"


def parse_descriptor(s: str) -> ResourceRef:
    """Parse ``<kind>://<id>[/<subpath>…]`` into a :class:`ResourceRef`.

    Raises :class:`InvalidDescriptor` on bad syntax: a non-string, a missing
    ``://`` separator, an unknown kind, or an empty id.
    """
    if not isinstance(s, str):
        raise InvalidDescriptor(
            f"descriptor must be a string, got {type(s).__name__}"
        )
    if "://" not in s:
        raise InvalidDescriptor(f"invalid descriptor {s!r}: missing '://'")
    kind, _, rest = s.partition("://")
    if kind not in RESOURCE_KINDS:
        raise InvalidDescriptor(
            f"invalid descriptor {s!r}: unknown kind {kind!r}"
        )
    if not rest:
        raise InvalidDescriptor(f"invalid descriptor {s!r}: empty id")
    segments = rest.split("/")
    rid = segments[0]
    if not rid:
        raise InvalidDescriptor(f"invalid descriptor {s!r}: empty id")
    subpath = tuple(segments[1:]) if len(segments) > 1 else ()
    return ResourceRef(kind, rid, subpath)
