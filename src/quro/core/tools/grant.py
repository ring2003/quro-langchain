"""Tool grant profile and dynamic-grant registry (step-anchored authorization).

This module carries the *authorization semantics* that used to be scattered
across the coordinator.  It separates the two halves of a tool grant:

- **data** — ``ToolProfile``, a pure, frozen, capability-typed value the domain
  author declares (``tools`` / ``exclude`` / ``grant_name``).
- **function** — ``GrantRegistry``, the name → callable index for the *dynamic*
  grant (``grant_name``), exactly parallel to ``HookRegistry``.

The callable is **never serialized**: a pure ``f(step)`` function's output is
uniquely determined by its ``StepSpec`` input, so serialization persists only
the ``grant_name`` string and rebuild resolves it via ``GrantRegistry.create``.

Design (``tool-governance-step-anchored.md`` §4.3 / §4.3a / §4.5):

- ``ToolProfile`` is **pure data** — no methods carry behaviour, no executor
  hints, no MRO.  Reuse is by composition and derivation (``.excluding``).
- ``exclude`` is **capability-typed** and applied against the final capability
  set (``static ∪ dynamic − exclude``).  A bare ``exclude`` that never matches
  the set is a no-op.
- ``GrantRegistry`` mirrors ``HookRegistry`` (``register`` / ``create``), so a
  domain author registers a grant function by name once and each ``StepType``
  references it via ``grant_name``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, TYPE_CHECKING

from quro.core.tools.capability import (
    Capability,
    resolve_capability,
    tools_for_capabilities,
)

if TYPE_CHECKING:
    from quro.steps.core import StepSpec


# ---------------------------------------------------------------------------
# ToolProfile — the pure, capability-typed grant value
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolProfile:
    """A pure, capability-typed tool-grant declaration.

    Fields:
        tools: Capability types granted statically (``Readonly | Shell``).
        exclude: Capability types denied from the composed grant
            (``static ∪ dynamic − exclude``).
        grant_name: Name of a dynamic grant callable registered in the
            ``GrantRegistry`` (``Callable[[StepSpec], ...]``, pure over the
            step).  ``None`` = no dynamic grant.

    This is a **data value**, not a class with behaviour: it carries no
    callable, no executor hints, and no MRO.  Reuse is by composition and
    derivation (:meth:`excluding`), which *bakes* the exclusion into a new
    value so a derived profile carries no pending ``exclude`` state.
    """

    tools: tuple[type[Capability], ...] = ()
    exclude: tuple[type[Capability], ...] = ()
    grant_name: str | None = None

    def excluding(self, *capabilities: type[Capability]) -> "ToolProfile":
        """Return a new profile with *capabilities* removed (derivation).

        The exclusion is **baked into ``tools``**: the returned value has an
        empty ``exclude`` field and a reduced ``tools`` set, so a derived
        profile is a clean, fully-resolved value (no deferred deny).
        """
        removed = set(capabilities)
        bake = tuple(c for c in self.tools if c not in removed)
        return replace(self, tools=bake, exclude=())

    def without(self, *capabilities: type[Capability]) -> "ToolProfile":
        """Alias for :meth:`excluding` (author-facing, derivation-friendly)."""
        return self.excluding(*capabilities)


# ---------------------------------------------------------------------------
# GrantRegistry — the name → callable index (mirrors HookRegistry)
# ---------------------------------------------------------------------------


class GrantRegistry:
    """Registry of named dynamic-grant callables.

    A grant function takes a ``StepSpec`` and returns an iterable of capability
    types (never tool names).  Registering by name decouples the callable from
    ``StepType`` (which stays pure data) and gives the callable a serializable
    identity — the ``grant_name`` string.

    Usage::

        registry = GrantRegistry()
        registry.register("researcher_ro", lambda step: (
            (Readonly, Shell) if step.features.get("needs_shell") else (Readonly,)
        ))

        fn = registry.create("researcher_ro")     # the callable
        caps = registry.eval("researcher_ro", step)  # tuple[Capability, ...]
    """

    def __init__(self) -> None:
        self._grants: dict[str, Callable[[StepSpec], Iterable[type[Capability]]]] = {}

    def register(
        self,
        name: str,
        fn: Callable[[StepSpec], Iterable[type[Capability]]],
    ) -> None:
        """Register a grant callable under *name* (overwrite warns + replaces)."""
        self._grants[name] = fn

    def create(self, name: str) -> Callable[[StepSpec], Iterable[type[Capability]]]:
        """Return the grant callable for *name*.

        Raises:
            KeyError: If *name* is not registered.
        """
        fn = self._grants.get(name)
        if fn is None:
            raise KeyError(f"Unknown grant: '{name}'. Registered: {sorted(self._grants)}")
        return fn

    def eval(self, name: str, step: "StepSpec") -> tuple[type[Capability], ...]:
        """Resolve *name* against *step* into a capability tuple.

        The grant callable is pure over ``StepSpec``; its output is normalized
        to a deduped tuple of capability types, dropping any non-capability
        values (defensive against an author returning a tool-name string).
        """
        raw = self.create(name)(step)
        result: list[type[Capability]] = []
        seen: set[type[Capability]] = set()
        for value in raw:
            if isinstance(value, type) and issubclass(value, Capability):
                if value not in seen:
                    seen.add(value)
                    result.append(value)
        return tuple(result)

    def names(self) -> list[str]:
        """Return all registered grant names (insertion order)."""
        return list(self._grants)


# ---------------------------------------------------------------------------
# evaluate_grant — the pure authorization function
# ---------------------------------------------------------------------------


def evaluate_grant(
    profile: ToolProfile,
    step: "StepSpec",
    registry: GrantRegistry | None = None,
) -> tuple[type[Capability], ...]:
    """Resolve *profile* against *step* into a final capability tuple.

    The authorization chain, collapsed into one pure function (so it is unit
    testable and deterministic on rebuild):

        static  = profile.tools
        dynamic = registry.eval(profile.grant_name, step)  (when grant_name set)
        result  = (static ∪ dynamic) − profile.exclude

    Returns a deduped tuple of capability types.  No tool-name resolution
    happens here — that is the coordinator's (or a caller's) final
    ``tools_for_capabilities`` step.
    """
    registry = registry if registry is not None else get_grant_registry()

    static = set(profile.tools)
    dynamic: set[type[Capability]] = set()
    if profile.grant_name is not None:
        dynamic = set(registry.eval(profile.grant_name, step))

    composed = static | dynamic
    denied = set(profile.exclude)
    result = composed - denied

    # Preserve a stable, declaration-first order: static tools, then dynamic,
    # minus excluded.
    ordered: list[type[Capability]] = []
    for value in (*profile.tools, *dynamic):
        if value in denied:
            continue
        if value not in ordered:
            ordered.append(value)
    return tuple(ordered)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def capabilities_to_names(values: Iterable[Any]) -> list[str]:
    """Serialize an iterable of capability types / names to name strings.

    A capability type becomes its ``__name__`` (``"Readonly"``); a plain string
    (literal tool name, backward-compatible) round-trips as itself.  Unknown
    values are dropped.
    """
    result: list[str] = []
    for value in values:
        if isinstance(value, type) and issubclass(value, Capability):
            result.append(value.__name__)
        elif isinstance(value, str):
            result.append(value)
    return result


def resolve_capabilities(names: Iterable[str]) -> tuple[Any, ...]:
    """Rebuild capability types / literal names from name strings (round-trip).

    A name in ``CAPABILITY_BY_NAME`` becomes its capability type; an unknown
    name (a literal tool name kept for backward compatibility, e.g.
    ``"commit_clue"``) is passed through as-is so serialization stays
    symmetric.  This mirrors the historical ``StepSpec.tools`` "capability or
    literal name" union.
    """
    result: list[Any] = []
    for name in names:
        cap = resolve_capability(name)
        result.append(cap if cap is not None else name)
    return tuple(result)


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_default_grant_registry: GrantRegistry | None = None


def get_grant_registry() -> GrantRegistry:
    """Return the global (singleton) ``GrantRegistry``."""
    global _default_grant_registry
    if _default_grant_registry is None:
        _default_grant_registry = GrantRegistry()
    return _default_grant_registry
