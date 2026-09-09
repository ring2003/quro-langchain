"""Capability types — the step-anchored tool-grant vocabulary.

A ``Capability`` is a **pure marker type** (zero data, zero behaviour) that
names *what a tool can do* (read files, write files, run a shell).  A
``StepType.tools`` / ``StepSpec.tools`` declaration names capabilities, not
tool names: the framework resolves a capability set into concrete tools via
the single ``CAPABILITY_TOOLS`` table below.

Design decisions (``tool-governance-step-anchored.md`` §4.3a):

- **Composable + ``isinstance``-able.**  ``Readonly | Write`` is a set of
  capabilities, and ``issubclass(Shell, Capability)`` reads naturally.  These
  are ordinary classes, not ``enum`` values, so they can be combined with the
  ``|`` operator and introspected by type.

- **Names only, serializable.**  A capability is known by its ``__name__``
  (``"Readonly"`` / ``"Write"`` / ``"Shell"``); ``StepConfig`` round-trips the
  *name*, and ``resolve_capability`` rebuilds the type.  No method, no state,
  no MRO to serialize.

- **The capability → tool-name map is a single central table**, not a property
  of the tools themselves.  A filesystem tool object is unchanged; it acquires
  its capability by appearing in ``CAPABILITY_TOOLS``.  This keeps the grant
  vocabulary declarative and auditable in one place.

This is deliberately a *marker* layer only.  ``exclude_tools`` and the dynamic
``tools_fn: Callable[[StepSpec], ...]`` grant land in a later batch — a
capability type does not yet carry ``exclude``.
"""

from __future__ import annotations

from typing import Any


class Capability:
    """Base marker for a step-tool capability (read / write / shell)."""

    __slots__ = ()


class Readonly(Capability):
    """Read-only filesystem access: ``read`` / ``ls`` / ``grep`` / ``find``."""

    __slots__ = ()


class Write(Capability):
    """Write access: ``write`` / ``edit`` (does not imply ``Readonly``)."""

    __slots__ = ()


class Shell(Capability):
    """Arbitrary code execution: ``shell`` (never folded into a safe default)."""

    __slots__ = ()


# capability type → concrete tool names.  Single authority for "what tools does
# this capability grant".  ``Readonly | Write`` is the union of the two sets.
CAPABILITY_TOOLS: dict[type[Capability], tuple[str, ...]] = {
    Readonly: ("read", "ls", "grep", "find"),
    Write: ("write", "edit"),
    Shell: ("shell",),
}

# capability name → type (for StepConfig serialization round-trips).
CAPABILITY_BY_NAME: dict[str, type[Capability]] = {
    cls.__name__: cls for cls in CAPABILITY_TOOLS
}


def is_capability(value: Any) -> bool:
    """Return True when *value* is a capability type."""
    return isinstance(value, type) and issubclass(value, Capability)


def tools_for_capability(value: Any) -> tuple[str, ...]:
    """Resolve a capability (type) or a plain tool name to tool names.

    A capability type resolves through ``CAPABILITY_TOOLS``; an unknown value
    (a literal tool name, kept for backward compatibility with the old
    ``StepSpec.tools`` string lists) resolves to itself.
    """
    if is_capability(value):
        return CAPABILITY_TOOLS[value]
    if isinstance(value, str):
        return (value,)
    return ()


def tools_for_capabilities(values: Any) -> tuple[str, ...]:
    """Resolve a capability set (or a single capability) to a deduped tool list.

    Accepts a single capability type, a ``frozenset``/``set``/``tuple``/``list``
    of capability types, or a plain string tool name.
    """
    if is_capability(values) or isinstance(values, str):
        return tools_for_capability(values)

    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        for name in tools_for_capability(value):
            if name not in seen:
                seen.add(name)
                result.append(name)
    return tuple(result)


def resolve_capability(name: str) -> type[Capability] | None:
    """Return the capability type for *name*, or ``None`` when unknown."""
    return CAPABILITY_BY_NAME.get(name)
