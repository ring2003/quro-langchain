"""Feature — the closed set of framework-owned step-level capabilities.

A feature is a serializable **toggle** for a step-level capability: a plain
identifier (a ``str``-based ``Enum`` member) that a domain declares on a
``StepType`` and that flows name-only through the runtime.  A feature carries
**no behavior** — the ``feature → effect`` binding is an implicit contract
owned by the runtime consumer, never by the feature itself.

This enum is the single authority for "which step-level capabilities exist".
It is a **closed set**: the framework owns it; a domain references members, it
never invents them.  Domain-specific capabilities are expressed through the
step type / toolsets / parameters, not through features.

Layered defaults:

- ``RECOVERY``  — opt-in:  on step failure, trigger the resume hook.
- ``HIL``       — opt-in:  human-in-the-loop (migration target for
  ``StepSpec.hil``; not yet effective).
- ``CHECKPOINT`` — always-on: step-pre snapshot (framework default, not
  declared by a domain).

See ``docs/unsat-policy-loop/architecture/feature-gate.md`` and
``unified-resource-layer-phase4-implementation-plan.md`` §2.5.
"""

from __future__ import annotations

from enum import Enum


class Feature(str, Enum):
    """Framework-owned step-level capabilities (serializable toggles)."""

    RECOVERY = "recovery"      # opt-in: on step failure, trigger resume
    HIL = "hil"                # opt-in: human-in-the-loop (migration target)
    CHECKPOINT = "checkpoint"  # always-on: framework default, not declared
