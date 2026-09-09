"""StepType — a domain-defined step kind that declares a skill pool (Phase B).

A ``StepType`` is domain *content*: the kinds of step a domain's planner may
create.  It is the **single identity word** in the runtime — there is no
parallel ``role`` / ``capability`` concept (Phase 0 cleanup).  A step type
carries:

- ``skill_pool`` — the default skills it may load (``create_step(skills ⊆ pool)``),
- ``identity_block`` — the identity text projected as the step's system prompt
  (what used to be a ``Role.system_prompt``),
- ``executor_hints`` — execution configuration (e.g. the terminal tool that
  ends the step, or the initial phase).  The exact hint vocabulary is settled
  by the view stage; Phase 0 only threads the ``terminal_tool`` hint needed by
  the governor.
- ``features`` — the framework-owned step-level capabilities this type enables
  (a subset of the closed ``Feature`` enum in ``quro.core.features``).  A
  plain name-only field, like ``skill_pool`` — no decorator, no registry.

    create_step(step_type, objective, skills ⊆ pool)
        planner picks a bounded subset of the StepType's pool; it cannot
        name a skill that does not exist.

``StepTypeCatalog`` indexes the domain's StepTypes and renders a compact
block for planner-context injection.  It is pure data (no runtime deps).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from quro.core.domain.mount_ref import MountRef


@dataclass
class StepType:
    """A kind of step a domain's planner can create.

    Attributes:
        name: Unique step-type id (e.g. ``"implement"``).
        skill_pool: Default skill names this step kind may reference.  The
            planner may declare a subset via ``create_step(skills=[...])``.
        identity_block: The identity text projected as this step type's
            system prompt.  Empty means "no identity block" (the view layer
            falls back to a generic identity).
        executor_hints: Execution configuration keyed by hint name.  Phase 0
            recognizes ``terminal_tool`` (the tool that ends a step of this
            type) and ``phase`` (the initial session phase).  The full
            vocabulary is settled by the view stage.
        features: Framework-owned step-level capabilities (closed ``Feature``
            enum subset).
        tools: Capability types (or literal tool names) this step type is
            granted — external filesystem/shell grants declared as capability
            markers (``Readonly`` / ``Write`` / ``Shell``).  Mirrored onto
            ``StepSpec.tools`` at assembly.
        grant_name: Name of a dynamic grant callable registered in the
            ``GrantRegistry`` (``Callable[[StepSpec], ...]``).  The callable is
            resolved at evaluate time and is never serialized into ``StepConfig``
            (only this name is).  ``None`` = no dynamic grant.
        exclude: Capability types denied from the composed grant (``static ∪
            dynamic − exclude``).  Capability-typed; meaningful when inheriting
            a profile then removing one capability.
        mounted: Optional ``MountRef`` declaring that this step type's body is
            a foreign, frozen pipeline (reuse-mount, ``mount-semantics.md`` §5).
            ``None`` = an ordinary step (no mount).
        access: Artifact-access policy for this step type.  ``""`` = no
            cross-step artifact reads; ``"hints"`` = allow reading artifacts
            from ``depends_on`` predecessors via domain tools.
    """

    name: str
    skill_pool: list[str] = field(default_factory=list)
    identity_block: str = ""
    executor_hints: dict[str, Any] = field(default_factory=dict)
    features: tuple[str, ...] = field(default_factory=tuple)
    tools: tuple[Any, ...] = field(default_factory=tuple)
    grant_name: str | None = None
    exclude: tuple[Any, ...] = field(default_factory=tuple)
    mounted: MountRef | None = None
    access: str = ""

    def with_features(self, *names: str) -> "StepType":
        """Return a copy with *names* appended to ``features`` (no duplicates)."""
        merged = list(self.features)
        for n in names:
            if n not in merged:
                merged.append(n)
        return StepType(
            name=self.name,
            skill_pool=list(self.skill_pool),
            identity_block=self.identity_block,
            executor_hints=dict(self.executor_hints),
            features=tuple(merged),
            tools=tuple(self.tools),
            grant_name=self.grant_name,
            exclude=tuple(self.exclude),
            mounted=self.mounted,
            access=self.access,
        )


class StepTypeCatalog:
    """Index of a domain's StepTypes by name."""

    def __init__(self, step_types: Iterable[StepType] | None = None) -> None:
        self._types: dict[str, StepType] = {}
        for step_type in step_types or []:
            self._types[step_type.name] = step_type

    def get(self, name: str) -> StepType | None:
        """Return the StepType named *name*, or None."""
        return self._types.get(name)

    def has(self, name: str) -> bool:
        """Return True when *name* is a known StepType."""
        return name in self._types

    def names(self) -> list[str]:
        """Return all StepType names (insertion order)."""
        return list(self._types)

    def skill_pool_for(self, name: str) -> list[str]:
        """Return the skill pool for StepType *name* (empty when unknown)."""
        step_type = self._types.get(name)
        return list(step_type.skill_pool) if step_type else []

    def identity_block_for(self, name: str) -> str:
        """Return the identity block for StepType *name* (empty when unknown)."""
        step_type = self._types.get(name)
        return step_type.identity_block if step_type else ""

    def executor_hints_for(self, name: str) -> dict[str, Any]:
        """Return the executor hints for StepType *name* (empty when unknown)."""
        step_type = self._types.get(name)
        return dict(step_type.executor_hints) if step_type else {}

    def features_for(self, name: str) -> tuple[str, ...]:
        """Return the features for StepType *name* (empty when unknown)."""
        step_type = self._types.get(name)
        return tuple(step_type.features) if step_type else ()

    def tools_for(self, name: str) -> tuple[Any, ...]:
        """Return the tool grants (capabilities / names) for StepType *name*."""
        step_type = self._types.get(name)
        return tuple(step_type.tools) if step_type else ()

    def grant_name_for(self, name: str) -> str | None:
        """Return the dynamic-grant name for StepType *name* (``None`` when
        unknown or no dynamic grant)."""
        step_type = self._types.get(name)
        return step_type.grant_name if step_type else None

    def exclude_for(self, name: str) -> tuple[Any, ...]:
        """Return the excluded capabilities for StepType *name* (empty when
        unknown or no exclusions)."""
        step_type = self._types.get(name)
        return tuple(step_type.exclude) if step_type else ()

    def mounted_for(self, name: str) -> Any:
        """Return the ``MountRef`` for StepType *name* (``None`` when unknown
        or unmounted)."""
        step_type = self._types.get(name)
        return step_type.mounted if step_type else None

    def operator_summary(self, name: str) -> Any:
        """Return an ``OperatorSummary`` for StepType *name* (for IStepCatalogView).

        Returns ``None`` when the name is not in the catalog.
        """
        from quro.core.protocols import OperatorSummary

        step_type = self._types.get(name)
        if step_type is None:
            return None
        return OperatorSummary(
            name=step_type.name,
            description=step_type.identity_block or step_type.name,
            cost=1.0,
            skill_pool=tuple(step_type.skill_pool),
            access=step_type.access,
        )

    def describe_operators(self) -> list[Any]:
        """Return ``list[OperatorSummary]`` for all registered StepTypes.

        Implements the ``IStepCatalogView`` contract — each StepType is
        mapped to an ``OperatorSummary``.
        """
        from quro.core.protocols import OperatorSummary

        summaries: list[OperatorSummary] = []
        for step_type in self._types.values():
            summaries.append(OperatorSummary(
                name=step_type.name,
                description=step_type.identity_block or step_type.name,
                cost=1.0,
                skill_pool=tuple(step_type.skill_pool),
                access=step_type.access,
            ))
        return summaries

    def describe(self) -> str:
        """Render a compact catalog block for planner-context injection.

        One line per StepType: ``- <name> (skills=<pool>)``.  The identity
        block is shown only when declared.
        """
        lines: list[str] = []
        for step_type in self._types.values():
            pool = ", ".join(step_type.skill_pool) if step_type.skill_pool else "—"
            parts = [f"skills={pool}"]
            lines.append(
                f"- {step_type.name} " + " (" + ", ".join(parts) + ")"
            )
        return "\n".join(lines)
