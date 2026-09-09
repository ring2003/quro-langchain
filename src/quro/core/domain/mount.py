"""reuse-mount — the execution consumer of ``sub_steps`` (mount-semantics.md).

This module lands the **static, frozen** half of "mount": flatten a foreign,
frozen ``[StepConfig]`` sequence into a single step (the §5 execution chain).
It never executes — it checks (``check_mount``) and resolves (``MountResolver``);
the actual flatten + fold happens in ``PipelineRunner`` (§5 step 4/5).

The stage-one constraints (§3.3) are enforced by :func:`check_mount` as a
**static traversal + assertion over frozen data** — not a type-inference pass
and not a dry-run (§4):

- **depth ≤ 1** — the mounted unit does not itself mount (no recursive
  ``sub_steps`` mount).
- **signature compat** — the outer step's ``inputs`` / ``depends_on`` must be
  satisfiable by the inner unit's ``expected_output`` (declared cross-boundary
  extension of ``Pipeline.validate()``).
- **closure** — the terminal step produces an ``Artifact`` (non-empty
  ``expected_output``).

The behavioral backstop ("a step returns no artifact") stays with
``RecoveryHook`` (unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from quro.core.domain.mount_ref import MountRef
from quro.core.domain.step_config import StepConfig


@dataclass
class MountCheckResult:
    """The outcome of ``check_mount`` — errors, or the resolvable ref."""

    ref: MountRef | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def check_mount(
    configs: list[StepConfig],
    outer: Any,
) -> list[str]:
    """Return the mount contract violations for a frozen ``[StepConfig]`` unit.

    ``outer`` is a ``StepSpec`` (or any object exposing ``depends_on`` /
    ``inputs`` / ``expected_output`` / ``mounted`` attributes).  The check is
    structural only — it inspects frozen ``StepConfig`` data (§4).

    The constraints checked (mount-semantics.md §3.3 / §4):

    1. **depth ≤ 1** — no ``mounted`` config inside the unit (no recursive
       mount); no nested ``sub_steps`` carrying their own mount.
    2. **closure** — the terminal step (no downstream dependents) must produce
       an ``Artifact``: it has a non-empty ``expected_output`` (the artifact
       descriptor).
    3. **signature compat** — outer ``depends_on`` / ``inputs`` must be
       satisfiable by the inner unit's ``expected_output`` descriptors (a
       declared, cross-boundary extension of ``Pipeline.validate()``).

    Returns an empty list when the mount is valid.
    """
    errors: list[str] = []

    if not configs:
        return ["mount unit is empty — no StepConfig to flatten"]

    # 1. depth ≤ 1: no recursive mount inside the mounted unit.
    for c in configs:
        if c.mounted is not None:
            errors.append(
                f"mounted step '{c.id}' declares its own mount "
                f"({c.mounted.domain}:{c.mounted.pipeline}) — depth > 1"
            )
        for sub in c.sub_steps:
            if sub.mounted is not None:
                errors.append(
                    f"sub_step '{sub.id}' under '{c.id}' declares a mount — "
                    "depth > 1"
                )

    # 2. closure: the terminal step produces an Artifact.
    terminal = _terminal_steps(configs)
    for c in terminal:
        if not (c.expected_output or "").strip():
            errors.append(
                f"terminal step '{c.id}' has no expected_output — the mounted "
                "unit must close into an Artifact"
            )

    # 3. signature compat: outer inputs/depends_on vs inner expected_output.
    inner_outputs = {
        _norm(c.expected_output)
        for c in terminal
        if (c.expected_output or "").strip()
    }
    for dep in getattr(outer, "depends_on", []) or []:
        if not inner_outputs:
            errors.append(
                f"outer step depends on '{dep}' but the mounted unit declares "
                "no expected_output to satisfy it"
            )
            continue
    for inp in getattr(outer, "inputs", []) or []:
        # ``inputs`` name artifacts the step expects; the mounted unit's
        # terminal ``expected_output`` must cover them by name.
        if inner_outputs and not _covers(inp, inner_outputs):
            errors.append(
                f"outer input '{inp}' is not covered by the mounted unit's "
                f"expected_output {sorted(inner_outputs)}"
            )

    return errors


@runtime_checkable
class IMountResolver(Protocol):
    """Resolve a ``MountRef`` into a frozen ``[StepConfig]`` sequence."""

    def resolve(self, ref: MountRef) -> list[StepConfig]:
        """Return the foreign pipeline's frozen ``[StepConfig]`` list."""
        ...


class MountResolver:
    """A name-addressed registry of foreign frozen pipelines (domain, name)."""

    def __init__(self, registry: dict[tuple[str, str], list[StepConfig]] | None = None) -> None:
        self._registry: dict[tuple[str, str], list[StepConfig]] = dict(registry or {})

    def register(
        self,
        domain: str,
        pipeline: str,
        configs: list[StepConfig],
    ) -> None:
        """Register a foreign frozen ``[StepConfig]`` under ``(domain, name)``."""
        self._registry[(domain, pipeline)] = list(configs)

    def resolve(self, ref: MountRef) -> list[StepConfig]:
        """Return the registered ``[StepConfig]`` for *ref*.

        Raises:
            KeyError: When ``ref`` is not registered (unknown mount target).
        """
        key = (ref.domain, ref.pipeline)
        if key not in self._registry:
            raise KeyError(
                f"unknown mount target {ref.domain}:{ref.pipeline} — "
                f"registered: {sorted(self._registry) or '(none)'}"
            )
        return list(self._registry[key])


def _terminal_steps(configs: list[StepConfig]) -> list[StepConfig]:
    """Return the configs no other config depends on (the fold points)."""
    ids = {c.id for c in configs}
    dependencies = {d for c in configs for d in c.depends_on}
    return [c for c in configs if c.id not in dependencies]


def _norm(value: Any) -> str:
    """Normalize an expected_output descriptor to comparable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(str(v).strip() for v in value if str(v).strip())
    return str(value)


def _covers(need: str, available: set[str]) -> bool:
    """Return True when *need* is covered by *available* (name match).

    A coarse declared check: exact token presence.  It does **not** parse
    natural language — it only catches obviously-missing wiring (§4 "declared
    check, not a type-inference pass").
    """
    need = (need or "").strip()
    if not need:
        return True
    if need in available:
        return True
    need_tokens = set(need.lower().split())
    for a in available:
        if need_tokens <= set(a.lower().split()):
            return True
    return False
