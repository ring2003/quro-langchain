"""StepBuilder — IStepBuilder implementation.

Single step-construction path that wraps StepTypeCatalog and provides:
- Eager resolution of features, tools, access from StepType
- Structural dependency validation (same-round via scope,
  cross-round via ICrossRoundHandoff)
- Skill-pool membership check (existing behavior, preserved)

Phase 2 deliverable (architecture §3.1).
"""

from __future__ import annotations

import re
import time
from typing import Any, Sequence

_CROSS_ROUND_REF_RE = re.compile(r"^r\d+:")

from quro.core.domain.step_type import StepTypeCatalog
from quro.core.protocols import (
    AuditEvent,
    DependencyCheck,
    IAuditLog,
    ICrossRoundHandoff,
    StepBuildResult,
)


class StepBuilder:
    """Single step-construction path — IStepBuilder implementation.

    Wraps StepTypeCatalog and provides:
    - Eager resolution of features, tools, access from StepType
    - Structural dependency validation (same-round via scope,
      cross-round via ICrossRoundHandoff)
    - Skill-pool membership check (existing behavior, preserved)
    """

    def __init__(
        self,
        step_types: StepTypeCatalog,
        scope: Any | None = None,
        handoff: ICrossRoundHandoff | None = None,
        audit_log: IAuditLog | None = None,
    ) -> None:
        self._step_types = step_types
        self._scope = scope
        self._handoff = handoff
        self._audit_log = audit_log
        self._pending: dict[str, StepBuildResult] = {}

    def create(
        self,
        step_id: str,
        objective: str,
        step_type: str = "",
        depends_on: Sequence[str] = (),
        skills: Sequence[str] = (),
        expected_output: str = "",
        validation: str = "",
    ) -> StepBuildResult:
        """Create a step with eager resolution and dependency validation."""
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Validate step_type exists in StepTypeCatalog.
        if step_type and not self._step_types.has(step_type):
            errors.append(
                f"unknown step_type '{step_type}'. "
                f"Available: {self._step_types.names()}"
            )

        # 2. Validate skills ⊆ step_type.skill_pool.
        if step_type and skills:
            pool = self._step_types.skill_pool_for(step_type)
            unknown = [s for s in skills if s not in pool]
            if unknown:
                errors.append(
                    f"skill(s) {unknown} not in pool for "
                    f"'{step_type}'. Pool: {pool}"
                )

        # 3. Validate each depends_on entry.
        for dep in depends_on:
            if _CROSS_ROUND_REF_RE.match(dep):
                # Cross-round reference: "r<N>:<step_id>"
                if self._handoff is not None:
                    ref = self._handoff.resolve_step_ref(dep)
                    if ref is None:
                        errors.append(f"cross-round ref '{dep}' not found")
                else:
                    errors.append(
                        f"cross-round ref '{dep}' requires ICrossRoundHandoff"
                    )
            else:
                # Same-round reference
                if self._scope is not None and not self._scope.has(dep):
                    errors.append(
                        f"same-round dep '{dep}' not found in round scope"
                    )

        # 4. Check for duplicate step_id.
        if step_id in self._pending:
            errors.append(f"step '{step_id}' already exists. Use a different id.")

        # 5. Eagerly resolve features/tools/access from StepTypeCatalog.
        resolved_features: tuple[str, ...] = ()
        resolved_tools: tuple[Any, ...] = ()
        resolved_access: str = ""

        if step_type:
            st = self._step_types.get(step_type)
            if st is not None:
                resolved_features = tuple(st.features)
                resolved_tools = tuple(st.tools)
                resolved_access = st.access
                if not skills:
                    warnings.append(
                        f"no skills specified; defaulting to pool {st.skill_pool}"
                    )

        # 6. Build the StepSpec.
        from quro.steps.core import StepSpec

        spec = StepSpec(
            id=step_id,
            step_type=step_type or "",
            objective=objective,
            depends_on=list(depends_on),
            skills=list(skills),
            expected_output=expected_output,
            validation=validation,
        )

        result = StepBuildResult(
            spec=spec,
            resolved_features=resolved_features,
            resolved_tools=resolved_tools,
            resolved_access=resolved_access,
            dependency_check=DependencyCheck(ok=len(errors) == 0, errors=tuple(errors)),
            warnings=tuple(warnings),
        )

        # 7. Store if valid.
        if not errors:
            self._pending[step_id] = result

        # 8. Emit AuditEvent if audit_log provided.
        if self._audit_log is not None:
            self._audit_log.emit(AuditEvent(
                round_idx=-1,  # will be set by RoundController
                kind="step_built",
                payload={
                    "step_id": step_id,
                    "step_type": step_type,
                    "ok": len(errors) == 0,
                    "errors": errors,
                },
                ts=time.time(),
            ))

        return result

    def update(self, step_id: str, **fields: Any) -> StepBuildResult:
        """Update an existing step (returns new StepBuildResult — immutable pattern)."""
        existing = self._pending.get(step_id)
        if existing is None:
            return StepBuildResult(
                spec=None,
                resolved_features=(),
                resolved_tools=(),
                resolved_access="",
                dependency_check=DependencyCheck(
                    ok=False, errors=(f"step '{step_id}' not found",)
                ),
            )

        spec = existing.spec
        if "objective" in fields:
            spec.objective = fields["objective"]
        if "depends_on" in fields:
            spec.depends_on = fields["depends_on"]
        if "expected_output" in fields:
            spec.expected_output = fields["expected_output"]
        if "validation" in fields:
            spec.validation = fields["validation"]

        result = StepBuildResult(
            spec=spec,
            resolved_features=existing.resolved_features,
            resolved_tools=existing.resolved_tools,
            resolved_access=existing.resolved_access,
            dependency_check=DependencyCheck(ok=True),
            warnings=existing.warnings,
        )
        self._pending[step_id] = result
        return result

    def list_pending(self) -> list[StepBuildResult]:
        """Return all pending (not yet executed) step build results."""
        return list(self._pending.values())

    def clear(self) -> None:
        """Clear all pending steps (called after execution)."""
        self._pending.clear()
